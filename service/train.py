"""Обучение и оценка рекомендательной модели (метрика Precision@3), сохранение артефактов для сервиса.

Запуск из корня проекта:  python service/train.py
Читает: events.csv, features/{item,user}_features.pkl, features/meta.pkl
Пишет:  service/artifacts/*, service/docs/experiments.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.features import FEATURES, ITEM_COLS, USER_COLS, build_ui_tables, make_features  # noqa: E402

ART = ROOT / 'service' / 'artifacts'
DOCS = ROOT / 'service' / 'docs'
ART.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)

K = 3                # три слота на главной странице
POP_N = 40           # размер каждого популярного пула кандидатов
N_NEG_USERS = 5000   # пользователи без покупок в test: добавляют отрицательные примеры
SEED = 42
rng = np.random.default_rng(SEED)


def precision_at_k(df, score, eval_users, k=K):
    """Средняя по eval_users (все пользователи с >=1 покупкой/корзиной в test) доля релевантных среди top-k.
    Пользователи, чьи покупки не попали в пул кандидатов, считаются с нулём, а не отбрасываются."""
    d = df[['visitorid', 'target']].assign(s=score)
    d = d.sort_values(['visitorid', 's'], ascending=[True, False])
    d['rank'] = d.groupby('visitorid').cumcount()
    hits = d[d['rank'] < k].groupby('visitorid')['target'].sum()
    return float(hits.reindex(eval_users).fillna(0).mean() / k)


# ---------------------------------------------------------------- данные
t0 = time.time()
cutoff = pd.read_pickle(ROOT / 'features' / 'meta.pkl')['cutoff']
events = pd.read_csv(ROOT / 'events.csv', dtype={'visitorid': 'int32', 'itemid': 'int32', 'event': 'category'})
train, test = events[events['timestamp'] < cutoff], events[events['timestamp'] >= cutoff]

items = pd.read_pickle(ROOT / 'features' / 'item_features.pkl').set_index('itemid')[ITEM_COLS].astype('float32')
users = pd.read_pickle(ROOT / 'features' / 'user_features.pkl').set_index('visitorid')[USER_COLS].astype('float32')
ui, ucat = build_ui_tables(train, cutoff, items['categoryid'])
ucat.index = ucat.index.set_levels(ucat.index.levels[1].astype('float32'), level=1)

pos = test[test['event'].isin(['addtocart', 'transaction'])][['visitorid', 'itemid']].drop_duplicates().assign(target=1)
print(f'данные загружены за {time.time() - t0:.0f}с | положительных пар в test: {len(pos)}, пользователей: {pos.visitorid.nunique()}')

# ---------------------------------------------------------------- кандидаты
# популярные товары: по корзинам+покупкам и по активности за последнюю неделю перед отсечкой
pop_cart = (items['item_carts'] + items['item_buys']).nlargest(POP_N).index
pop_trend = items['item_events_last7d'].nlargest(POP_N).index
pool = pd.Index(pop_cart.union(pop_trend))

pos_users = pos['visitorid'].unique()
other = train['visitorid'].drop_duplicates()
other = other[~other.isin(pos_users)]
sample_users = np.concatenate([pos_users, rng.choice(other.values, N_NEG_USERS, replace=False)])

hist = ui.reset_index()[['visitorid', 'itemid']]
hist = hist[hist['visitorid'].isin(sample_users)]
popular = pd.DataFrame({'itemid': pool.values}).merge(pd.DataFrame({'visitorid': sample_users}), how='cross')
pairs = pd.concat([hist, popular[['visitorid', 'itemid']]]).drop_duplicates().reset_index(drop=True)
pairs = pairs.merge(pos, on=['visitorid', 'itemid'], how='left')
pairs['target'] = pairs['target'].fillna(0).astype('int8')
X = make_features(pairs, items, users, ui, ucat)
X['target'] = pairs['target'].values
print(f'кандидатов: {len(X)}, положительных: {int(X.target.sum())} '
      f'(из {len(pos)} пар test; остальные вне пула кандидатов)')

# ---------------------------------------------------------------- оценка
experiments = []


def log(name, params, p3, extra=None):
    row = {'model': name, 'params': params, 'precision_at_3': round(p3, 5), **(extra or {})}
    experiments.append(row)
    print(f'{name:<28} P@3 = {p3:.5f}  {params}')


eval_all = pos_users
eval_hist = pos_users[np.isin(pos_users, users.index.values)]
eval_cold = pos_users[~np.isin(pos_users, users.index.values)]
print(f'eval-пользователей: {len(eval_all)} (с историей {len(eval_hist)}, холодных {len(eval_cold)})')

# --- бейзлайны (без обучения)
log('baseline: popular carts+buys', {}, precision_at_k(X, X['item_carts'] + X['item_buys'], eval_all))
log('baseline: popular last 7d', {}, precision_at_k(X, X['item_events_last7d'], eval_all))
log('baseline: history recency', {}, precision_at_k(X, -X['ui_hours_since_last'].fillna(1e9) + 1e-3 * X['item_carts'], eval_all))

# --- XGBoost, 5-fold CV по пользователям (много положительных у разных пользователей, утечки между фолдами нет)
folds = list(GroupKFold(n_splits=5).split(X[FEATURES], X['target'], groups=X['visitorid']))
y = X['target'].values
spw = (y == 0).sum() / max(y.sum(), 1)

configs = [
    ('xgboost classifier', dict(max_depth=3, n_estimators=200, learning_rate=0.05, scale_pos_weight=1)),
    ('xgboost classifier', dict(max_depth=5, n_estimators=200, learning_rate=0.05, scale_pos_weight=1)),
    ('xgboost classifier', dict(max_depth=5, n_estimators=200, learning_rate=0.05, scale_pos_weight=float(np.sqrt(spw)))),
    ('xgboost classifier', dict(max_depth=7, n_estimators=400, learning_rate=0.03, scale_pos_weight=float(np.sqrt(spw)))),
    ('xgboost ranker (pairwise)', dict(max_depth=5, n_estimators=200, learning_rate=0.05)),
]
common = dict(subsample=0.8, colsample_bytree=0.8, tree_method='hist', random_state=SEED, n_jobs=-1)
best = None
for name, params in configs:
    oof = np.zeros(len(X))
    for tr_idx, va_idx in folds:
        if 'ranker' in name:
            tr_idx = tr_idx[np.argsort(X['visitorid'].values[tr_idx], kind='stable')]
            m = xgb.XGBRanker(objective='rank:pairwise', **params, **common)
            m.fit(X[FEATURES].iloc[tr_idx], y[tr_idx], qid=X['visitorid'].values[tr_idx])
            oof[va_idx] = m.predict(X[FEATURES].iloc[va_idx])
        else:
            m = xgb.XGBClassifier(**params, **common)
            m.fit(X[FEATURES].iloc[tr_idx], y[tr_idx])
            oof[va_idx] = m.predict_proba(X[FEATURES].iloc[va_idx])[:, 1]
    p3 = precision_at_k(X, oof, eval_all)
    seg = {f'p3_{n}': round(precision_at_k(X, oof, u), 5)
           for n, u in [('users_with_history', eval_hist), ('cold_users', eval_cold)]}
    log(name, params, p3, seg)
    if best is None or p3 > best[0]:
        best = (p3, name, params)

# ---------------------------------------------------------------- финальная модель и артефакты
p3, name, params = best
print('лучшая конфигурация:', name, params, p3)
if 'ranker' in name:
    X = X.sort_values('visitorid', kind='stable').reset_index(drop=True)
    final = xgb.XGBRanker(objective='rank:pairwise', **params, **common).fit(X[FEATURES], X['target'], qid=X['visitorid'])
else:
    final = xgb.XGBClassifier(**params, **common).fit(X[FEATURES], X['target'])
final.get_booster().save_model(str(ART / 'model.json'))

items.to_pickle(ART / 'items.pkl')
users.to_pickle(ART / 'users.pkl')
ui.to_pickle(ART / 'ui.pkl')
ucat.to_pickle(ART / 'ucat.pkl')
meta = {
    'cutoff': int(cutoff), 'features': FEATURES, 'model': name, 'params': params,
    'is_ranker': 'ranker' in name, 'popular_pool': [int(i) for i in pool], 'k': K,
    'precision_at_3_cv': round(p3, 5),
}
(ART / 'meta.json').write_text(json.dumps(meta, indent=2))

imp = pd.Series(final.get_booster().get_score(importance_type='gain')).sort_values(ascending=False)
report = {
    'cutoff': int(cutoff), 'positive_users': int(len(pos_users)), 'positive_pairs_test': int(len(pos)),
    'candidates': int(len(X)), 'positives_in_candidates': int(y.sum()),
    'experiments': experiments, 'best': {'model': name, 'params': params, 'precision_at_3': round(p3, 5)},
    'top_features_by_gain': {k: round(float(v), 2) for k, v in imp.head(15).items()},
}
(DOCS / 'experiments.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(f'готово за {time.time() - t0:.0f}с; артефакты в {ART}')
