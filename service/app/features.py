"""Сборка признаков пары (пользователь, товар). Общий код для обучения и сервиса,
чтобы не было расхождения (train/serve skew)."""
import numpy as np
import pandas as pd

H = 3_600_000  # мс в часе

ITEM_COLS = [
    'item_views', 'item_carts', 'item_buys', 'item_users', 'item_cart_rate', 'item_buy_rate',
    'item_buy_per_cart', 'item_events_last1d', 'item_events_last7d', 'item_trend_7d',
    'item_pop_rank', 'item_buy_rank', 'item_days_since_last', 'item_age_days',
    'item_available', 'item_log_price', 'item_price_vs_cat', 'cat_buy_rate', 'cat_depth', 'categoryid',
]
USER_COLS = [
    'user_views', 'user_carts', 'user_buys', 'user_events', 'user_items', 'user_sessions',
    'user_days_since_last', 'user_cart_rate', 'user_buy_rate', 'user_mean_price',
]
UI_COLS = [
    'ui_views', 'ui_carts', 'ui_buys', 'ui_hours_since_last', 'ui_last_event',
    'ui_in_history', 'ui_cat_events', 'ui_cat_share',
]
FEATURES = ITEM_COLS + USER_COLS + UI_COLS
LAST_EVENT = {'view': 0, 'addtocart': 1, 'transaction': 2}


def build_ui_tables(train: pd.DataFrame, cutoff: int, item_category: pd.Series):
    """По train-событиям считает признаки истории: пара (user, item) и пара (user, категория)."""
    g = train.groupby(['visitorid', 'itemid', 'event'], observed=True).size().unstack(fill_value=0)
    ui = pd.DataFrame({
        'ui_views': g.get('view', 0), 'ui_carts': g.get('addtocart', 0), 'ui_buys': g.get('transaction', 0),
    }).astype('float32')
    keys = ['visitorid', 'itemid']
    ui['ui_hours_since_last'] = ((cutoff - train.groupby(keys)['timestamp'].max()) / H).astype('float32')
    last = train.sort_values('timestamp').groupby(keys, observed=True)['event'].last().astype(str)
    ui['ui_last_event'] = last.map(LAST_EVENT).astype('float32')

    cat = train[['visitorid', 'itemid']].assign(categoryid=train['itemid'].map(item_category))
    ucat = cat.dropna().groupby(['visitorid', 'categoryid']).size().rename('ui_cat_events').astype('float32')
    return ui, ucat.to_frame()


def make_features(pairs: pd.DataFrame, items, users, ui, ucat) -> pd.DataFrame:
    """pairs: колонки visitorid, itemid. Возвращает DataFrame с колонками FEATURES (float32).
    items/users индексированы по itemid/visitorid, ui — по (visitorid, itemid), ucat — по (visitorid, categoryid)."""
    X = pairs[['visitorid', 'itemid']].reset_index(drop=True)
    X = X.join(items[ITEM_COLS], on='itemid').join(users[USER_COLS], on='visitorid')
    X = X.join(ui, on=['visitorid', 'itemid'])
    X['ui_in_history'] = X['ui_views'].notna().astype('float32')
    X[['ui_views', 'ui_carts', 'ui_buys']] = X[['ui_views', 'ui_carts', 'ui_buys']].fillna(0)
    X['ui_last_event'] = X['ui_last_event'].fillna(-1)
    X = X.join(ucat, on=['visitorid', 'categoryid'])
    X['ui_cat_events'] = X['ui_cat_events'].fillna(0)
    X['ui_cat_share'] = X['ui_cat_events'] / X['user_events'].clip(lower=1)
    return X[['visitorid', 'itemid'] + FEATURES].astype({c: 'float32' for c in FEATURES})
