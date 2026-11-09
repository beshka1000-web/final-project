import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import make_features


class Recommender:
    def __init__(self, art_dir: Path):
        self.meta = json.loads((art_dir / 'meta.json').read_text())
        self.items = pd.read_pickle(art_dir / 'items.pkl')
        self.users = pd.read_pickle(art_dir / 'users.pkl').sort_index()
        self.ui = pd.read_pickle(art_dir / 'ui.pkl').sort_index()
        self.ucat = pd.read_pickle(art_dir / 'ucat.pkl').sort_index()
        # .loc[[id]] по таблицам в миллионы строк занимает сотни мс, поэтому режем по отсортированным id через searchsorted
        self._ui_vid = self.ui.index.get_level_values(0).values
        self._ucat_vid = self.ucat.index.get_level_values(0).values
        self._users_vid = self.users.index.values
        self.booster = xgb.Booster()
        self.booster.load_model(str(art_dir / 'model.json'))
        self.pool = np.array(self.meta['popular_pool'], dtype='int64')
        self.features = self.meta['features']

    @staticmethod
    def _rows(table, ids, user_id):
        lo, hi = np.searchsorted(ids, user_id, 'left'), np.searchsorted(ids, user_id, 'right')
        return table.iloc[lo:hi]

    def is_known(self, user_id: int) -> bool:
        return len(self._rows(self.users, self._users_vid, user_id)) > 0

    def recommend(self, user_id: int, k: int = 3):
        """Топ-k товаров: кандидаты = история пользователя + популярный пул, ранжирует XGBoost.
        Для неизвестного пользователя остаётся только популярный пул (признаки пользователя пустые)."""
        users_u = self._rows(self.users, self._users_vid, user_id)
        known = len(users_u) > 0
        ui_u = self._rows(self.ui, self._ui_vid, user_id)
        ucat_u = self._rows(self.ucat, self._ucat_vid, user_id)
        cand = np.union1d(ui_u.index.get_level_values(1).values, self.pool)
        pairs = pd.DataFrame({'visitorid': np.int32(user_id), 'itemid': cand.astype('int32')})
        # в make_features уходят только срезы одного пользователя
        X = make_features(pairs, self.items.reindex(cand), users_u, ui_u, ucat_u)
        score = self.booster.predict(xgb.DMatrix(X[self.features]))
        order = np.argsort(-score)[:k]
        recs = [{'rank': i + 1, 'item_id': int(cand[j]), 'score': float(score[j])} for i, j in enumerate(order)]
        return recs, ('personalized' if known else 'popular_fallback')
