"""
Онлайн-скоринг одной транзакции обученной моделью.
Признаки считаются теми же функциями, что и при обучении (compare_models / fpgrowth_features),
а история карты хранится в компактной сводке CardState (формулы повторяют card_features).
"""
import math
import threading
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd

from compare_models import CARD, build_base, card_items
from fpgrowth_features import apply_patterns, haversine_km, make_items

EPOCH = pd.Timestamp("1970-01-01")


class CardState:
    """Сводка истории одной карты (только прошлые операции)."""

    def __init__(self):
        self.n = 0
        self.s_amt = self.s_amt2 = self.max_amt = 0.0
        self.s_d = self.s_d2 = 0.0
        self.last_t = self.last_mlat = self.last_mlon = None
        self.blk = [0] * 6                 # операции по 4-часовым блокам суток
        self.cat = defaultdict(int)        # операции по категориям
        self.mer = set()                   # продавцы, у которых карта уже покупала
        self.recent = deque(maxlen=500)    # времена последних операций (окна 1ч / 24ч)

    def features(self, t, amt, dist, mlat, mlon, hour, cat, mer, med):
        n = self.n

        def z(x, s1, s2):  # как _past_z: при истории короче 5 операций z = 0
            if n < 5:
                return 0.0
            m = s1 / n
            sd = math.sqrt(max(s2 / n - m * m, 0.0))
            return (x - m) / max(sd, 0.1 * m, 1.0)

        f = {
            "card_log_n_prev": math.log1p(n),
            "card_amt_z": z(amt, self.s_amt, self.s_amt2),
            "card_dist_z": z(dist, self.s_d, self.s_d2),
            "card_new_merchant": float(mer not in self.mer),
            "card_tx_1h": float(sum(x >= t - 3600 for x in self.recent)),
            "card_tx_24h": float(sum(x >= t - 86400 for x in self.recent)),
        }
        if n == 0:  # первая операция карты: медианы (как fillna в card_features)
            for k in ("card_amt_ratio_mean", "card_amt_ratio_max", "card_log_dt_prev",
                      "card_speed_kmh", "card_hour_share", "card_cat_share"):
                f[k] = med[k]
        else:
            dt_h = (t - self.last_t) / 3600
            km = float(haversine_km(self.last_mlat, self.last_mlon, mlat, mlon))
            f["card_amt_ratio_mean"] = amt / max(self.s_amt / n, 1.0)
            f["card_amt_ratio_max"] = amt / max(self.max_amt, 1.0)
            f["card_log_dt_prev"] = math.log1p(max(dt_h, 0.0))
            f["card_speed_kmh"] = min(km / max(dt_h, 1 / 60), 2000.0)
            f["card_hour_share"] = self.blk[hour // 4] / n
            f["card_cat_share"] = self.cat.get(cat, 0) / n
        return f

    def update(self, t, amt, dist, mlat, mlon, hour, cat, mer):
        self.n += 1
        self.s_amt += amt
        self.s_amt2 += amt * amt
        self.max_amt = max(self.max_amt, amt)
        self.s_d += dist
        self.s_d2 += dist * dist
        if self.last_t is None or t >= self.last_t:
            self.last_t, self.last_mlat, self.last_mlon = t, mlat, mlon
        self.blk[hour // 4] += 1
        self.cat[cat] += 1
        self.mer.add(mer)
        self.recent.append(t)


def build_card_state(df):
    """Начальное состояние сервиса: прогоняем всю историю train в порядке времени."""
    d = df.copy()
    ts = pd.to_datetime(d["trans_date_trans_time"])
    d["t"] = (ts - EPOCH).dt.total_seconds()
    d["hour"] = ts.dt.hour
    d["dist"] = haversine_km(d["lat"], d["long"], d["merch_lat"], d["merch_long"])
    d = d.sort_values(["cc_num", "t"], kind="stable")
    states = {}
    for cc, t, amt, dist, mlat, mlon, h, cat, mer in zip(
            d["cc_num"].astype(str), d["t"], d["amt"], d["dist"], d["merch_lat"],
            d["merch_long"], d["hour"], d["category"], d["merchant"]):
        if cc not in states:
            states[cc] = CardState()
        states[cc].update(t, float(amt), float(dist), float(mlat), float(mlon), int(h), cat, mer)
    return states


class Scorer:
    def __init__(self, path="artifacts/model.joblib"):
        a = joblib.load(path)
        self.model, self.cols = a["model"], a["columns"]
        self.pats, self.items_cols = a["fp_patterns"], a["items_columns"]
        self.medians, self.threshold = a["medians"], float(a["threshold"])
        self.cards = a["card_state"]
        self.lock = threading.Lock()  # FastAPI выполняет sync-эндпоинты в потоках

    def score(self, tx: dict):
        row = pd.DataFrame([tx])
        ts = pd.Timestamp(tx["trans_date_trans_time"])
        t, hour = (ts - EPOCH).total_seconds(), ts.hour
        dist = float(haversine_km(tx["lat"], tx["long"], tx["merch_lat"], tx["merch_long"]))
        args = (t, float(tx["amt"]), dist, float(tx["merch_lat"]), float(tx["merch_long"]),
                hour, tx["category"], tx.get("merchant") or "unknown")

        with self.lock:
            st = self.cards.setdefault(str(tx["cc_num"]), CardState())
            cvec = st.features(*args, self.medians)
            cf = pd.DataFrame([cvec])[CARD].astype(np.float32)
            items = pd.concat([make_items(row), card_items(cf)], axis=1) \
                      .reindex(columns=self.items_cols, fill_value=False)
            fp = apply_patterns(items, self.pats)
            st.update(*args)  # историю обновляем ПОСЛЕ расчёта признаков

        base, _ = build_base(row)  # те же обычные признаки, что при обучении
        x = (base.assign(**cvec, fp_count=float(fp["pat_count"].iloc[0]),
                         fp_max_lift=float(fp["pat_max_lift"].iloc[0]))
                 .reindex(columns=self.cols, fill_value=0.0).to_numpy(np.float32))

        p = float(self.model.predict_proba(x)[0, 1])
        contrib = self.model.booster_.predict(x, pred_contrib=True)[0][:-1]  # SHAP-вклады признаков
        top = [i for i in np.argsort(-contrib)[:3] if contrib[i] > 0]
        reasons = [f"{self.cols[i]} = {x[0, i]:.3g} (вклад {contrib[i]:+.2f})" for i in top]
        return p, reasons or ["Стандартный профиль операции (низкий риск)"]
