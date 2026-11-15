"""
Пакетный скоринг таблицы операций обученной моделью.
Признаки считаются теми же функциями, что при обучении и в онлайн-сервисе (serving.py): карточные признаки идут
последовательно по времени от состояния истории карт на конец train (хранится в model.joblib), остальные признаки
и FP-паттерны считаются векторно.
"""
import hashlib

import joblib
import numpy as np
import pandas as pd

from compare_models import CARD, build_base, card_items
from fpgrowth_features import apply_patterns, haversine_km, make_items
from serving import EPOCH, CardState


def model_version(path: str) -> str:
    """Версия модели = первые 12 символов SHA-256 файла model.joblib."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def load_artifact(path: str) -> dict:
    return joblib.load(path)  # свежая копия истории карт при каждой загрузке: результат воспроизводим


def card_features_sequential(df: pd.DataFrame, art: dict) -> pd.DataFrame:
    states, med = art["card_state"], art["medians"]
    ts = pd.to_datetime(df["trans_date_trans_time"])
    t = (ts - EPOCH).dt.total_seconds().to_numpy()
    hour = ts.dt.hour.to_numpy()
    dist = np.asarray(haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"]), dtype=float)
    rows = []
    for cc, ti, amt, d, mlat, mlon, h, cat, mer in zip(
            df["cc_num"].astype(str), t, df["amt"].astype(float), dist, df["merch_lat"].astype(float),
            df["merch_long"].astype(float), hour, df["category"], df["merchant"].fillna("unknown")):
        st = states.get(cc)
        if st is None:
            st = states[cc] = CardState()
        args = (float(ti), float(amt), float(d), float(mlat), float(mlon), int(h), cat, mer or "unknown")
        rows.append(st.features(*args, med))   # сначала признаки по прошлому,
        st.update(*args)                       # затем обновление истории (как в онлайн-сервисе)
    return pd.DataFrame(rows)[CARD].astype(np.float32)


def score_frame(df: pd.DataFrame, art: dict, chunk: int = 100_000) -> pd.DataFrame:
    """Возвращает DataFrame(trans_num, fraud_score) для всех строк df."""
    df = df.sort_values("trans_date_trans_time", kind="stable").reset_index(drop=True)
    cf = card_features_sequential(df, art)

    X = build_base(df)[0]
    X[CARD] = cf.to_numpy()
    items = pd.concat([make_items(df), card_items(cf)], axis=1).reindex(columns=art["items_columns"], fill_value=False)
    fp = apply_patterns(items, art["fp_patterns"])
    X["fp_count"] = fp["pat_count"].to_numpy()
    X["fp_max_lift"] = fp["pat_max_lift"].to_numpy()
    Xa = X.reindex(columns=art["columns"], fill_value=0.0).to_numpy(np.float32)

    model = art["model"]
    p = np.concatenate([model.predict_proba(Xa[i:i + chunk])[:, 1] for i in range(0, len(Xa), chunk)])
    return pd.DataFrame({"trans_num": df["trans_num"].to_numpy(), "fraud_score": p})
