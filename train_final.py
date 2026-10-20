"""
Обучает финальную модель (LightGBM) и сохраняет всё, что нужно сервису, в artifacts/model.joblib:
модель, список колонок, FP-паттерны, медианы карточных признаков, порог (по F1 на валидации),
начальное состояние истории карт.

Запуск:
    python train_final.py --train data/fraudTrain.csv --out artifacts
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from compare_models import CARD, FP, USECOLS, build_base, card_features, card_items, fp_oof, make_model
from fpgrowth_features import make_items, mine_patterns
from serving import build_card_state

TOKENS = ["amt", "log_amt", "hour", "dow", "age", "gender", "dist_km", "city_pop",
          "category", "card_all", "fp_count", "fp_max_lift"]

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--out", default="artifacts")
ap.add_argument("--max_rows", type=int, default=300_000, help="0 = весь train")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

full = pd.read_csv(args.train, usecols=USECOLS)
cf = card_features(full)                 # признаки карт по ПОЛНОЙ истории
medians = cf.median().to_dict()
state = build_card_state(full)           # стартовое состояние истории карт для сервиса

tr = full
if args.max_rows and len(full) > args.max_rows:
    tr = full.sample(args.max_rows, random_state=42)
    cf = cf.loc[tr.index]
tr, cf = tr.reset_index(drop=True), cf.reset_index(drop=True)
y = tr["is_fraud"].to_numpy()

X, groups = build_base(tr)
X[CARD] = cf.to_numpy()
items = pd.concat([make_items(tr), card_items(cf)], axis=1)
mine_kw = dict(min_support=0.05, max_len=4, min_cover=100, top_n=30, min_lift=3.0)
X[FP] = fp_oof(items, y, 5, mine_kw, 42)  # для обучения: out-of-fold
pats = mine_patterns(items, y, **mine_kw)  # для сервиса: паттерны со всего train
for t in FP + CARD:
    groups[t] = [t]
groups["card_all"] = list(CARD)
cols = [c for t in TOKENS for c in groups[t]]

Xa = X[cols].to_numpy(np.float32)
i_fit, i_val = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
pos_w = (1 - y[i_fit].mean()) / y[i_fit].mean()
model = make_model("lgbm", pos_w, 42, -1)
model.fit(Xa[i_fit], y[i_fit])

p = model.predict_proba(Xa[i_val])[:, 1]
prec, rec, thr = precision_recall_curve(y[i_val], p)
f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
th = float(thr[np.argmax(f1)])
print(f"Порог по F1 на валидации: {th:.3f}, F1={f1.max():.3f}")

path = os.path.join(args.out, "model.joblib")
joblib.dump({"model": model, "columns": cols, "fp_patterns": pats,
             "items_columns": list(items.columns), "medians": medians,
             "threshold": th, "card_state": state}, path)
print(f"Сохранено: {path}")
