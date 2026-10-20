"""
Оценка обученной модели на отложенной по времени выборке fraudTest.csv (быстро, без сервиса).
Признаки считаются теми же функциями, что при обучении; история карт берётся по train + test (только прошлое каждой карты).

Запуск:
    python evaluate.py --train data/fraudTrain.csv --test data/fraudTest.csv --model artifacts/model.joblib
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score

from compare_models import CARD, USECOLS, build_base, card_features, card_items
from fpgrowth_features import apply_patterns, make_items

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="data/fraudTrain.csv")
ap.add_argument("--test", default="data/fraudTest.csv")
ap.add_argument("--model", default="artifacts/model.joblib")
args = ap.parse_args()

a = joblib.load(args.model)
model, cols, th_saved = a["model"], a["columns"], float(a["threshold"])

tr = pd.read_csv(args.train, usecols=USECOLS)
te = pd.read_csv(args.test, usecols=USECOLS)
print("Считаю признаки карт (история каждой карты)...")
cf_all = card_features(pd.concat([tr, te], ignore_index=True))
cf_te = cf_all.iloc[len(tr):].reset_index(drop=True)
del tr, cf_all
te = te.reset_index(drop=True)

print("Считаю остальные признаки и FP-паттерны...")
X = build_base(te)[0]
X[CARD] = cf_te.to_numpy()
items = pd.concat([make_items(te), card_items(cf_te)], axis=1).reindex(columns=a["items_columns"], fill_value=False)
fp = apply_patterns(items, a["fp_patterns"])
X["fp_count"] = fp["pat_count"].to_numpy()
X["fp_max_lift"] = fp["pat_max_lift"].to_numpy()
Xa = X.reindex(columns=cols, fill_value=0.0).to_numpy(np.float32)

y = te["is_fraud"].to_numpy()
amt = te["amt"].to_numpy(float)
p = model.predict_proba(Xa)[:, 1]
rate = y.mean()

print("\n" + "=" * 60)
print(f"Операций в test: {len(y):,}, мошеннических: {int(y.sum()):,} ({rate:.3%})")
print(f"ROC-AUC: {roc_auc_score(y, p):.4f}")
print(f"PR-AUC:  {average_precision_score(y, p):.4f}   (модель-угадайка дала бы ~{rate:.4f})")


def report(th, name):
    pred = p >= th
    tn, fp_, fn, tp = confusion_matrix(y, pred).ravel()
    prec = tp / max(tp + fp_, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    money = amt[pred & (y == 1)].sum() / max(amt[y == 1].sum(), 1e-9)
    print(f"\n[{name}] порог {th:.3f}")
    print(f"  поймано мошенничества: {tp:,} из {tp + fn:,}  (recall {rec:.1%}, по сумме {money:.1%})")
    print(f"  заблокировано всего: {tp + fp_:,}  из них ложных: {fp_:,}  (precision {prec:.1%})")
    print(f"  F1 = {f1:.3f}   пропущено фрода: {fn:,}")


report(th_saved, "порог из обучения")
report(0.5, "порог 0.5")
prec, rec, thr = precision_recall_curve(y, p)
f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
report(float(thr[np.argmax(f1)]), "лучший порог на test (только для диагностики)")

print("\nЕсли проверять фиксированную долю самых подозрительных операций:")
order = np.argsort(-p)
for r in (0.001, 0.003, 0.005, 0.01):
    k = int(r * len(y))
    hit = y[order[:k]].sum()
    print(f"  топ {r:.1%} ({k:,} операций): поймано {int(hit):,} фрода, precision {hit / k:.1%}, recall {hit / y.sum():.1%}")

print("\nСамые важные признаки модели (gain):")
imp = pd.Series(model.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
print((imp / imp.sum()).head(15).round(4).to_string())
