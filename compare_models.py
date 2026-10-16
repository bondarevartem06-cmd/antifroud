"""
Сравнение влияния признаков (в т.ч. FP-Growth) на LogReg / RandomForest / XGBoost / LightGBM.

Нужен файл fpgrowth_features.py в той же папке.

ДОСТУПНЫЕ ПРИЗНАКИ (токены):
  числовые : amt log_amt hour dow month is_weekend age dist_km city_pop lat long merch_lat merch_long
  категор. : category gender state          (автоматически one-hot)
  FP       : fp_count fp_max_lift           (из FP-Growth)
  карта    : card_all (все сразу) или по отдельности: card_log_n_prev card_amt_z card_amt_ratio_mean
             card_amt_ratio_max card_log_dt_prev card_tx_1h card_tx_24h card_speed_kmh card_dist_z
             card_hour_share card_cat_share card_new_merchant   (поведение владельца карты по cc_num)

ПРИМЕРЫ
  # по умолчанию: набор "base" (все обычные признаки) против "base+fp"
  python compare_models.py --train fraudTrain.csv --test fraudTest.csv

  # свои наборы (можно сколько угодно, первый считается базовым для расчёта дельты)
  python compare_models.py --train fraudTrain.csv --test fraudTest.csv \
      --set "mini=amt,hour,category" \
      --set "mini+fp=amt,hour,category,fp_count,fp_max_lift" \
      --set "only_fp=fp_count,fp_max_lift" \
      --models lr rf xgb lgbm

  # + абляция: убираем по одному признаку из первого набора (на выбранной модели)
  python compare_models.py --train fraudTrain.csv --test fraudTest.csv --ablation --ablation_model lgbm

FP-признаки для train считаются out-of-fold (паттерны для каждой части ищутся на остальных
частях), для test - паттерны из всего train. Так модель не видит собственные метки через паттерны.
"""
import argparse
import contextlib
import io
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fpgrowth_features import apply_patterns, haversine_km, make_items, mine_patterns

NUMERIC = ["amt", "log_amt", "hour", "dow", "month", "is_weekend", "age",
           "dist_km", "city_pop", "lat", "long", "merch_lat", "merch_long"]
CATEG = ["category", "gender", "state"]
FP = ["fp_count", "fp_max_lift"]
CARD = [  # признаки поведения владельца карты (cc_num), считаются только по ПРОШЛЫМ операциям этой карты
    "card_log_n_prev",      # log(1 + число прошлых операций карты)
    "card_amt_z",           # z-оценка суммы относительно прошлых сумм этой карты
    "card_amt_ratio_mean",  # сумма / средняя прошлая сумма карты
    "card_amt_ratio_max",   # сумма / максимальная прошлая сумма карты (> 1 = новый рекорд)
    "card_log_dt_prev",     # log(1 + часов с предыдущей операции карты)
    "card_tx_1h",           # сколько операций карты было за предыдущий час
    "card_tx_24h",          # сколько операций карты было за предыдущие 24 часа
    "card_speed_kmh",       # скорость перемещения между точками продаж двух последних операций
    "card_dist_z",          # z-оценка расстояния клиент-продавец относительно привычного для карты
    "card_hour_share",      # доля прошлых операций карты в тот же 4-часовой интервал суток
    "card_cat_share",       # доля прошлых операций карты в этой категории (0 = новая категория)
    "card_new_merchant",    # 1 = карта впервые у этого продавца
]
ALL_TOKENS = NUMERIC + CATEG + FP + CARD + ["card_all"]  # card_all = все признаки CARD сразу

USECOLS = ["trans_date_trans_time", "category", "amt", "gender", "dob", "lat", "long",
           "merch_lat", "merch_long", "city_pop", "state", "is_fraud", "cc_num", "merchant"]


# ----------------------------------------------------------------------------
# Данные и признаки
# ----------------------------------------------------------------------------
def build_base(df: pd.DataFrame):
    """Обычные признаки. Возвращает (матрица, словарь токен -> список колонок)."""
    ts = pd.to_datetime(df["trans_date_trans_time"])
    out = pd.DataFrame(index=df.index)
    out["amt"] = df["amt"]
    out["log_amt"] = np.log1p(df["amt"])
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["age"] = (ts - pd.to_datetime(df["dob"])).dt.days / 365.25
    out["dist_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    for c in ["city_pop", "lat", "long", "merch_lat", "merch_long"]:
        out[c] = df[c]

    groups = {t: [t] for t in NUMERIC}
    parts = [out]
    for c in CATEG:
        d = pd.get_dummies(df[c].astype(str), prefix=c).astype(np.float32)
        groups[c] = list(d.columns)
        parts.append(d)
    return pd.concat(parts, axis=1), groups


def _window_counts(cc, t, window):
    """Сколько операций той же карты было за предыдущие window секунд (cc и t отсортированы по карте и времени)."""
    out = np.zeros(len(t), np.float32)
    starts = np.flatnonzero(np.r_[True, cc[1:] != cc[:-1]])
    ends = np.r_[starts[1:], len(t)]
    for a, b in zip(starts, ends):
        tt = t[a:b]
        out[a:b] = np.arange(b - a) - np.searchsorted(tt, tt - window, side="left")
    return out


def _past_z(d, col, n_prev, min_hist=5):
    """z-оценка значения относительно прошлых значений той же карты (текущее не учитывается).
    При истории короче min_hist операций z = 0; знаменатель не меньше max(1, 10% среднего)."""
    x, c = d[col], d["cc"]
    s1 = x.groupby(c).cumsum() - x
    s2 = (x * x).groupby(c).cumsum() - x * x
    n = n_prev.replace(0, np.nan)
    mean = s1 / n
    std = np.sqrt(np.clip(s2 / n - mean * mean, 0, None))
    scale = pd.concat([std, 0.1 * mean], axis=1).max(axis=1).clip(lower=1.0)
    z = ((x - mean) / scale).where(n_prev >= min_hist, 0.0)
    return mean, z


def card_features(df):
    """Поведение владельца карты. Для каждой операции считается, насколько она необычна ИМЕННО ДЛЯ ЭТОЙ КАРТЫ,
    по её прошлым операциям (без текущей и без меток is_fraud, поэтому утечки нет). Возвращает DataFrame
    с колонками CARD в порядке строк df. У первых операций карты вместо пропусков стоят медианы."""
    ts = pd.to_datetime(df["trans_date_trans_time"])
    d = pd.DataFrame({
        "cc": df["cc_num"].to_numpy(),
        "t": (ts - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(),
        "amt": df["amt"].to_numpy(float),
        "dist": np.asarray(haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"]), float),
        "mlat": df["merch_lat"].to_numpy(float), "mlon": df["merch_long"].to_numpy(float),
        "blk": (ts.dt.hour // 4).to_numpy(),
        "cat": pd.factorize(df["category"])[0], "mer": pd.factorize(df["merchant"])[0],
    })
    d["pos"] = np.arange(len(d))
    d = d.sort_values(["cc", "t", "pos"], kind="stable").reset_index(drop=True)  # по карте, затем по времени
    g = d.groupby("cc", sort=False)
    n_prev = g.cumcount()  # сколько операций у карты было раньше
    out = pd.DataFrame(index=d.index)
    out["card_log_n_prev"] = np.log1p(n_prev)
    mean_amt, out["card_amt_z"] = _past_z(d, "amt", n_prev)
    out["card_amt_ratio_mean"] = d["amt"] / mean_amt.clip(lower=1.0)
    prev_max = g["amt"].cummax().groupby(d["cc"]).shift(1)
    out["card_amt_ratio_max"] = d["amt"] / prev_max.clip(lower=1.0)
    dt_h = (d["t"] - g["t"].shift(1)) / 3600  # часов с предыдущей операции карты
    out["card_log_dt_prev"] = np.log1p(dt_h.clip(lower=0))
    out["card_tx_1h"] = _window_counts(d["cc"].to_numpy(), d["t"].to_numpy(), 3600)
    out["card_tx_24h"] = _window_counts(d["cc"].to_numpy(), d["t"].to_numpy(), 86400)
    prev_km = haversine_km(g["mlat"].shift(1), g["mlon"].shift(1), d["mlat"], d["mlon"])
    out["card_speed_kmh"] = (prev_km / dt_h.clip(lower=1 / 60)).clip(upper=2000)
    _, out["card_dist_z"] = _past_z(d, "dist", n_prev)
    out["card_hour_share"] = d.groupby(["cc", "blk"]).cumcount() / n_prev.replace(0, np.nan)
    out["card_cat_share"] = d.groupby(["cc", "cat"]).cumcount() / n_prev.replace(0, np.nan)
    out["card_new_merchant"] = (d.groupby(["cc", "mer"]).cumcount() == 0).astype(float)
    out = out.fillna(out.median())  # первая операция карты: нейтральные значения
    out.index = d["pos"].to_numpy()
    return out.sort_index().set_axis(df.index, axis=0).astype(np.float32)


def card_items(cf):
    """Дискретизированные «личные» признаки карты для FP-Growth: паттерны вида
    'новый продавец & сумма намного выше обычной для карты'."""
    it = pd.DataFrame(index=cf.index)
    it["card_history"] = pd.cut(np.expm1(cf["card_log_n_prev"]), [-1, 4, 50, np.inf],
                                labels=["short<5", "mid5-50", "long>50"]).astype(str)
    it["card_amt_vs_usual"] = pd.cut(cf["card_amt_z"], [-np.inf, 1, 3, np.inf],
                                     labels=["usual", "high", "extreme"]).astype(str)
    it["card_dist_vs_usual"] = pd.cut(cf["card_dist_z"], [-np.inf, 1, 3, np.inf],
                                      labels=["usual", "far", "very_far"]).astype(str)
    it["card_speed"] = pd.cut(cf["card_speed_kmh"], [-1, 50, 500, np.inf], labels=["<50", "50-500", ">500"]).astype(str)
    it["card_gap"] = pd.cut(np.expm1(cf["card_log_dt_prev"]), [-1, 1, 24, np.inf],
                            labels=["<1h", "1-24h", ">24h"]).astype(str)
    it["card_burst_1h"] = pd.cut(cf["card_tx_1h"], [-1, 0, 1, np.inf], labels=["0", "1", "2+"]).astype(str)
    it["card_new_merchant"] = np.where(cf["card_new_merchant"] > 0, "yes", "no")
    it["card_category_usage"] = pd.cut(cf["card_cat_share"], [-1, 0, 0.05, 1.01],
                                       labels=["new", "rare", "usual"]).astype(str)
    it["card_hour_usage"] = pd.cut(cf["card_hour_share"], [-1, 0.05, 1.01], labels=["rare_hour", "usual_hour"]).astype(str)
    return pd.get_dummies(it, prefix_sep="=").astype(bool)


def fp_aggregate(items_fit, y_fit, items_apply, mine_kw):
    """Ищет паттерны на items_fit и считает fp_count / fp_max_lift для items_apply."""
    with contextlib.redirect_stdout(io.StringIO()):  # глушим print из mine_patterns
        try:
            pats = mine_patterns(items_fit, y_fit, **mine_kw)
        except RuntimeError:  # паттернов не нашлось
            return np.zeros((len(items_apply), 2), np.float32)
        feats = apply_patterns(items_apply, pats)
    return feats[["pat_count", "pat_max_lift"]].to_numpy(np.float32)


def fp_oof(items, y, n_splits, mine_kw, seed):
    """Out-of-fold FP-признаки для обучающей выборки."""
    oof = np.zeros((len(items), 2), np.float32)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(skf.split(items, y), 1):
        oof[va] = fp_aggregate(items.iloc[tr], y[tr], items.iloc[va], mine_kw)
        print(f"  FP out-of-fold: фолд {k}/{n_splits}")
    return oof


def prepare(train_path, test_path, max_train_rows, n_splits, mine_kw, seed, card_fp=True):
    tr = pd.read_csv(train_path, usecols=USECOLS)
    te = pd.read_csv(test_path, usecols=USECOLS)
    for nm, d in (("train", tr), ("test", te)):
        f = d[d["is_fraud"] == 1]
        print(f"ВЕСЬ {nm}: {len(d):,} транзакций, мошеннических: {len(f):,}, "
              f"уникальных карт с мошенничеством: {f['cc_num'].nunique():,}")

    # поведение владельцев карт: по ПОЛНЫМ train + test (до подвыборки), только по прошлому каждой карты
    print("Считаю признаки по картам (история каждой карты)...")
    cf_all = card_features(pd.concat([tr, te], ignore_index=True))
    cf_tr, cf_te = cf_all.iloc[:len(tr)], cf_all.iloc[len(tr):].reset_index(drop=True)

    if max_train_rows and len(tr) > max_train_rows:
        tr = tr.sample(max_train_rows, random_state=seed)
    cf_tr = cf_tr.loc[tr.index].reset_index(drop=True)  # признаки карт тех же строк, что попали в подвыборку
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    y_tr, y_te = tr["is_fraud"].to_numpy(), te["is_fraud"].to_numpy()
    print(f"train: {len(tr):,} (fraud {y_tr.mean():.3%}), test: {len(te):,} (fraud {y_te.mean():.3%})")
    print("Мошенничество по категориям (train):")
    print(tr.groupby("category")["is_fraud"].agg(fraud="sum", total="count", rate="mean")
            .sort_values("fraud", ascending=False).round(4).to_string())

    # обычные признаки: считаем на объединении, чтобы one-hot колонки совпали
    base, groups = build_base(pd.concat([tr, te], ignore_index=True))
    X_tr, X_te = base.iloc[:len(tr)].copy(), base.iloc[len(tr):].reset_index(drop=True)
    X_tr[CARD], X_te[CARD] = cf_tr.to_numpy(), cf_te.to_numpy()

    # FP-признаки (при card_fp личные признаки карты тоже участвуют в паттернах)
    items_tr, items_te = make_items(tr), make_items(te)
    if card_fp:
        items_tr = pd.concat([items_tr, card_items(cf_tr)], axis=1)
        items_te = pd.concat([items_te, card_items(cf_te)], axis=1)
    items_te = items_te.reindex(columns=items_tr.columns, fill_value=False)
    print("Считаю FP-Growth признаки...")
    X_tr[FP] = fp_oof(items_tr, y_tr, n_splits, mine_kw, seed)
    X_te[FP] = fp_aggregate(items_tr, y_tr, items_te, mine_kw)
    for t in FP + CARD:
        groups[t] = [t]
    groups["card_all"] = list(CARD)
    return X_tr.astype(np.float32), y_tr, X_te.astype(np.float32), y_te, groups, te.assign(day=pd.to_datetime(te["trans_date_trans_time"]).dt.strftime("%Y-%m-%d"))[["amt", "cc_num", "day"]]


# ----------------------------------------------------------------------------
# Модели и оценка
# ----------------------------------------------------------------------------
def make_model(key, pos_w, seed, n_jobs):
    if key == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=seed))
    if key == "rf":
        return RandomForestClassifier(
            n_estimators=150, max_depth=20, min_samples_leaf=3, n_jobs=n_jobs,
            class_weight="balanced_subsample", random_state=seed)
    if key == "xgb":
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_w, tree_method="hist",
            eval_metric="aucpr", n_jobs=n_jobs, random_state=seed)
    if key == "lgbm":
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=pos_w,
            n_jobs=n_jobs, random_state=seed, verbose=-1)
    raise ValueError(key)


def capacity_flags(score, day, rate):
    """Лимит службы мониторинга: в каждые сутки помечаем не более rate * (число операций за сутки)
    операций с наибольшим score. Возвращает булев массив (True = отправить на проверку)."""
    s = pd.Series(np.asarray(score))
    day = np.asarray(day)
    rank = s.groupby(day).rank(method="first", ascending=False)  # 1 = самая подозрительная за сутки
    quota = np.floor(rate * s.groupby(day).transform("size") + 1e-9)
    return (rank <= quota).to_numpy()


def cost_stats(pred, yt, amt, costs):
    """Экономика проверки, у.е. costs = (проверка одной операции, фикс. штраф за пропуск, ложная блокировка).
    Потери = проверки всех помеченных + (сумма + фикс. штраф) за пропущенное мошенничество + ложные блокировки."""
    if costs is None or amt is None:
        return {}
    c_review, c_miss, c_block = costs
    fraud = yt == 1
    miss, fa = (~pred) & fraud, pred & ~fraud
    review = c_review * float(pred.sum())
    missed = float(amt[miss].sum()) + c_miss * float(miss.sum())
    block = c_block * float(fa.sum())
    total = review + missed + block
    no_model = float(amt[fraud].sum()) + c_miss * float(fraud.sum())  # если ничего не проверять
    return {
        "cost_review": review,             # стоимость проверок аналитиком
        "cost_missed": missed,             # потери от пропущенного мошенничества
        "cost_false_block": block,         # потери от ложных блокировок
        "cost_total": total,               # суммарные потери (меньше = лучше)
        "cost_no_model": no_model,         # потери без проверки вообще
        "net_benefit": no_model - total,   # чистая выгода (больше = лучше; < 0 = проверка убыточна)
    }


def money_stats(pred, yt, amt, cc):
    """Мошенники и деньги при выбранном пороге (pred - предсказанные мошенничества на test)."""
    if amt is None:
        return {}
    fraud = yt == 1
    hit, miss, fa = pred & fraud, (~pred) & fraud, pred & ~fraud
    saved, missed = float(amt[hit].sum()), float(amt[miss].sum())
    return {
        "flagged": int(pred.sum()),                 # сколько операций отправлено на проверку
        "flagged_share": float(pred.mean()),        # их доля от всех операций test (должна быть <= лимита)
        "frauds_total": int(fraud.sum()),           # всего мошеннических транзакций в test
        "detected": int(hit.sum()),                 # из них обнаружено
        "missed": int(miss.sum()),                  # пропущено
        "false_alarms": int(fa.sum()),              # честные транзакции, помеченные как мошенничество
        "cards_fraud_total": int(len(np.unique(cc[fraud]))),  # карт с мошенничеством
        "cards_detected": int(len(np.unique(cc[hit]))),       # карт, где мошенничество обнаружено
        "saved_money": saved,                       # сумма обнаруженных мошеннических платежей
        "missed_money": missed,                     # сумма пропущенных
        "saved_share": saved / max(saved + missed, 1e-9),
        "blocked_legit_money": float(amt[fa].sum()),  # сумма ошибочно помеченных честных платежей
    }


def curve_rows(p_te, yt, amt, day, caps, rank, costs):
    """Кривая «бюджет проверок -> precision / recall / деньги»: тот же отбор операций, но для набора лимитов caps."""
    if rank == "money":
        score = p_te * amt
    elif rank == "cost" and costs:
        score = p_te * (amt + costs[1] + costs[2])
    else:
        score = p_te
    s = pd.Series(np.asarray(score))
    day = np.asarray(day)
    rank_in_day = s.groupby(day).rank(method="first", ascending=False)  # ранг считаем один раз
    n_day = s.groupby(day).transform("size")
    fraud = yt == 1
    rows = []
    for c in caps:
        pred = (rank_in_day <= np.floor(c * n_day + 1e-9)).to_numpy()
        tp, flagged = int((pred & fraud).sum()), int(pred.sum())
        row = {"cap": c, "flagged": flagged, "detected": tp,
               "precision": tp / max(flagged, 1), "recall": tp / max(int(fraud.sum()), 1),
               "saved_money": float(amt[pred & fraud].sum())}
        row.update({k: v for k, v in cost_stats(pred, yt, amt, costs).items() if k in ("cost_total", "net_benefit")})
        rows.append(row)
    return rows


def evaluate(key, Xf, yf, Xv, yv, Xt, yt, seed, n_jobs, wf=None, amt=None, cc=None,
             day=None, cap=None, rank="prob", costs=None, caps=None):
    """Обучает на fit, порог выбирает по F1 на val, метрики считает на test."""
    pos_w = (1 - yf.mean()) / max(yf.mean(), 1e-9)
    model = make_model(key, pos_w, seed, n_jobs)
    t0 = time.time()
    if key == "lr":
        model.fit(Xf, yf, logisticregression__sample_weight=wf)
    else:
        model.fit(Xf, yf, sample_weight=wf)
    fit_sec = time.time() - t0  # время обучения модели, секунды
    p_val, p_te = model.predict_proba(Xv)[:, 1], model.predict_proba(Xt)[:, 1]
    # контроль переобучения: PR-AUC на обучающей части (подвыборка для скорости), на валидации и на test
    sub = np.random.default_rng(seed).choice(len(Xf), min(len(Xf), 150_000), replace=False)
    ap_fit = average_precision_score(yf[sub], model.predict_proba(Xf[sub])[:, 1])
    ap_val = average_precision_score(yv, p_val)
    ap_te = average_precision_score(yt, p_te)

    prec, rec, thr = precision_recall_curve(yv, p_val)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    th = thr[int(np.argmax(f1))]
    if cap is not None and day is not None:  # лимит службы мониторинга (доля операций в сутки)
        if rank == "money":  # ожидаемая сумма мошенничества
            score = p_te * amt
        elif rank == "cost" and costs:  # ожидаемая экономия: p * (сумма + штраф за пропуск + ложная блокировка)
            score = p_te * (amt + costs[1] + costs[2])
        else:
            score = p_te
        pred = capacity_flags(score, day, cap)
    else:  # без лимита: порог по максимуму F1 на валидации
        pred = p_te >= th
    tp = int((pred & (yt == 1)).sum())
    precision = tp / max(int(pred.sum()), 1)
    recall = tp / max(int((yt == 1).sum()), 1)
    out = {
        "PR-AUC": ap_te,
        "PR-AUC_fit": ap_fit,
        "PR-AUC_val": ap_val,
        "gap_fit_val": ap_fit - ap_val,   # большой = переобучение
        "gap_val_test": ap_val - ap_te,   # большой = смещение данных во времени или подбор по val
        "ROC-AUC": roc_auc_score(yt, p_te),
        "precision": precision,
        "recall": recall,
        "F1": 2 * precision * recall / max(precision + recall, 1e-9),
        "fit_sec": fit_sec,               # время обучения модели, секунды
        "sec": time.time() - t0,          # обучение + предсказания + расчёт метрик
        **money_stats(pred, yt, amt, cc),
        **cost_stats(pred, yt, amt, costs),
    }
    if caps and day is not None:  # кривая precision / recall / выгода по набору лимитов (служебное поле)
        out["_curve"] = curve_rows(p_te, yt, amt, day, caps, rank, costs)
    return out


# ----------------------------------------------------------------------------
# Основной сценарий
# ----------------------------------------------------------------------------
def parse_sets(raw_sets):
    sets = {}
    for s in raw_sets or []:
        name, toks = s.split("=", 1)
        toks = [t.strip() for t in toks.split(",") if t.strip()]
        bad = [t for t in toks if t not in ALL_TOKENS]
        if bad:
            raise SystemExit(f"Неизвестные признаки: {bad}\nДоступны: {ALL_TOKENS}")
        sets[name.strip()] = toks
    if not sets:  # по умолчанию: все обычные признаки против них же + FP
        sets = {"base": NUMERIC + CATEG, "base+fp": NUMERIC + CATEG + FP}
    return sets


AMT_LABELS = ["<10", "10-50", "50-100", "100-200", "200-300", "300-600", "600-1000", ">1000"]
AMT_BINS = [-np.inf, 10, 50, 100, 200, 300, 600, 1000, np.inf]


def make_weights(X_tr, cat_w, amt_w):
    """Вес обучающей строки = вес категории покупки * вес размера платежа (по умолчанию 1)."""
    w = np.ones(len(X_tr), np.float32)
    for cat, v in (cat_w or {}).items():
        col = f"category_{cat}"
        if col not in X_tr.columns:
            raise SystemExit(f"В данных нет категории '{cat}'")
        w[X_tr[col].to_numpy() == 1] *= v
    if amt_w:
        bins = pd.cut(X_tr["amt"], AMT_BINS, labels=AMT_LABELS).astype(str).to_numpy()
        for lab, v in amt_w.items():
            w[bins == lab] *= v
    return w


def run(train_path, test_path, sets, models=("lr", "rf", "xgb", "lgbm"), max_train_rows=300_000,
        n_splits=5, mine_kw=None, ablation=False, ablation_model="lgbm", seed=42, n_jobs=-1,
        scenarios=None, cap=0.003, rank_by="prob", costs=(3, 5, 25),
        caps=None, curve_path=None, card_fp=True):
    """Возвращает (results, ablation_df). results: метрики по (сценарий весов, набор, модель)."""
    mine_kw = mine_kw or dict(min_support=0.05, max_len=4, min_cover=100, top_n=30, min_lift=3.0)
    scenarios = scenarios or {"plain": {"cat": {}, "amt": {}}}
    X_tr, y_tr, X_te, y_te, groups, te_info = prepare(train_path, test_path, max_train_rows,
                                                      n_splits, mine_kw, seed, card_fp)
    amt_te, cc_te = te_info["amt"].to_numpy(), te_info["cc_num"].to_numpy()
    day_te = te_info["day"].to_numpy()
    if cap:
        per_day = pd.Series(day_te).value_counts()
        print(f"Лимит мониторинга {cap:.2%} операций в сутки: суток в test {len(per_day)}, "
              f"в среднем {per_day.mean():.0f} операций/сутки, всего можно проверить "
              f"{int(np.floor(cap * per_day + 1e-9).sum()):,} операций (способ выбора: {rank_by})")
    idx_fit, idx_val = train_test_split(np.arange(len(y_tr)), test_size=0.2,
                                        stratify=y_tr, random_state=seed)

    def cols_of(tokens):
        return [c for t in tokens for c in groups[t]]

    def score(tokens, key, w):
        cols = cols_of(tokens)
        a, b = X_tr[cols].to_numpy(), X_te[cols].to_numpy()
        return evaluate(key, a[idx_fit], y_tr[idx_fit], a[idx_val], y_tr[idx_val],
                        b, y_te, seed, n_jobs, wf=w[idx_fit], amt=amt_te, cc=cc_te,
                        day=day_te, cap=cap or None, rank=rank_by, costs=costs, caps=caps)

    rows, curve_all, w_first = [], [], None
    for sc_name, sc in scenarios.items():
        w = make_weights(X_tr, sc.get("cat"), sc.get("amt"))
        w_first = w if w_first is None else w_first
        for name, tokens in sets.items():
            for key in models:
                print(f"[{sc_name}] [{name}] {key} ...", end=" ", flush=True)
                m = score(tokens, key, w)
                curve_all += [{"scenario": sc_name, "set": name, "model": key, **r} for r in m.pop("_curve", [])]
                print(f"PR-AUC fit/val/test={m['PR-AUC_fit']:.3f}/{m['PR-AUC_val']:.3f}/{m['PR-AUC']:.3f}  обнаружено={m['detected']}  "
                      f"сэкономлено={m['saved_money']:,.0f}  выгода={m.get('net_benefit', 0):,.0f}  ({m['sec']:.0f}s)")
                rows.append({"scenario": sc_name, "set": name, "model": key,
                             "cat_weights": str(sc.get("cat") or {}),
                             "amt_weights": str(sc.get("amt") or {}), **m})
    res = pd.DataFrame(rows)
    if curve_path and curve_all:
        pd.DataFrame(curve_all).to_csv(curve_path, index=False)

    abl = None
    if ablation:  # абляция - на весах первого сценария
        first_name, first_tokens = next(iter(sets.items()))
        full = score(first_tokens, ablation_model, w_first)["PR-AUC"]
        out = []
        for t in first_tokens:
            rest = [x for x in first_tokens if x != t]
            if not rest:
                continue
            print(f"[ablation] без {t} ...", flush=True)
            out.append({"removed": t, "PR-AUC": score(rest, ablation_model, w_first)["PR-AUC"]})
        abl = pd.DataFrame(out)
        abl["delta_vs_full"] = abl["PR-AUC"] - full
        abl = abl.sort_values("delta_vs_full").reset_index(drop=True)  # самые важные сверху
        abl.attrs["full"] = full
    return res, abl


KNOWN_CATEGORIES = ["misc_net", "grocery_pos", "entertainment", "gas_transport", "misc_pos",
                    "grocery_net", "shopping_net", "shopping_pos", "food_dining", "personal_care",
                    "health_fitness", "travel", "kids_pets", "home"]


def parse_weights(s, valid, what):
    """'a=3,b=2' -> {'a': 3.0, 'b': 2.0} с проверкой названий."""
    out = {}
    for part in s.split(","):
        if part.strip():
            k, v = part.rsplit("=", 1)
            if k.strip() not in valid:
                raise SystemExit(f"Неизвестный {what} '{k}'. Доступны: {valid}")
            out[k.strip()] = float(v)
    return out


def parse_scenario(text):
    """'имя|cat:shopping_net=3,grocery_pos=2|amt:>1000=3' -> (имя, {'cat': {...}, 'amt': {...}})"""
    name, *parts = [p.strip() for p in text.split("|")]
    sc = {"cat": {}, "amt": {}}
    for p in parts:
        kind, _, body = p.partition(":")
        if kind == "cat":
            sc["cat"] = parse_weights(body, KNOWN_CATEGORIES, "категория")
        elif kind == "amt":
            sc["amt"] = parse_weights(body, AMT_LABELS, "размер платежа")
        else:
            raise SystemExit(f"В сценарии '{name}' ожидается 'cat:...' или 'amt:...', а не '{p}'")
    return name, sc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--set", action="append", help='"имя=признак1,признак2,..." (можно несколько раз)')
    ap.add_argument("--models", nargs="+", default=["lr", "rf", "xgb", "lgbm"],
                    choices=["lr", "rf", "xgb", "lgbm"])
    ap.add_argument("--max_train_rows", type=int, default=300_000,
                    help="подвыборка train для скорости (0 = всё)")
    ap.add_argument("--min_support", type=float, default=0.05)
    ap.add_argument("--min_lift", type=float, default=3.0)
    ap.add_argument("--min_cover", type=int, default=100)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--ablation_model", default="lgbm", choices=["lr", "rf", "xgb", "lgbm"])
    ap.add_argument("--scenario", action="append", default=[],
                    help='сценарий весов: "имя|cat:shopping_net=3,grocery_pos=2|amt:>1000=3" '
                         '(можно несколько раз; сценарий plain без весов добавляется сам)')
    ap.add_argument("--no_baseline", action="store_true", help="не добавлять сценарий plain")
    ap.add_argument("--out_prefix", default="results",
                    help="префикс файла результатов: <prefix>_summary.csv")
    ap.add_argument("--cap", type=float, default=0.003,
                    help="лимит службы мониторинга: доля операций в сутки (0.003 = 0,3%%); 0 = без лимита")
    ap.add_argument("--rank_by", choices=["prob", "money", "cost"], default="prob",
                    help="чем ранжировать операции для проверки: prob = вероятность мошенничества, "
                         "money = вероятность * сумма, cost = вероятность * (сумма + штрафы), "
                         "т.е. по ожидаемой экономии")
    ap.add_argument("--cost_review", type=float, default=3, help="стоимость проверки одной операции, у.е.")
    ap.add_argument("--cost_missed_fixed", type=float, default=5,
                    help="фиксированная добавка к сумме пропущенного мошенничества, у.е.")
    ap.add_argument("--cost_false_block", type=float, default=25,
                    help="потери от ложной блокировки честной операции, у.е.")
    ap.add_argument("--cap_grid", type=float, nargs="+", default=[0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02],
                    help="набор лимитов проверок (доли операций в сутки) для кривой precision/recall/выгода; "
                         "результат в <prefix>_curve.csv")
    ap.add_argument("--no_card_fp", action="store_true",
                    help="не включать личные признаки карты в паттерны FP-Growth (для сравнения)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    scenarios = {} if args.no_baseline else {"plain": {"cat": {}, "amt": {}}}
    for text in args.scenario:
        name, sc = parse_scenario(text)
        scenarios[name] = sc

    mine_kw = dict(min_support=args.min_support, max_len=4, min_cover=args.min_cover,
                   top_n=30, min_lift=args.min_lift)
    res, abl = run(args.train, args.test, parse_sets(args.set), args.models, args.max_train_rows,
                   mine_kw=mine_kw, ablation=args.ablation, ablation_model=args.ablation_model,
                   seed=args.seed, scenarios=scenarios, cap=args.cap, rank_by=args.rank_by,
                   costs=(args.cost_review, args.cost_missed_fixed, args.cost_false_block),
                   caps=args.cap_grid, curve_path=f"{args.out_prefix}_curve.csv", card_fp=not args.no_card_fp)

    pd.set_option("display.width", 220)
    res.to_csv(f"{args.out_prefix}_summary.csv", index=False)  # единственный файл: все сценарии и наборы

    order = list(res["scenario"].unique())
    for metric in ["PR-AUC", "gap_fit_val", "gap_val_test", "precision", "detected", "saved_money", "cost_total", "net_benefit", "flagged"]:
        pv = res.pivot_table(index=["set", "model"], columns="scenario", values=metric)[order]
        dec = 0 if any(k in metric for k in ("money", "cost", "benefit")) else 4
        print(f"\n=== {metric} на test (строки: набор, модель; столбцы: сценарий весов) ===")
        print(pv.round(dec).to_string())
        if pv.shape[1] > 1 and metric in ("PR-AUC", "detected", "net_benefit", "cost_total"):
            print(f"--- разница относительно '{order[0]}' ---")
            print(pv.sub(pv.iloc[:, 0], axis=0).iloc[:, 1:].round(dec).to_string())
    if abl is not None:
        print(f"\n=== Абляция ({args.ablation_model}), PR-AUC полной модели = {abl.attrs['full']:.4f} ===")
        print("(чем отрицательнее delta, тем важнее убранный признак)")
        print(abl.round(4).to_string(index=False))
    print("\nФайлы с результатами:", f"{args.out_prefix}_summary.csv, {args.out_prefix}_curve.csv")


if __name__ == "__main__":
    main()
