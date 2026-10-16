"""
FP-Growth для генерации новых признаков на Credit Card Transactions Fraud Detection Dataset.

Идея:
  1. Признаки дискретизируются и превращаются в "items" вида "amt_bin=300-1000".
  2. FP-Growth запускается на мошеннических транзакциях train (их мало, поэтому быстро)
     и находит частые комбинации items (паттерны).
  3. Для каждого паттерна считаем на ВСЁМ train: покрытие, precision (доля fraud), lift.
     Оставляем TOP_N самых "мошеннических" паттернов.
  4. Новые признаки для любой выборки (train/test):
        pat_k       - 1, если транзакция содержит паттерн k
        pat_count   - сколько паттернов сработало
        pat_max_lift- максимальный lift среди сработавших паттернов

Паттерны ищутся ТОЛЬКО на train, к test они лишь применяются -> нет утечки из test.

Запуск:
    python fpgrowth_features.py --train fraudTrain.csv --test fraudTest.csv --out out_dir
"""
import argparse
import os

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth


# ----------------------------------------------------------------------------
# 1. Дискретизация -> бинарная матрица items
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def make_items(df: pd.DataFrame) -> pd.DataFrame:
    """Возвращает DataFrame bool: столбец = один item ('признак=значение')."""
    ts = pd.to_datetime(df["trans_date_trans_time"])
    hour = ts.dt.hour
    age = (ts - pd.to_datetime(df["dob"])).dt.days / 365.25
    dist = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    cats = pd.DataFrame(index=df.index)
    cats["category"] = df["category"].astype(str)
    cats["gender"] = df["gender"].astype(str)
    cats["amt_bin"] = pd.cut(
        df["amt"], [-np.inf, 10, 50, 100, 200, 300, 600, 1000, np.inf],
        labels=["<10", "10-50", "50-100", "100-200", "200-300", "300-600", "600-1000", ">1000"],
    ).astype(str)
    cats["hour_bin"] = pd.cut(
        hour.where(hour >= 4, hour + 24),  # 0-3 -> 24-27, чтобы ночь была непрерывной
        [3, 9, 15, 21, 27],
        labels=["morning_04-09", "day_10-15", "evening_16-21", "night_22-03"],
    ).astype(str)
    cats["weekend"] = np.where(ts.dt.dayofweek >= 5, "yes", "no")
    cats["age_bin"] = pd.cut(age, [0, 25, 40, 55, 70, 120],
                             labels=["<25", "25-40", "40-55", "55-70", "70+"]).astype(str)
    cats["dist_bin"] = pd.cut(dist, [-1, 25, 50, 100, 200, np.inf],
                              labels=["<25km", "25-50km", "50-100km", "100-200km", ">200km"]).astype(str)
    cats["city_pop_bin"] = pd.cut(df["city_pop"], [-1, 1_000, 10_000, 100_000, np.inf],
                                  labels=["<1k", "1k-10k", "10k-100k", ">100k"]).astype(str)

    # one-hot: колонка "признак=значение"
    return pd.get_dummies(cats, prefix_sep="=").astype(bool)


# ----------------------------------------------------------------------------
# 2. Поиск паттернов
# ----------------------------------------------------------------------------
def mine_patterns(items_train: pd.DataFrame, y_train: np.ndarray,
                  min_support=0.05, max_len=4, min_cover=300, top_n=30, min_lift=3.0):
    """FP-Growth по fraud-транзакциям + оценка паттернов на всём train."""
    fraud_items = items_train[y_train == 1]
    print(f"[FP-Growth] fraud-транзакций: {len(fraud_items):,}, items: {fraud_items.shape[1]}")

    freq = fpgrowth(fraud_items, min_support=min_support, use_colnames=True, max_len=max_len)
    print(f"[FP-Growth] найдено частых наборов: {len(freq):,}")

    base_rate = y_train.mean()
    arr = items_train.to_numpy()
    col_idx = {c: i for i, c in enumerate(items_train.columns)}

    rows = []
    for itemset, sup_fraud in zip(freq["itemsets"], freq["support"]):
        idx = [col_idx[c] for c in itemset]
        mask = arr[:, idx].all(axis=1)
        cover = int(mask.sum())
        if cover < min_cover:
            continue
        precision = y_train[mask].mean()
        rows.append({
            "pattern": tuple(sorted(itemset)),
            "len": len(itemset),
            "support_fraud": sup_fraud,        # какую долю fraud покрывает паттерн (recall)
            "cover_all": cover,                # сколько всего транзакций подходит
            "precision": precision,            # доля fraud среди подходящих
            "lift": precision / base_rate,
        })

    pats = pd.DataFrame(rows)
    if pats.empty:
        raise RuntimeError("Паттернов не найдено: уменьшите min_support или min_cover.")
    pats = pats[pats["lift"] >= min_lift].sort_values(["lift", "cover_all"], ascending=False)
    pats = pats.head(top_n).reset_index(drop=True)
    pats["name"] = [f"pat_{i:02d}" for i in range(len(pats))]
    print(f"[FP-Growth] оставлено паттернов (lift >= {min_lift}): {len(pats)}")
    return pats


# ----------------------------------------------------------------------------
# 3. Применение паттернов -> новые признаки
# ----------------------------------------------------------------------------
def apply_patterns(items: pd.DataFrame, pats: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=items.index)
    for _, p in pats.iterrows():
        cols = [c for c in p["pattern"] if c in items.columns]
        if len(cols) < len(p["pattern"]):  # item не встретился в этой выборке
            out[p["name"]] = 0
        else:
            out[p["name"]] = items[cols].all(axis=1).astype(np.int8)

    pat_cols = list(pats["name"])
    out["pat_count"] = out[pat_cols].sum(axis=1)
    lifts = pats["lift"].to_numpy()
    out["pat_max_lift"] = (out[pat_cols].to_numpy() * lifts).max(axis=1)
    return out


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--out", default="fp_out")
    ap.add_argument("--min_support", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=4)
    ap.add_argument("--min_cover", type=int, default=300)
    ap.add_argument("--min_lift", type=float, default=3.0)
    ap.add_argument("--top_n", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train = pd.read_csv(args.train, index_col=0)
    items_tr = make_items(train)
    y_tr = train["is_fraud"].to_numpy()

    pats = mine_patterns(items_tr, y_tr, args.min_support, args.max_len,
                         args.min_cover, args.top_n, args.min_lift)
    pats_print = pats.copy()
    pats_print["pattern"] = pats_print["pattern"].apply(lambda t: " & ".join(t))
    pats_print.to_csv(os.path.join(args.out, "patterns.csv"), index=False)
    print(pats_print[["name", "pattern", "cover_all", "precision", "lift"]]
          .head(10).to_string(index=False))

    feats_tr = apply_patterns(items_tr, pats)
    pd.concat([train, feats_tr], axis=1).to_csv(os.path.join(args.out, "train_fp.csv"))

    if args.test:
        test = pd.read_csv(args.test, index_col=0)
        # приводим items теста к колонкам train (недостающие = False)
        items_te = make_items(test).reindex(columns=items_tr.columns, fill_value=False)
        feats_te = apply_patterns(items_te, pats)
        pd.concat([test, feats_te], axis=1).to_csv(os.path.join(args.out, "test_fp.csv"))

    print(f"Готово. Результаты в: {args.out}/")


if __name__ == "__main__":
    main()
