"""
Загрузка отложенной выборки (fraudTest.csv) в таблицу holdout_transactions PostgreSQL.
Идемпотентно: ключ trans_num, уже загруженные строки пропускаются (ON CONFLICT DO NOTHING).

Запуск:  python load_holdout.py --csv data/fraudTest.csv [--limit 100000]
"""
import argparse

import pandas as pd
from sqlalchemy import text

import db

COLS = ["trans_num", "trans_time", "cc_num", "merchant", "category", "amt", "gender", "dob", "lat", "long",
        "merch_lat", "merch_long", "city_pop", "state", "is_fraud"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/fraudTest.csv")
    ap.add_argument("--limit", type=int, default=0, help="0 = все строки")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype={"cc_num": str})
    if args.limit:
        df = df.head(args.limit)
    df = df.rename(columns={"trans_date_trans_time": "trans_time"}).drop_duplicates("trans_num")
    df["trans_time"] = pd.to_datetime(df["trans_time"]).astype(str)
    df["dob"] = pd.to_datetime(df["dob"]).dt.strftime("%Y-%m-%d")
    df["is_fraud"] = df["is_fraud"].astype(bool)
    data = [df[c].tolist() for c in COLS]  # tolist() отдаёт обычные типы Python

    with db.get_engine().begin() as conn:
        conn.execute(text("CREATE TEMP TABLE _holdout_stage (LIKE holdout_transactions INCLUDING DEFAULTS) ON COMMIT DROP"))
        with conn.connection.driver_connection.cursor() as cur:
            with cur.copy(f"COPY _holdout_stage ({', '.join(COLS)}) FROM STDIN") as cp:
                for row in zip(*data):
                    cp.write_row(row)
        res = conn.execute(text(f"INSERT INTO holdout_transactions ({', '.join(COLS)}) "
                                f"SELECT {', '.join(COLS)} FROM _holdout_stage ON CONFLICT (trans_num) DO NOTHING"))
        inserted = res.rowcount
        total = conn.execute(text("SELECT count(*) FROM holdout_transactions")).scalar_one()
    print(f"Прочитано строк: {len(df):,}. Добавлено новых: {inserted:,}. Всего в holdout_transactions: {total:,}.")


if __name__ == "__main__":
    main()
