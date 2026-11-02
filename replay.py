"""
Replay script to stream holdout transactions to the FastAPI service.
Course: Feature Engineering - Project 3 "Real-time Anti-fraud"
Usage:
    python replay.py --count 100 --delay 0.1                          # синтетические операции
    python replay.py --csv data/fraudTest.csv --count 500 --delay 0   # реальные операции из test по времени
"""

import argparse
import os
import time
import random
import requests
import pandas as pd
import numpy as np

API_URL = os.getenv("API_URL", "http://localhost:8000")

CSV_COLS = ["trans_date_trans_time", "cc_num", "category", "amt", "gender", "dob", "lat", "long",
            "merch_lat", "merch_long", "city_pop", "state", "merchant", "is_fraud"]


def generate_holdout_samples(n=200):
    np.random.seed(42)
    random.seed(42)
    categories = ["grocery_pos", "shopping_net", "entertainment", "gas_transport", "food_dining", "travel"]
    data = []
    for i in range(n):
        is_f = 1 if i % 12 == 0 else 0
        amt = float(np.random.uniform(600, 3500) if is_f else np.random.uniform(5, 250))
        cat = "shopping_net" if is_f else str(np.random.choice(categories))
        data.append({
            "trans_date_trans_time": f"2020-06-21 {np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}:{np.random.randint(0,60):02d}",
            "cc_num": str(np.random.randint(100000000000, 999999999999, dtype='int64')),
            "category": cat,
            "amt": round(amt, 2),
            "gender": str(np.random.choice(["M", "F"])),
            "dob": "1985-05-15",
            "lat": float(40.7128 + (0.5 if is_f else 0.01)),
            "long": float(-74.0060),
            "merch_lat": float(40.7128),
            "merch_long": float(-74.0060),
            "city_pop": 150000,
            "state": "NY",
            "is_fraud": is_f
        })
    return data


def load_csv_samples(path, n, start=0):
    """Реальные операции из fraudTest.csv в порядке времени (нужно для карточных признаков)."""
    df = pd.read_csv(path, usecols=CSV_COLS).sort_values("trans_date_trans_time", kind="stable")
    df = df.iloc[start:start + n]
    data = []
    for r in df.to_dict("records"):
        r["cc_num"] = str(r["cc_num"])
        r["amt"] = float(r["amt"])
        r["is_fraud"] = int(r["is_fraud"])
        data.append(r)
    return data


def run_replay(count, delay, csv=None, start=0):
    print(f"🚀 Запуск симуляции потока транзакций: {count} операций с задержкой {delay}с...")

    samples = load_csv_samples(csv, count, start) if csv else generate_holdout_samples(count)
    count = len(samples)
    sent_ok = 0
    fraud_detected = 0
    tp = fp = fn = tn = 0  # матрица ошибок по фактическим меткам серии

    for i, item in enumerate(samples, 1):
        is_fraud_fact = item.pop("is_fraud")
        
        try:
            # 1. Predict
            r_pred = requests.post(f"{API_URL}/predict", json=item, timeout=5)
            if r_pred.status_code == 200:
                p_data = r_pred.json()
                tx_id = p_data["transaction_id"]
                score = p_data["fraud_score"]
                pred = p_data["is_fraud_pred"]
                
                if pred == 1:
                    fraud_detected += 1
                    status_str = f"🚨 ФРОД ({score*100:.1f}%)"
                else:
                    status_str = f"✅ ЧЕСТНАЯ ({score*100:.1f}%)"

                # 2. Label
                requests.post(f"{API_URL}/label", json={"transaction_id": tx_id, "is_fraud": is_fraud_fact}, timeout=5)
                
                if pred == 1 and is_fraud_fact == 1:
                    tp += 1      # фрод пойман
                elif pred == 1:
                    fp += 1      # честная операция заблокирована
                elif is_fraud_fact == 1:
                    fn += 1      # фрод пропущен
                else:
                    tn += 1      # честная пропущена верно

                sent_ok += 1
                print(f"[{i:03d}/{count}] Tx: {tx_id[:8]} | Amt: {item['amt']:7.2f} у.е. | Модель: {status_str} | Факт: {is_fraud_fact}")
            else:
                print(f"[{i:03d}/{count}] Ошибка API {r_pred.status_code}")
        except Exception as e:
            print(f"[{i:03d}/{count}] Ошибка подключения: {e}")

        time.sleep(delay)

    print("\n" + "="*50)
    print(f"🎉 Симуляция завершена! Успешно отправлено: {sent_ok}/{count}. Модель заблокировала: {fraud_detected} операций.")

    # --- качество решений модели в этой серии ---
    n = tp + fp + fn + tn
    if n:
        def pct(v):
            return f"{v:.1%}" if v is not None else "—"
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else None
        print(f"\nФактического фрода в серии: {tp + fn} из {n} ({(tp + fn) / n:.2%})")
        print(f"Поймано: {tp} | Ложных блокировок: {fp} | Пропущено фрода: {fn} | Верных пропусков: {tn}")
        print(f"Accuracy: {(tp + tn) / n:.2%} | Precision: {pct(precision)} | Recall: {pct(recall)} | F1: {pct(f1)}")
        if tp + fn == 0:
            print("В серии нет мошеннических операций, поэтому recall посчитать нельзя. Увеличьте --count или измените --start.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay holdout stream to Anti-Fraud API")
    parser.add_argument("--count", type=int, default=50, help="Количество операций")
    parser.add_argument("--delay", type=float, default=0.1, help="Задержка между операциями в сек")
    parser.add_argument("--csv", default=None, help="Путь к fraudTest.csv (реальные операции вместо синтетических)")
    parser.add_argument("--start", type=int, default=0, help="Смещение в отсортированном по времени CSV")
    args = parser.parse_args()
    
    run_replay(args.count, args.delay, args.csv, args.start)
