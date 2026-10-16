"""
Replay script to stream holdout transactions to the FastAPI service.
Course: Feature Engineering - Project 3 "Real-time Anti-fraud"
Usage:
    python replay.py --count 100 --delay 0.1
"""

import argparse
import time
import random
import requests
import pandas as pd
import numpy as np

API_URL = "http://localhost:8000"

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
            "cc_num": str(np.random.randint(100000000000, 999999999999, dtype='int64')),            "category": cat,
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

def run_replay(count, delay):
    print(f"🚀 Запуск симуляции потока транзакций: {count} операций с задержкой {delay}с...")
    
    samples = generate_holdout_samples(count)
    sent_ok = 0
    fraud_detected = 0

    for i, item in enumerate(samples, 1):
        is_fraud_fact = item.pop("is_fraud")
        
        try:
            # 1. Predict
            r_pred = requests.post(f"{API_URL}/predict", json=item, timeout=2)
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
                requests.post(f"{API_URL}/label", json={"transaction_id": tx_id, "is_fraud": is_fraud_fact}, timeout=2)
                
                sent_ok += 1
                print(f"[{i:03d}/{count}] Tx: {tx_id[:8]} | Amt: {item['amt']:7.2f} у.е. | Модель: {status_str} | Факт: {is_fraud_fact}")
            else:
                print(f"[{i:03d}/{count}] Ошибка API {r_pred.status_code}")
        except Exception as e:
            print(f"[{i:03d}/{count}] Ошибка подключения: {e}")

        time.sleep(delay)

    print("\n" + "="*50)
    print(f"🎉 Симуляция завершена! Успешно отправлено: {sent_ok}/{count}. Модель заблокировала: {fraud_detected} операций.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay holdout stream to Anti-Fraud API")
    parser.add_argument("--count", type=int, default=50, help="Количество операций")
    parser.add_argument("--delay", type=float, default=0.1, help="Задержка между операциями в сек")
    args = parser.parse_args()
    
    run_replay(args.count, args.delay)
