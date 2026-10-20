"""
Streamlit Operator Interface for Anti-Fraud Real-Time Scoring
Course: Feature Engineering - Project 3 "Real-time Anti-fraud"
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Антифрод: Интерфейс Оператора",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Интерфейс Оператора Службы Мониторинга")
st.caption("Проверка транзакций в реальном времени через FastAPI Сервис Скоринга")

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------
@st.cache_data
def load_holdout_sample():
    """Генерирует или загружает синтетическую отложенную выборку holdout.csv"""
    np.random.seed(42)
    categories = ["grocery_pos", "shopping_net", "entertainment", "gas_transport", "food_dining", "travel"]
    n_rows = 150
    
    data = []
    for i in range(n_rows):
        is_f = 1 if i % 15 == 0 else 0
        amt = np.random.uniform(500, 3000) if is_f else np.random.uniform(5, 200)
        cat = "shopping_net" if is_f else np.random.choice(categories)
        data.append({
            "trans_date_trans_time": f"2020-06-21 {np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}:{np.random.randint(0,60):02d}",
            "cc_num": str(np.random.randint(100000000000, 999999999999, dtype='int64')),            "category": cat,
            "amt": round(amt, 2),
            "gender": np.random.choice(["M", "F"]),
            "dob": "1985-05-15",
            "lat": 40.7128 + np.random.normal(0, 0.05 if not is_f else 0.5),
            "long": -74.0060 + np.random.normal(0, 0.05 if not is_f else 0.5),
            "merch_lat": 40.7128,
            "merch_long": -74.0060,
            "city_pop": 150000,
            "state": "NY",
            "is_fraud": is_f
        })
    return pd.DataFrame(data)

holdout_df = load_holdout_sample()

def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200, r.json()
    except Exception:
        return False, None

api_alive, health_info = check_api_health()

if not api_alive:
    st.error("⚠️ **Сервис скоринга недоступен!** Запустите FastAPI: `uvicorn service.main:app --reload --port 8000`")
else:
    st.success(f"✅ Сервис доступен (Порог: {health_info.get('current_threshold', 0.5):.2f})")

# ----------------------------------------------------------------------------
# Sidebar Options
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Панель управления")
mode = st.sidebar.radio(
    "Выберите режим работы:",
    ["1. Ввод одиночной операции", "2. Отложенная выборка (Holdout)", "3. Серия операций (Replay)"]
)

# ----------------------------------------------------------------------------
# Mode 1: Manual Input
# ----------------------------------------------------------------------------
if mode == "1. Ввод одиночной операции":
    st.subheader("📝 Ввод параметров транзакции вручную")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        amt = st.number_input("Сумма транзакции (у.е.)", min_value=0.1, value=120.0, step=10.0)
        category = st.selectbox("Категория мерчанта", ["grocery_pos", "shopping_net", "food_dining", "gas_transport", "travel", "misc_net"])
        cc_num = st.text_input("Номер карты (cc_num)", value="453210984321")
    
    with col2:
        dt_str = st.text_input("Дата и время", value="2020-06-21 02:15:00")
        gender = st.selectbox("Пол клиента", ["M", "F"])
        dob = st.text_input("Дата рождения", value="1988-04-12")

    with col3:
        lat = st.number_input("Широта клиента (lat)", value=40.7128)
        long = st.number_input("Долгота клиента (long)", value=-74.0060)
        merch_lat = st.number_input("Широта мерчанта", value=42.5000)
        merch_long = st.number_input("Долгота мерчанта", value=-71.1000)

    if st.button("🚀 Отправить на скоринг", type="primary"):
        payload = {
            "trans_date_trans_time": dt_str,
            "cc_num": cc_num,
            "category": category,
            "amt": amt,
            "gender": gender,
            "dob": dob,
            "lat": lat,
            "long": long,
            "merch_lat": merch_lat,
            "merch_long": merch_long,
            "city_pop": 150000,
            "state": "NY"
        }
        
        with st.spinner("Запрос к FastAPI..."):
            res = requests.post(f"{API_URL}/predict", json=payload)
            if res.status_code == 200:
                data = res.json()
                st.markdown("---")
                st.subheader("Результат скоринга:")
                
                score = data["fraud_score"]
                is_fraud = data["is_fraud_pred"]
                thresh = data["threshold"]
                reasons = data["reasons"]

                m1, m2, m3 = st.columns(3)
                m1.metric("Transaction ID", data["transaction_id"][:8] + "...")
                m2.metric("Вероятность фрода", f"{score * 100:.1f}%")
                m3.metric("Порог блокировки", f"{thresh * 100:.0f}%")

                if is_fraud:
                    st.error("🚨 **РЕШЕНИЕ: ФРОД / БЛОКИРОВКА!** Операция отправлена аналитику.")
                else:
                    st.success("✅ **РЕШЕНИЕ: ЧЕСТНАЯ ОПЕРАЦИЯ.** Пропустить транзакцию.")

                st.markdown("**Ключевые факторы риска (Reason Logs):**")
                for r in reasons:
                    st.write(f"• {r}")
            else:
                st.error(f"Ошибка сервера: {res.status_code} - {res.text}")

# ----------------------------------------------------------------------------
# Mode 2: Holdout Row Picker
# ----------------------------------------------------------------------------
elif mode == "2. Отложенная выборка (Holdout)":
    st.subheader("📂 Выбор строки из отложенной выборки (Holdout Dataset)")
    
    row_idx = st.number_input("Индекс строки holdout", min_value=0, max_value=len(holdout_df)-1, value=0)
    
    if st.button("🎲 Выбрать случайную транзакцию"):
        row_idx = np.random.randint(0, len(holdout_df))
        st.info(f"Выбрана случайная строка #{row_idx}")

    selected_row = holdout_df.iloc[row_idx].to_dict()
    fact_is_fraud = selected_row.pop("is_fraud")

    st.write("**Параметры выбранной строки:**")
    st.json(selected_row)

    if st.button("🔍 Оценить моделью", type="primary"):
        res = requests.post(f"{API_URL}/predict", json=selected_row)
        if res.status_code == 200:
            data = res.json()
            tx_id = data["transaction_id"]
            pred = data["is_fraud_pred"]
            score = data["fraud_score"]

            col_a, col_b = st.columns(2)
            with col_a:
                if pred == 1:
                    st.error(f"🚨 Предсказание модели: **ФРОД** (Скор: {score:.4f})")
                else:
                    st.success(f"✅ Предсказание модели: **ЧЕСТНАЯ** (Скор: {score:.4f})")

            with col_b:
                if fact_is_fraud == 1:
                    st.warning("🎯 Фактическая метка (Fact): **ФРОД (1)**")
                else:
                    st.info("🎯 Фактическая метка (Fact): **ЧЕСТНАЯ (0)**")

            # Сверка
            if pred == fact_is_fraud:
                st.success("🎉 **Решение модели СОВПАЛО с фактом!**")
            else:
                st.error("⚠️ **Ошибка первого/второго рода (Несовпадение с фактом)!**")

            # Отправка факта в /label
            label_res = requests.post(f"{API_URL}/label", json={"transaction_id": tx_id, "is_fraud": fact_is_fraud})
            if label_res.status_code == 200:
                st.caption(f"✓ Метка {fact_is_fraud} отправлена в сервис по /label для транзакции {tx_id[:8]}")

# ----------------------------------------------------------------------------
# Mode 3: Batch Replay
# ----------------------------------------------------------------------------
elif mode == "3. Серия операций (Replay)":
    st.subheader("⚡ Подача серии операций для эмуляции реального потока")
    st.caption("Посылает N транзакций подряд в FastAPI с задержкой для демонстрации мониторинга.")

    n_ops = st.slider("Количество операций в серии", min_value=10, max_value=100, value=30, step=10)
    delay = st.slider("Задержка между отправками (сек)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    if st.button("▶️ Запустить серию транзакций", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results_list = []
        
        for i in range(n_ops):
            row = holdout_df.iloc[i % len(holdout_df)].to_dict()
            fact_is_fraud = row.pop("is_fraud")

            # 1. Predict
            r_pred = requests.post(f"{API_URL}/predict", json=row)
            if r_pred.status_code == 200:
                p_data = r_pred.json()
                tx_id = p_data["transaction_id"]
                pred = p_data["is_fraud_pred"]
                
                # 2. Label
                requests.post(f"{API_URL}/label", json={"transaction_id": tx_id, "is_fraud": fact_is_fraud})
                
                results_list.append({
                    "№": i + 1,
                    "Amount": row["amt"],
                    "Category": row["category"],
                    "Score": p_data["fraud_score"],
                    "Pred": "🚨 Fraud" if pred else "✅ Legit",
                    "Fact": "1 (Fraud)" if fact_is_fraud else "0 (Legit)"
                })

            progress_bar.progress((i + 1) / n_ops)
            status_text.text(f"Обработано {i + 1} / {n_ops} операций...")
            time.sleep(delay)

        st.success(f"✅ Серия из {n_ops} операций успешно отправлена в сервис!")
        st.dataframe(pd.DataFrame(results_list), height=300)
