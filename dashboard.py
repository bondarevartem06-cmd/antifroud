"""
Streamlit Monitoring Dashboard for Anti-Fraud Real-Time Scoring Service
Course: Feature Engineering - Project 3 "Real-time Anti-fraud"
"""

import os
import time
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Антифрод: Панель Мониторинга",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Панель Мониторинга Антифрод-Сервиса")
st.caption("Автоматический мониторинг KPI, деградации и качества работы модели в реальном времени")

def fetch_metrics():
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# Sidebar Controls
st.sidebar.header("⚙️ Настройки панели")
auto_refresh = st.sidebar.checkbox("Включить автообновление (каждые 5 сек)", value=True)
refresh_button = st.sidebar.button("🔄 Обновить сейчас")

# Dynamic Threshold Changer
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Динамический порог")
metrics_data = fetch_metrics()

if metrics_data:
    current_thresh = metrics_data.get("current_threshold", 0.5)
    current_thresh = min(max(round(current_thresh / 0.05) * 0.05, 0.05), 0.95)  # в границах и шаге слайдера
    new_thresh = st.sidebar.slider("Рабочий порог blocking threshold", 0.05, 0.95, float(current_thresh), 0.05)
    
    if st.sidebar.button("Применить новый порог"):
        r_th = requests.put(f"{API_URL}/threshold", json={"threshold": new_thresh})
        if r_th.status_code == 200:
            st.sidebar.success(f"Порог изменен на {new_thresh:.2f}")
            st.rerun()

# Container for Metrics
if not metrics_data:
    st.warning("⚠️ **Сервис скоринга недоступен!** Проверьте, запущен ли FastAPI на `http://localhost:8000`")
else:
    # ----------------------------------------------------------------------------
    # KPI Summary Tiles
    # ----------------------------------------------------------------------------
    st.subheader("📌 Ключевые показатели эффективности (KPI)")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    tot = metrics_data["total_requests"]
    err = metrics_data["error_count"]
    succ_rate = metrics_data["success_rate"]
    fr_model = metrics_data["fraud_rate_model"]
    fr_fact = metrics_data["fraud_rate_fact"]
    labeled = metrics_data["labeled_count"]

    kpi1.metric("Всего запросов", f"{tot}", delta=f"-{err} ошибок" if err > 0 else "Ошибок нет")
    
    # Success Rate Alarm
    succ_color = "normal" if succ_rate >= 95.0 else "inverse"
    kpi2.metric("Success Rate", f"{succ_rate:.1f}%", delta="ОК" if succ_rate >= 95 else "ТРЕВОГА!", delta_color=succ_color)
    
    kpi3.metric("Fraud Rate (Модель)", f"{fr_model:.2f}%", delta=f"Порог {metrics_data['current_threshold']:.2f}")
    kpi4.metric("Fraud Rate (Факт)", f"{fr_fact:.2f}%", delta=f"{labeled} размечено")

    # Alarm Banner if discrepancy is large
    if labeled >= 10 and abs(fr_model - fr_fact) > 2.0:
        st.warning(f"⚠️ **Внимание: Расхождение моделей!** Fraud Rate по модели ({fr_model:.2f}%) и по факту ({fr_fact:.2f}%) различаются. Возможно смещение распределения (Data Drift) или неоптимальный порог.")

    st.markdown("---")

    # ----------------------------------------------------------------------------
    # Charts: Time Series Analysis
    # ----------------------------------------------------------------------------
    col_chart1, col_chart2 = st.columns(2)
    
    ts_data = metrics_data.get("time_series", [])
    if ts_data:
        df_ts = pd.DataFrame(ts_data)
        
        with col_chart1:
            st.subheader("📈 Динамика Fraud Rate (Модель vs Факт)")
            st.line_chart(
                df_ts,
                x="batch_index",
                y=["fraud_rate_model", "fraud_rate_fact"],
                color=["#FF4B4B", "#1C83E1"]
            )
        
        with col_chart2:
            st.subheader("🛡️ Доступность сервиса (Success Rate %)")
            st.line_chart(
                df_ts,
                x="batch_index",
                y="success_rate",
                color="#00C853"
            )
    else:
        st.info("💡 Отправьте транзакции через интерфейс оператора, чтобы увидеть графики динамики.")

    st.markdown("---")

    # ----------------------------------------------------------------------------
    # Table of Recent Fraud Detections
    # ----------------------------------------------------------------------------
    st.subheader("🚨 Последние заблокированные операции (Recent Frauds)")
    recent_frauds = metrics_data.get("recent_frauds", [])
    
    if recent_frauds:
        df_rf = pd.DataFrame(recent_frauds)
        st.dataframe(
            df_rf[["timestamp", "amt", "category", "fraud_score", "reasons", "is_fraud_fact"]],
            column_config={
                "timestamp": "Время",
                "amt": st.column_config.NumberColumn("Сумма (у.е.)", format="%.2f"),
                "category": "Категория",
                "fraud_score": st.column_config.ProgressColumn("Скор Фрода", min_value=0.0, max_value=1.0, format="%.4f"),
                "reasons": "Причины блокировки (Reason Logs)",
                "is_fraud_fact": "Фактическая метка"
            },
            use_container_width=True,
            height=300
        )
    else:
        st.success("На данный момент заблокированных подозрительных операций нет.")

# Auto Refresh loop
if auto_refresh:
    time.sleep(5)
    st.rerun()
