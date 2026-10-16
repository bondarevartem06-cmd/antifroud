"""
FastAPI Anti-Fraud Real-Time Scoring Service
Course: Feature Engineering - Project 3 "Real-time Anti-fraud"
"""

import time
import math
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Real-Time Anti-Fraud Scoring API",
    description="API для скоринга транзакций в реальном времени, приема меток и сбора метрик мониторинга.",
    version="1.0.0"
)

# Enable CORS for browser frontend if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
DEFAULT_THRESHOLD = 0.50
CURRENT_THRESHOLD = DEFAULT_THRESHOLD

# In-memory transaction and metrics logs
# In production this would be backed by PostgreSQL/SQLite
TRANSACTION_LOG: List[Dict[str, Any]] = []
ERROR_LOG: List[Dict[str, Any]] = []
CARD_HISTORY: Dict[str, List[Dict[str, Any]]] = {}  # {cc_num: [tx1, tx2, ...]}

# ----------------------------------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------------------------------
class TransactionInput(BaseModel):
    trans_date_trans_time: str = Field(..., json_schema_extra={"example": "2020-06-21 12:14:25"})
    cc_num: str = Field(..., json_schema_extra={"example": "2291163834988"})
    category: str = Field(..., json_schema_extra={"example": "grocery_pos"})
    amt: float = Field(..., json_schema_extra={"example": "124.50"})
    gender: Optional[str] = Field("M", json_schema_extra={"example": "M"})
    dob: Optional[str] = Field("1988-04-12", json_schema_extra={"example": "1988-04-12"})
    lat: Optional[float] = Field(40.7128, json_schema_extra={"example": 40.7128})
    long: Optional[float] = Field(-74.0060, json_schema_extra={"example": -74.0060})
    merch_lat: Optional[float] = Field(40.7500, json_schema_extra={"example": 40.7500})
    merch_long: Optional[float] = Field(-73.9900, json_schema_extra={"example": -73.9900})
    city_pop: Optional[float] = Field(150000.0, json_schema_extra={"example": 150000.0})
    state: Optional[str] = Field("NY", json_schema_extra={"example": "NY"})

class PredictResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    is_fraud_pred: int
    threshold: float
    reasons: List[str]

class LabelInput(BaseModel):
    transaction_id: str
    is_fraud: int = Field(..., description="1 = Мошенничество, 0 = Честная операция")

class ThresholdInput(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0, description="Новый порог срабатывания (от 0 до 1)")

# ----------------------------------------------------------------------------
# Error Handlers
# ----------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    ERROR_LOG.append({
        "timestamp": datetime.now().isoformat(),
        "type": "422_validation_error",
        "detail": str(exc)
    })
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation Error", "errors": exc.errors()}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    ERROR_LOG.append({
        "timestamp": datetime.now().isoformat(),
        "type": "500_internal_error",
        "detail": str(exc)
    })
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )

# ----------------------------------------------------------------------------
# Helper Scoring Logic & Feature Calculation
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(math.radians, (float(lat1), float(lon1), float(lat2), float(lon2)))
        a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
        return 6371 * 2 * math.asin(math.sqrt(a))
    except Exception:
        return 0.0

def calculate_score_and_reasons(tx: TransactionInput) -> tuple[float, List[str]]:
    """
    Рассчитывает вероятность фрода и интерпретируемые причины (reasons 1..3).
    Включает признаки по истории карты (CARD_HISTORY) и гео-параметры.
    """
    reasons = []
    base_score = 0.02  # априорная вероятность фрода ~1%

    # 1. Анализ суммы
    amt = tx.amt
    if amt > 1000:
        base_score += 0.45
        reasons.append(f"Аномально высокая сумма платежа ({amt:.2f} у.е. > 1000)")
    elif amt > 300:
        base_score += 0.25
        reasons.append(f"Крупная сумма транзакции ({amt:.2f} у.е. > 300)")
    elif amt < 2.0:
        base_score += 0.10
        reasons.append("Микро-транзакция (возможная проверка карты)")

    # 2. Время суток
    try:
        dt = datetime.strptime(tx.trans_date_trans_time, "%Y-%m-%d %H:%M:%S")
        hour = dt.hour
        if 22 <= hour or hour <= 4:
            base_score += 0.20
            reasons.append(f"Ночная транзакция (время {hour:02d}:00)")
    except Exception:
        dt = datetime.now()

    # 3. Категория риска
    high_risk_categories = ["shopping_net", "grocery_pos", "misc_net", "travel"]
    if tx.category in high_risk_categories:
        base_score += 0.15
        reasons.append(f"Категория повышенного риска ({tx.category})")

    # 4. Расстояние до мерчанта (Haversine)
    dist = haversine_km(tx.lat, tx.long, tx.merch_lat, tx.merch_long)
    if dist > 200:
        base_score += 0.25
        reasons.append(f"Удаленная точка оплаты (расстояние {dist:.1f} км > 200км)")
    elif dist > 80:
        base_score += 0.10
        reasons.append(f"Повышенное расстояние до мерчанта ({dist:.1f} км)")

    # 5. Признаки по истории карты (в памяти)
    history = CARD_HISTORY.get(tx.cc_num, [])
    if len(history) > 0:
        # Проверка частоты транзакций за короткое время
        recent_txs = [h for h in history if (dt - h['time']).total_seconds() <= 3600]
        if len(recent_txs) >= 3:
            base_score += 0.30
            reasons.append(f"Всплеск активности: {len(recent_txs)} операций за последний час")

        # Отклонение от среднего чека по карте
        avg_amt = np.mean([h['amt'] for h in history])
        if avg_amt > 0 and amt > 3.5 * avg_amt:
            base_score += 0.20
            reasons.append(f"Превышение среднего чека клиента в {amt/avg_amt:.1f} раз")

    # Сохраняем операцию в историю карты
    CARD_HISTORY.setdefault(tx.cc_num, []).append({
        "time": dt,
        "amt": amt,
        "category": tx.category
    })

    # Ограничиваем вероятность диапазоном [0.001, 0.999]
    score = float(np.clip(base_score, 0.001, 0.999))
    
    if not reasons:
        reasons.append("Стандартный профиль операции (низкий риск)")

    return score, reasons[:3]

# ----------------------------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------------------------
@app.get("/health", summary="Проверка состояния сервиса")
def get_health():
    """Возвращает статус готовности модели и сервиса."""
    return {
        "status": "ok",
        "service": "antifraud_scoring",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": True,
        "current_threshold": CURRENT_THRESHOLD
    }

@app.post("/predict", response_model=PredictResponse, summary="Скоринг одиночной транзакции")
def predict_transaction(tx: TransactionInput):
    """
    Принимает одну транзакцию, рассчитывает признаки и возвращает решение и причины.
    """
    tx_id = str(uuid4())
    score, reasons = calculate_score_and_reasons(tx)
    is_fraud_pred = int(score >= CURRENT_THRESHOLD)

    log_entry = {
        "transaction_id": tx_id,
        "timestamp": datetime.now().isoformat(),
        "trans_date_trans_time": tx.trans_date_trans_time,
        "cc_num": tx.cc_num,
        "amt": tx.amt,
        "category": tx.category,
        "fraud_score": round(score, 4),
        "is_fraud_pred": is_fraud_pred,
        "threshold": CURRENT_THRESHOLD,
        "is_fraud_fact": None,  # заполняется позже через /label
        "reasons": reasons,
        "status": "ok"
    }
    TRANSACTION_LOG.append(log_entry)

    return PredictResponse(
        transaction_id=tx_id,
        fraud_score=round(score, 4),
        is_fraud_pred=is_fraud_pred,
        threshold=CURRENT_THRESHOLD,
        reasons=reasons
    )

@app.post("/label", summary="Обновление фактической метки фрода")
def label_transaction(payload: LabelInput):
    """
    Привязывает подтвержденный факт мошенничества (0 или 1) к ранее оцененной транзакции.
    """
    for entry in reversed(TRANSACTION_LOG):
        if entry["transaction_id"] == payload.transaction_id:
            entry["is_fraud_fact"] = payload.is_fraud
            return {
                "status": "ok",
                "message": f"Label {payload.is_fraud} attached to transaction {payload.transaction_id}"
            }
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Transaction ID {payload.transaction_id} not found in log."
    )

@app.put("/threshold", summary="Динамическое изменение порога")
def update_threshold(payload: ThresholdInput):
    """Изменяет рабочий порог принятия решений без перезапуска сервиса."""
    global CURRENT_THRESHOLD
    old_threshold = CURRENT_THRESHOLD
    CURRENT_THRESHOLD = payload.threshold
    return {
        "status": "ok",
        "old_threshold": old_threshold,
        "new_threshold": CURRENT_THRESHOLD
    }

@app.get("/metrics", summary="Метрики для панели мониторинга")
def get_metrics():
    """
    Возвращает сводную статистику работы сервиса для панели мониторинга.
    """
    total_requests = len(TRANSACTION_LOG) + len(ERROR_LOG)
    error_count = len(ERROR_LOG)
    success_requests = len(TRANSACTION_LOG)

    success_rate = (success_requests / total_requests * 100) if total_requests > 0 else 100.0

    # Fraud rate по модели
    model_positives = sum(1 for tx in TRANSACTION_LOG if tx["is_fraud_pred"] == 1)
    fraud_rate_model = (model_positives / success_requests * 100) if success_requests > 0 else 0.0

    # Fraud rate фактический (по размеченным)
    labeled_txs = [tx for tx in TRANSACTION_LOG if tx["is_fraud_fact"] is not None]
    fact_positives = sum(1 for tx in labeled_txs if tx["is_fraud_fact"] == 1)
    fraud_rate_fact = (fact_positives / len(labeled_txs) * 100) if len(labeled_txs) > 0 else 0.0

    # Формируем временной ряд (батчи по 10 транзакций)
    time_series = []
    batch_size = 10
    for i in range(0, len(TRANSACTION_LOG), batch_size):
        batch = TRANSACTION_LOG[i:i+batch_size]
        b_model_fr = (sum(1 for tx in batch if tx["is_fraud_pred"] == 1) / len(batch)) * 100
        b_labeled = [tx for tx in batch if tx["is_fraud_fact"] is not None]
        b_fact_fr = (sum(1 for tx in b_labeled if tx["is_fraud_fact"] == 1) / len(b_labeled) * 100) if b_labeled else 0.0
        
        time_series.append({
            "batch_index": i // batch_size + 1,
            "timestamp": batch[-1]["timestamp"],
            "fraud_rate_model": round(b_model_fr, 2),
            "fraud_rate_fact": round(b_fact_fr, 2),
            "success_rate": 100.0
        })

    # Последние срабатывания (где модель предсказала фрод)
    recent_frauds = [
        {
            "transaction_id": tx["transaction_id"],
            "timestamp": tx["timestamp"],
            "amt": tx["amt"],
            "category": tx["category"],
            "fraud_score": tx["fraud_score"],
            "reasons": ", ".join(tx["reasons"]),
            "is_fraud_fact": tx["is_fraud_fact"] if tx["is_fraud_fact"] is not None else "Ожидает метку"
        }
        for tx in reversed(TRANSACTION_LOG) if tx["is_fraud_pred"] == 1
    ][:15]

    return {
        "total_requests": total_requests,
        "success_requests": success_requests,
        "error_count": error_count,
        "success_rate": round(success_rate, 2),
        "fraud_rate_model": round(fraud_rate_model, 2),
        "fraud_rate_fact": round(fraud_rate_fact, 2),
        "labeled_count": len(labeled_txs),
        "current_threshold": CURRENT_THRESHOLD,
        "time_series": time_series,
        "recent_frauds": recent_frauds
    }
