"""
FastAPI Anti-Fraud Real-Time Scoring Service
Course: Feature Engineering - Project 3 "Real-time Anti-fraud"
Скоринг выполняет обученная модель (artifacts/model.joblib, см. train_final.py и serving.py).
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from serving import Scorer

app = FastAPI(
    title="Real-Time Anti-Fraud Scoring API",
    description="API для скоринга транзакций в реальном времени, приема меток и сбора метрик мониторинга.",
    version="2.0.0"
)

# Enable CORS for browser frontend if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Обученная модель + история карт (загружается один раз при старте)
scorer = Scorer(os.getenv("MODEL_PATH", "artifacts/model.joblib"))

# Global State: порог по умолчанию берётся из обучения (максимум F1 на валидации)
DEFAULT_THRESHOLD = scorer.threshold
CURRENT_THRESHOLD = DEFAULT_THRESHOLD

# In-memory transaction and metrics logs
# In production this would be backed by PostgreSQL/SQLite
TRANSACTION_LOG: List[Dict[str, Any]] = []
ERROR_LOG: List[Dict[str, Any]] = []
# Единый журнал запросов /predict (успехи и ошибки по порядку): из него считается доступность
REQUEST_LOG: List[Dict[str, Any]] = []  # {"timestamp", "ok", "tx": запись из TRANSACTION_LOG или None}

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
    merchant: Optional[str] = Field("unknown", json_schema_extra={"example": "fraud_Rippin, Kub and Mann"})

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
def log_failure(request: Request, kind: str, detail: str):
    now = datetime.now().isoformat()
    ERROR_LOG.append({"timestamp": now, "type": kind, "detail": detail})
    if request.url.path == "/predict":  # неуспешный запрос скоринга
        REQUEST_LOG.append({"timestamp": now, "ok": False, "tx": None})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    log_failure(request, "422_validation_error", str(exc))
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation Error", "errors": exc.errors()}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log_failure(request, "500_internal_error", str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )

# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def calculate_score_and_reasons(tx: TransactionInput) -> tuple[float, List[str]]:
    """
    Вероятность фрода от обученной модели и интерпретируемые причины
    (три признака с наибольшим положительным вкладом, SHAP-значения LightGBM).
    История карты обновляется внутри Scorer.
    """
    return scorer.score(tx.model_dump())

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
        "model_loaded": scorer.model is not None,
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
    REQUEST_LOG.append({"timestamp": log_entry["timestamp"], "ok": True, "tx": log_entry})

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
    total_requests = len(REQUEST_LOG)  # все запросы /predict
    success_requests = sum(1 for r in REQUEST_LOG if r["ok"])
    error_count = total_requests - success_requests

    success_rate = (success_requests / total_requests * 100) if total_requests > 0 else 100.0

    # Fraud rate по модели
    model_positives = sum(1 for tx in TRANSACTION_LOG if tx["is_fraud_pred"] == 1)
    fraud_rate_model = (model_positives / success_requests * 100) if success_requests > 0 else 0.0

    # Fraud rate фактический (по размеченным)
    labeled_txs = [tx for tx in TRANSACTION_LOG if tx["is_fraud_fact"] is not None]
    fact_positives = sum(1 for tx in labeled_txs if tx["is_fraud_fact"] == 1)
    fraud_rate_fact = (fact_positives / len(labeled_txs) * 100) if len(labeled_txs) > 0 else 0.0

    # Качество решений модели по размеченным операциям (матрица ошибок)
    def outcome(tx):
        pred, fact = tx["is_fraud_pred"], tx["is_fraud_fact"]
        return "tp" if (pred and fact) else "fp" if pred else "fn" if fact else "tn"

    def quality(txs):
        c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for tx in txs:
            c[outcome(tx)] += 1
        n = len(txs)
        acc = (c["tp"] + c["tn"]) / n * 100 if n else None
        prec = c["tp"] / (c["tp"] + c["fp"]) * 100 if (c["tp"] + c["fp"]) else None
        rec = c["tp"] / (c["tp"] + c["fn"]) * 100 if (c["tp"] + c["fn"]) else None
        return c, acc, prec, rec

    confusion, accuracy, precision, recall = quality(labeled_txs)

    # Формируем временной ряд (батчи по 10 транзакций)
    time_series = []
    batch_size = 10
    for i in range(0, len(REQUEST_LOG), batch_size):
        batch = REQUEST_LOG[i:i+batch_size]
        ok_txs = [r["tx"] for r in batch if r["ok"]]  # записи успешных запросов пачки
        b_model_fr = (sum(1 for tx in ok_txs if tx["is_fraud_pred"] == 1) / len(ok_txs) * 100) if ok_txs else 0.0
        b_labeled = [tx for tx in ok_txs if tx["is_fraud_fact"] is not None]
        b_fact_fr = (sum(1 for tx in b_labeled if tx["is_fraud_fact"] == 1) / len(b_labeled) * 100) if b_labeled else 0.0
        
        b_conf, b_acc, _, _ = quality(b_labeled)
        time_series.append({
            "batch_index": i // batch_size + 1,
            "timestamp": batch[-1]["timestamp"],
            "fraud_rate_model": round(b_model_fr, 2),
            "fraud_rate_fact": round(b_fact_fr, 2),
            "success_rate": round(len(ok_txs) / len(batch) * 100, 2),  # доступность: доля успешных запросов в пачке
            "decision_accuracy": round(b_acc, 2) if b_acc is not None else None,  # доля верных решений модели
            "false_positives": b_conf["fp"],   # честные операции, заблокированные моделью
            "false_negatives": b_conf["fn"]    # пропущенный фрод
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

    # Последние ошибки модели: решение не совпало с фактической меткой
    recent_errors = [
        {
            "transaction_id": tx["transaction_id"],
            "timestamp": tx["timestamp"],
            "amt": tx["amt"],
            "category": tx["category"],
            "fraud_score": tx["fraud_score"],
            "error_type": "Ложная блокировка" if tx["is_fraud_pred"] == 1 else "Пропущенный фрод",
            "reasons": ", ".join(tx["reasons"]),
        }
        for tx in reversed(labeled_txs) if tx["is_fraud_pred"] != tx["is_fraud_fact"]
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
        "recent_frauds": recent_frauds,
        "confusion": confusion,
        "accuracy": round(accuracy, 2) if accuracy is not None else None,
        "precision": round(precision, 2) if precision is not None else None,
        "recall": round(recall, 2) if recall is not None else None,
        "recent_errors": recent_errors
    }
