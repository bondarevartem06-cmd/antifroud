"""
FastAPI Anti-Fraud Real-Time Scoring Service
Course: Feature Engineering - Project 3 "Real-time Anti-fraud"
Скоринг выполняет обученная модель (artifacts/model.joblib, см. train_final.py и serving.py).
Каждый запрос (в том числе ошибочный), каждое решение и каждая метка пишутся в PostgreSQL (см. db.py),
а /metrics считает показатели SQL-запросами к базе.
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

import db
from serving import Scorer

app = FastAPI(
    title="Real-Time Anti-Fraud Scoring API",
    description="API для скоринга транзакций в реальном времени, приема меток и сбора метрик мониторинга.",
    version="3.0.0"
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

# Рабочий порог хранится в памяти процесса; по умолчанию берётся из обучения (максимум F1 на валидации).
# Каждое решение сохраняет в БД порог, с которым оно принято.
DEFAULT_THRESHOLD = scorer.threshold
CURRENT_THRESHOLD = DEFAULT_THRESHOLD

# Эндпоинты, запросы к которым попадают в журнал (мониторинговые /health и /metrics не логируем)
LOGGED_PATHS = {"/predict", "/label", "/threshold"}

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
    is_fraud: int = Field(..., ge=0, le=1, description="1 = Мошенничество, 0 = Честная операция")

class ThresholdInput(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0, description="Новый порог срабатывания (от 0 до 1)")

# ----------------------------------------------------------------------------
# Журнал запросов (middleware) и обработчики ошибок
# ----------------------------------------------------------------------------
@app.middleware("http")
async def audit_log(request: Request, call_next):
    """Пишет в таблицу requests каждый запрос к бизнес-эндпоинтам, в том числе ошибочный."""
    if request.url.path not in LOGGED_PATHS:
        return await call_next(request)

    request_id = uuid4()
    request.state.request_id = request_id
    request.state.payload = None
    request.state.error_type = None
    request.state.error_detail = None
    client_ip = request.client.host if request.client else None
    started = time.perf_counter()

    try:
        await run_in_threadpool(db.request_started, request_id, request.method, request.url.path, client_ip)
    except Exception as exc:  # без журнала решения не принимаем: аудит обязателен
        print(f"[audit] БД недоступна: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return JSONResponse(status_code=503, content={"detail": "Database unavailable"})

    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:  # необработанная ошибка: ответ 500 сформирует global_exception_handler
        request.state.error_type = "500_internal_error"
        request.state.error_detail = str(exc)
        raise
    finally:
        error_type = request.state.error_type
        if status_code >= 400 and error_type is None:
            error_type = f"{status_code}_error"
        try:
            await run_in_threadpool(
                db.request_finished, request_id, status_code,
                (time.perf_counter() - started) * 1000, error_type,
                request.state.error_detail, request.state.payload,
            )
        except Exception as exc:
            print(f"[audit] не удалось записать итог запроса: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request.state.error_type = "422_validation_error"
    request.state.error_detail = str(exc)
    request.state.payload = exc.body  # то, что прислал клиент
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation Error", "errors": jsonable_encoder(exc.errors())}
    )

@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    request.state.error_type = f"{exc.status_code}_http_error"
    request.state.error_detail = str(exc.detail)
    return await http_exception_handler(request, exc)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
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
    """Статус сервиса и базы данных. При недоступной БД отвечает 503."""
    try:
        db.ping()
    except Exception as exc:
        return JSONResponse(status_code=503, content={
            "status": "degraded", "db": "error", "detail": type(exc).__name__})
    return {
        "status": "ok",
        "service": "antifraud_scoring",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": scorer.model is not None,
        "db": "ok",
        "current_threshold": CURRENT_THRESHOLD
    }

@app.post("/predict", response_model=PredictResponse, summary="Скоринг одиночной транзакции")
def predict_transaction(tx: TransactionInput, request: Request):
    """
    Принимает одну транзакцию, рассчитывает признаки, сохраняет решение в БД
    и возвращает решение и причины.
    """
    request.state.payload = tx.model_dump()
    tx_id = uuid4()
    score, reasons = calculate_score_and_reasons(tx)
    threshold = CURRENT_THRESHOLD
    is_fraud_pred = score >= threshold

    db.insert_decision({
        "transaction_id": tx_id,
        "request_id": request.state.request_id,
        "trans_time": tx.trans_date_trans_time,
        "cc_num": tx.cc_num,
        "amt": tx.amt,
        "category": tx.category,
        "fraud_score": score,
        "is_fraud_pred": is_fraud_pred,
        "threshold": threshold,
        "reasons": reasons,
    })

    return PredictResponse(
        transaction_id=str(tx_id),
        fraud_score=round(score, 4),
        is_fraud_pred=int(is_fraud_pred),
        threshold=threshold,
        reasons=reasons
    )

@app.post("/label", summary="Обновление фактической метки фрода")
def label_transaction(payload: LabelInput, request: Request):
    """
    Привязывает подтвержденный факт мошенничества (0 или 1) к ранее оцененной транзакции.
    """
    request.state.payload = payload.model_dump()
    not_found = HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Transaction ID {payload.transaction_id} not found."
    )
    try:
        tx_uuid = UUID(payload.transaction_id)
    except ValueError:
        raise not_found
    if not db.upsert_label(tx_uuid, bool(payload.is_fraud), request.state.request_id):
        raise not_found
    return {
        "status": "ok",
        "message": f"Label {payload.is_fraud} attached to transaction {payload.transaction_id}"
    }

@app.put("/threshold", summary="Динамическое изменение порога")
def update_threshold(payload: ThresholdInput, request: Request):
    """Изменяет рабочий порог принятия решений без перезапуска сервиса."""
    global CURRENT_THRESHOLD
    request.state.payload = payload.model_dump()
    old_threshold = CURRENT_THRESHOLD
    CURRENT_THRESHOLD = payload.threshold
    return {
        "status": "ok",
        "old_threshold": old_threshold,
        "new_threshold": CURRENT_THRESHOLD
    }

@app.get("/reports/latest", summary="Последний проверенный бизнес-отчёт")
def latest_report():
    """
    Бизнес-отчёт по результатам прескоринга: финансовые показатели, резюме для руководства, допущения и
    результаты автоматической проверки чисел. Формируется DAG-ом Airflow antifraud_prescoring_report;
    отдаётся только отчёт со статусом verified.
    """
    report = db.get_latest_report()
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Проверенных отчётов пока нет. Запустите DAG antifraud_prescoring_report в Airflow."
        )
    return report

@app.get("/metrics", summary="Метрики для панели мониторинга")
def get_metrics():
    """
    Сводная статистика для панели мониторинга. Все показатели считаются SQL-запросами к PostgreSQL.
    """
    return db.compute_metrics(CURRENT_THRESHOLD)
