"""
Работа с PostgreSQL: запись запросов, решений и меток, расчёт метрик SQL-запросами.
Подключение берётся из переменных окружения (пароль приходит из .env через docker-compose):
POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST (по умолчанию db), POSTGRES_PORT.
Либо целиком из DATABASE_URL.
Схема создаётся не здесь, а миграциями Alembic (папка alembic/).
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url


def database_url() -> URL:
    url = os.getenv("DATABASE_URL")
    if url:
        return make_url(url)
    # URL.create сам экранирует спецсимволы в пароле
    return URL.create(
        "postgresql+psycopg",
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.getenv("POSTGRES_HOST", "db"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.environ["POSTGRES_DB"],
    )


_engine = None


def get_engine():
    """Движок создаётся при первом обращении, поэтому импорт модуля не требует работающей БД."""
    global _engine
    if _engine is None:
        _engine = create_engine(database_url(), pool_size=10, max_overflow=20, pool_pre_ping=True)
    return _engine


def _json(value: Any) -> Optional[str]:
    return None if value is None else json.dumps(value, ensure_ascii=False, default=str)


def _parse_dt(value: Any) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def ping() -> None:
    with get_engine().connect() as c:
        c.execute(text("SELECT 1"))


# ----------------------------------------------------------------------------
# Запись
# ----------------------------------------------------------------------------
def request_started(request_id: UUID, method: str, endpoint: str, client_ip: Optional[str]) -> None:
    """Строка запроса создаётся до обработки, чтобы на неё могли ссылаться решения и метки."""
    with get_engine().begin() as c:
        c.execute(
            text("INSERT INTO requests (request_id, method, endpoint, client_ip) "
                 "VALUES (:id, :method, :endpoint, :ip)"),
            {"id": request_id, "method": method, "endpoint": endpoint, "ip": client_ip},
        )


def request_finished(request_id: UUID, status_code: int, latency_ms: float,
                     error_type: Optional[str], error_detail: Optional[str], payload: Any) -> None:
    with get_engine().begin() as c:
        c.execute(
            text("UPDATE requests SET status_code = :status, ok = :ok, latency_ms = :latency, "
                 "error_type = :etype, error_detail = :edetail, payload = CAST(:payload AS jsonb) "
                 "WHERE request_id = :id"),
            {"id": request_id, "status": status_code, "ok": status_code < 400,
             "latency": latency_ms, "etype": error_type, "edetail": error_detail,
             "payload": _json(payload)},
        )


def insert_decision(d: Dict[str, Any]) -> None:
    with get_engine().begin() as c:
        c.execute(
            text("INSERT INTO decisions (transaction_id, request_id, trans_time, cc_num, amt, category, "
                 "fraud_score, is_fraud_pred, threshold, reasons) "
                 "VALUES (:transaction_id, :request_id, :trans_time, :cc_num, :amt, :category, "
                 ":fraud_score, :is_fraud_pred, :threshold, CAST(:reasons AS jsonb))"),
            {**d, "trans_time": _parse_dt(d.get("trans_time")), "reasons": _json(d["reasons"])},
        )


def upsert_label(transaction_id: UUID, is_fraud: bool, request_id: UUID) -> bool:
    """Привязывает фактическую метку к решению. False, если такого решения нет."""
    with get_engine().begin() as c:
        exists = c.execute(text("SELECT 1 FROM decisions WHERE transaction_id = :t"),
                           {"t": transaction_id}).first()
        if exists is None:
            return False
        c.execute(
            text("INSERT INTO labels (transaction_id, request_id, is_fraud) VALUES (:t, :r, :f) "
                 "ON CONFLICT (transaction_id) DO UPDATE SET is_fraud = EXCLUDED.is_fraud, "
                 "request_id = EXCLUDED.request_id, labeled_at = now()"),
            {"t": transaction_id, "r": request_id, "f": is_fraud},
        )
    return True


# ----------------------------------------------------------------------------
# Метрики: всё считается SQL-запросами
# ----------------------------------------------------------------------------
Q_REQUESTS = """
SELECT count(*) AS total,
       count(*) FILTER (WHERE ok) AS ok_n
FROM requests
WHERE endpoint = '/predict' AND status_code IS NOT NULL
"""

Q_DECISIONS = """
SELECT count(*) AS n,
       count(*) FILTER (WHERE d.is_fraud_pred) AS model_pos,
       count(l.transaction_id) AS labeled,
       count(*) FILTER (WHERE l.is_fraud) AS fact_pos,
       count(*) FILTER (WHERE d.is_fraud_pred AND l.is_fraud) AS tp,
       count(*) FILTER (WHERE d.is_fraud_pred AND NOT l.is_fraud) AS fp,
       count(*) FILTER (WHERE NOT d.is_fraud_pred AND l.is_fraud) AS fn,
       count(*) FILTER (WHERE NOT d.is_fraud_pred AND NOT l.is_fraud) AS tn
FROM decisions d
LEFT JOIN labels l ON l.transaction_id = d.transaction_id
"""

# Пачки по :bs последовательных запросов /predict (включая ошибочные); возвращаются последние :maxb пачек
Q_TIMESERIES = """
WITH r AS (
    SELECT q.request_id, q.created_at, q.ok,
           (row_number() OVER (ORDER BY q.created_at, q.request_id) - 1) / CAST(:bs AS integer) + 1 AS batch_index
    FROM requests q
    WHERE q.endpoint = '/predict' AND q.status_code IS NOT NULL
),
b AS (
    SELECT r.batch_index,
           max(r.created_at) AS ts,
           count(*) AS n,
           count(*) FILTER (WHERE r.ok) AS ok_n,
           count(d.transaction_id) AS dec_n,
           count(*) FILTER (WHERE d.is_fraud_pred) AS model_pos,
           count(l.transaction_id) AS lab_n,
           count(*) FILTER (WHERE l.is_fraud) AS fact_pos,
           count(*) FILTER (WHERE d.is_fraud_pred = l.is_fraud) AS correct,
           count(*) FILTER (WHERE d.is_fraud_pred AND NOT l.is_fraud) AS fp,
           count(*) FILTER (WHERE NOT d.is_fraud_pred AND l.is_fraud) AS fn
    FROM r
    LEFT JOIN decisions d ON d.request_id = r.request_id
    LEFT JOIN labels l ON l.transaction_id = d.transaction_id
    GROUP BY r.batch_index
)
SELECT * FROM b
WHERE batch_index > (SELECT COALESCE(max(batch_index), 0) FROM b) - CAST(:maxb AS integer)
ORDER BY batch_index
"""

Q_RECENT_FRAUDS = """
SELECT d.transaction_id, d.created_at, d.amt, d.category, d.fraud_score, d.reasons, l.is_fraud
FROM decisions d
LEFT JOIN labels l ON l.transaction_id = d.transaction_id
WHERE d.is_fraud_pred
ORDER BY d.created_at DESC
LIMIT 15
"""

Q_RECENT_ERRORS = """
SELECT d.transaction_id, d.created_at, d.amt, d.category, d.fraud_score, d.reasons, d.is_fraud_pred
FROM decisions d
JOIN labels l ON l.transaction_id = d.transaction_id
WHERE d.is_fraud_pred <> l.is_fraud
ORDER BY d.created_at DESC
LIMIT 15
"""


def _pct(a: int, b: int) -> Optional[float]:
    return a / b * 100 if b else None


def _round(v: Optional[float]) -> Optional[float]:
    return round(v, 2) if v is not None else None


def compute_metrics(current_threshold: float, batch_size: int = 10, max_batches: int = 200) -> Dict[str, Any]:
    with get_engine().connect() as c:
        req = c.execute(text(Q_REQUESTS)).mappings().one()
        dec = c.execute(text(Q_DECISIONS)).mappings().one()
        ts_rows = c.execute(text(Q_TIMESERIES), {"bs": batch_size, "maxb": max_batches}).mappings().all()
        frauds = c.execute(text(Q_RECENT_FRAUDS)).mappings().all()
        errors = c.execute(text(Q_RECENT_ERRORS)).mappings().all()

    total, ok_n = req["total"], req["ok_n"]
    tp, fp, fn, tn = dec["tp"], dec["fp"], dec["fn"], dec["tn"]
    labeled = dec["labeled"]

    time_series = [
        {
            "batch_index": int(r["batch_index"]),
            "timestamp": r["ts"].isoformat(),
            "fraud_rate_model": _round(_pct(r["model_pos"], r["dec_n"]) or 0.0),
            "fraud_rate_fact": _round(_pct(r["fact_pos"], r["lab_n"]) or 0.0),
            "success_rate": _round(_pct(r["ok_n"], r["n"])),
            "decision_accuracy": _round(_pct(r["correct"], r["lab_n"])),
            "false_positives": r["fp"],
            "false_negatives": r["fn"],
        }
        for r in ts_rows
    ]

    def fact_label(v):
        return "Ожидает метку" if v is None else int(v)

    recent_frauds = [
        {
            "transaction_id": str(r["transaction_id"]),
            "timestamp": r["created_at"].isoformat(),
            "amt": float(r["amt"]),
            "category": r["category"],
            "fraud_score": round(r["fraud_score"], 4),
            "reasons": ", ".join(r["reasons"]),
            "is_fraud_fact": fact_label(r["is_fraud"]),
        }
        for r in frauds
    ]
    recent_errors = [
        {
            "transaction_id": str(r["transaction_id"]),
            "timestamp": r["created_at"].isoformat(),
            "amt": float(r["amt"]),
            "category": r["category"],
            "fraud_score": round(r["fraud_score"], 4),
            "error_type": "Ложная блокировка" if r["is_fraud_pred"] else "Пропущенный фрод",
            "reasons": ", ".join(r["reasons"]),
        }
        for r in errors
    ]

    return {
        "total_requests": total,
        "success_requests": ok_n,
        "error_count": total - ok_n,
        "success_rate": _round(_pct(ok_n, total)) if total else 100.0,
        "fraud_rate_model": _round(_pct(dec["model_pos"], dec["n"]) or 0.0),
        "fraud_rate_fact": _round(_pct(dec["fact_pos"], labeled) or 0.0),
        "labeled_count": labeled,
        "current_threshold": current_threshold,
        "time_series": time_series,
        "recent_frauds": recent_frauds,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "accuracy": _round(_pct(tp + tn, labeled)),
        "precision": _round(_pct(tp, tp + fp)),
        "recall": _round(_pct(tp, tp + fn)),
        "recent_errors": recent_errors,
    }


# ----------------------------------------------------------------------------
# Бизнес-отчёты (заполняются DAG-ом Airflow, читаются через GET /reports/latest)
# ----------------------------------------------------------------------------
_REPORT_SELECT = ("SELECT report_key, status, created_at, updated_at, verified_at, model_version, threshold, "
                  "holdout_rows, llm_model, summary_source, summary, assumptions, metrics, validation, dag_run_id "
                  "FROM business_reports")


def fetch_report(report_key: str) -> Optional[Dict[str, Any]]:
    with get_engine().connect() as c:
        row = c.execute(text(_REPORT_SELECT + " WHERE report_key = :k"), {"k": report_key}).mappings().first()
    return dict(row) if row else None


def upsert_report_metrics(report_key: str, model_version: str, threshold: float, holdout_rows: int,
                          llm_model: str, assumptions: Any, metrics: Any, run_id: str) -> None:
    """Создаёт отчёт или сбрасывает существующий (тот же ключ) до стадии metrics. Дубликатов не бывает: report_key уникален."""
    with get_engine().begin() as c:
        c.execute(
            text("INSERT INTO business_reports (report_key, status, model_version, threshold, holdout_rows, "
                 "llm_model, assumptions, metrics, dag_run_id) "
                 "VALUES (:k, 'metrics', :mv, :th, :n, :llm, CAST(:a AS jsonb), CAST(:m AS jsonb), :run) "
                 "ON CONFLICT (report_key) DO UPDATE SET status = 'metrics', summary = NULL, summary_source = NULL, "
                 "validation = NULL, verified_at = NULL, assumptions = EXCLUDED.assumptions, "
                 "metrics = EXCLUDED.metrics, llm_model = EXCLUDED.llm_model, dag_run_id = EXCLUDED.dag_run_id, "
                 "updated_at = now()"),
            {"k": report_key, "mv": model_version, "th": threshold, "n": holdout_rows, "llm": llm_model,
             "a": _json(assumptions), "m": _json(metrics), "run": run_id},
        )


def touch_report(report_key: str, run_id: str) -> None:
    with get_engine().begin() as c:
        c.execute(text("UPDATE business_reports SET dag_run_id = :run, updated_at = now() WHERE report_key = :k"),
                  {"k": report_key, "run": run_id})


def save_summary(report_key: str, summary: str, source: str, llm_model: str, attempts: Any) -> None:
    with get_engine().begin() as c:
        c.execute(
            text("UPDATE business_reports SET status = 'drafted', summary = :s, summary_source = :src, "
                 "llm_model = :llm, validation = CAST(:v AS jsonb), updated_at = now() WHERE report_key = :k"),
            {"k": report_key, "s": summary, "src": source, "llm": llm_model, "v": _json({"llm_attempts": attempts})},
        )


def finalize_report(report_key: str, ok: bool, checks: Any) -> None:
    with get_engine().begin() as c:
        c.execute(
            text("UPDATE business_reports SET status = :st, "
                 "validation = COALESCE(validation, CAST('{}' AS jsonb)) || CAST(:v AS jsonb), "
                 "verified_at = CASE WHEN :ok THEN now() ELSE NULL END, updated_at = now() WHERE report_key = :k"),
            {"k": report_key, "st": "verified" if ok else "rejected", "ok": ok, "v": _json({"checks": checks})},
        )


def get_latest_report() -> Optional[Dict[str, Any]]:
    """Последний отчёт, прошедший проверку чисел (status = verified)."""
    with get_engine().connect() as c:
        row = c.execute(text(_REPORT_SELECT + " WHERE status = 'verified' ORDER BY verified_at DESC LIMIT 1")
                        ).mappings().first()
    if row is None:
        return None
    r = dict(row)
    for k in ("created_at", "updated_at", "verified_at"):
        r[k] = r[k].isoformat() if r[k] else None
    return r
