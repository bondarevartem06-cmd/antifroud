"""
Шаги пакетного процесса «прескоринг -> финансовый отчёт». Каждая подкоманда соответствует задаче DAG
antifraud_prescoring_report в Airflow, но запускается и отдельно:

    python pipeline.py extract    --out work/holdout.pkl
    python pipeline.py score      --inp work/holdout.pkl --out work/scored.pkl [--threshold 0.9]
    python pipeline.py load       --inp work/scored.pkl --run-id manual
    python pipeline.py financials --workdir work --run-id manual
    python pipeline.py wait-llm
    python pipeline.py summary    --workdir work --run-id manual
    python pipeline.py validate   --workdir work
    python pipeline.py cleanup    --workdir work

Идемпотентность: результаты прескоринга пишутся UPSERT-ом по ключу (model_version, trans_num), отчёт по ключу
report_key = хеш(версия модели, порог, допущения, размер выборки, LLM). Повторный запуск не создаёт дубликатов.
"""
import argparse
import json
import os
import shutil
import sys

import pandas as pd
from sqlalchemy import text

import db
import finance
import report_llm
from assumptions import ASSUMPTION_RULES, ASSUMPTIONS

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")

Q_EXTRACT = """
SELECT trans_num, trans_time AS trans_date_trans_time, cc_num, merchant, category, amt, gender,
       CAST(dob AS text) AS dob, lat, long, merch_lat, merch_long, city_pop, state, is_fraud
FROM holdout_transactions
ORDER BY trans_time, trans_num
"""


def _state_path(workdir: str) -> str:
    return os.path.join(workdir, "report.json")


def _read_state(workdir: str) -> dict:
    with open(_state_path(workdir), encoding="utf-8") as f:
        return json.load(f)


def _assumptions_from_args(a) -> dict:
    out = dict(ASSUMPTIONS)
    for k in ("chargeback_fee", "review_cost", "fp_fixed_cost", "fp_revenue_rate"):
        v = getattr(a, k, None)
        if v is not None:
            out[k] = float(v)
    return out


# ----------------------------------------------------------------------------
def cmd_extract(a):
    """Извлечение отложенной выборки из PostgreSQL во временный файл."""
    with db.get_engine().connect() as c:
        n = c.execute(text("SELECT count(*) FROM holdout_transactions")).scalar_one()
    if n == 0:
        sys.exit("Таблица holdout_transactions пуста. Загрузите выборку: "
                 "docker compose --profile tools run --rm holdout-loader")
    df = pd.read_sql(text(Q_EXTRACT), db.get_engine())
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_pickle(a.out)
    print(f"[extract] извлечено {len(df):,} операций, мошеннических {int(df['is_fraud'].sum()):,} "
          f"({df['is_fraud'].mean():.3%}), период {df['trans_date_trans_time'].min()} .. "
          f"{df['trans_date_trans_time'].max()}")


def cmd_score(a):
    """Скоринг извлечённой выборки обученной моделью."""
    import batch_scoring  # тяжёлые импорты (lightgbm и др.) только здесь

    df = pd.read_pickle(a.inp)
    mv = batch_scoring.model_version(MODEL_PATH)
    art = batch_scoring.load_artifact(MODEL_PATH)
    threshold = float(a.threshold) if a.threshold not in (None, "") else float(art["threshold"])
    scored = batch_scoring.score_frame(df, art)
    scored["is_fraud_pred"] = scored["fraud_score"] >= threshold
    pd.to_pickle({"scored": scored, "model_version": mv, "threshold": threshold}, a.out)
    print(f"[score] модель {mv}, порог {threshold:.4f}: оценено {len(scored):,} операций, "
          f"заблокировано {int(scored['is_fraud_pred'].sum()):,}")


def cmd_load(a):
    """Запись результатов в prescoring_results (UPSERT, без дубликатов)."""
    obj = pd.read_pickle(a.inp)
    scored, mv, th = obj["scored"].drop_duplicates("trans_num"), obj["model_version"], obj["threshold"]
    with db.get_engine().begin() as conn:
        conn.execute(text(
            "CREATE TEMP TABLE _stage (model_version text, trans_num text, fraud_score double precision, "
            "threshold double precision, is_fraud_pred boolean, run_id text) ON COMMIT DROP"))
        with conn.connection.driver_connection.cursor() as cur:
            with cur.copy("COPY _stage (model_version, trans_num, fraud_score, threshold, is_fraud_pred, run_id) "
                          "FROM STDIN") as cp:
                for tn, sc, pred in zip(scored["trans_num"].tolist(), scored["fraud_score"].tolist(),
                                        scored["is_fraud_pred"].tolist()):
                    cp.write_row((mv, tn, sc, th, pred, a.run_id))
        conn.execute(text(
            "INSERT INTO prescoring_results (model_version, trans_num, fraud_score, threshold, is_fraud_pred, run_id) "
            "SELECT model_version, trans_num, fraud_score, threshold, is_fraud_pred, run_id FROM _stage "
            "ON CONFLICT (model_version, trans_num) DO UPDATE SET fraud_score = EXCLUDED.fraud_score, "
            "threshold = EXCLUDED.threshold, is_fraud_pred = EXCLUDED.is_fraud_pred, run_id = EXCLUDED.run_id, "
            "scored_at = now()"))
        stored = conn.execute(text("SELECT count(*) FROM prescoring_results WHERE model_version = :mv"),
                              {"mv": mv}).scalar_one()
        holdout = conn.execute(text("SELECT count(*) FROM holdout_transactions")).scalar_one()
    print(f"[load] версия модели {mv}: в prescoring_results {stored:,} строк (в выборке {holdout:,})")
    if stored != holdout:
        sys.exit("Число результатов не совпадает с размером выборки")


def cmd_financials(a):
    """Финансовые показатели SQL-запросами; создание отчёта (стадия metrics)."""
    import batch_scoring  # только model_version; без загрузки модели
    mv = batch_scoring.model_version(MODEL_PATH)
    params = _assumptions_from_args(a)
    metrics = finance.compute_metrics(mv, params)

    ident = finance.check_identities(metrics)
    if not all(ident.values()):
        sys.exit(f"Внутренние тождества показателей нарушены: {ident}")

    llm_model = report_llm.OLLAMA_MODEL
    key = finance.report_key(mv, metrics["threshold"], params, metrics["n_transactions"], metrics["n_fraud"], llm_model)
    existing = db.fetch_report(key)
    same = existing is not None and json.loads(json.dumps(metrics)) == existing["metrics"]
    if existing and existing["status"] == "verified" and same and not a.force:
        db.touch_report(key, a.run_id)
        print(f"[financials] отчёт {key} уже существует и проверен: повторно не создаётся")
    else:
        db.upsert_report_metrics(key, mv, metrics["threshold"], metrics["n_transactions"], llm_model,
                                 {"parameters": params, "rules": ASSUMPTION_RULES}, metrics, a.run_id)
        print(f"[financials] отчёт {key}: стадия metrics")
    os.makedirs(a.workdir, exist_ok=True)
    with open(_state_path(a.workdir), "w", encoding="utf-8") as f:
        json.dump({"report_key": key}, f)

    m = metrics
    print(f"[financials] предотвращённые потери {m['prevented_loss']:,.2f}; понесённые {m['incurred_loss']:,.2f}; "
          f"ложные срабатывания {m['fp_cost']:,.2f}; ручные проверки {m['review_cost']:,.2f}; "
          f"чистый эффект {m['net_effect']:,.2f} {m['currency']}")


def cmd_wait_llm(a):
    report_llm.wait_ready(timeout_s=int(a.timeout_min) * 60)


def cmd_summary(a):
    """Резюме от языковой модели с автоматической проверкой чисел (до 3 попыток, затем шаблон)."""
    key = _read_state(a.workdir)["report_key"]
    row = db.fetch_report(key)
    facts = finance.build_facts(row["metrics"], row["assumptions"]["parameters"])
    if row["status"] == "verified" and not a.force:
        if finance.verify_summary(row["summary"], facts, row["metrics"])["ok"]:
            print(f"[summary] отчёт {key} уже проверен: генерация пропущена")
            return
        print(f"[summary] отчёт {key} не проходит текущие правила проверки: резюме будет создано заново")
    print("[summary] факты для языковой модели:\n" + facts)
    res = report_llm.generate_verified_summary(row["metrics"], facts, model=row["llm_model"] or report_llm.OLLAMA_MODEL,
                                               max_attempts=int(a.attempts),
                                               allow_fallback=os.getenv("REPORT_ALLOW_TEMPLATE_FALLBACK", "1") == "1")
    db.save_summary(key, res["summary"], res["source"], res["llm_model"], res["attempts"])
    print(f"[summary] источник: {res['source']}\n{res['summary']}")


def cmd_validate(a):
    """Итоговая проверка отчёта; непроверенный отчёт не публикуется, задача завершается ошибкой."""
    key = _read_state(a.workdir)["report_key"]
    row = db.fetch_report(key)
    try:
        checks = finance.validate_report(row)
    except Exception as exc:  # отчёт не должен зависать в статусе drafted: фиксируем ошибку в базе
        db.finalize_report(key, False, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        raise
    db.finalize_report(key, checks["ok"], checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2, default=str))
    if not checks["ok"]:
        sys.exit(f"Отчёт {key} не прошёл проверку (status = rejected)")
    print(f"[validate] отчёт {key} проверен и опубликован (GET /reports/latest)")


def cmd_cleanup(a):
    shutil.rmtree(a.workdir, ignore_errors=True)
    print(f"[cleanup] удалено {a.workdir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract"); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("score"); p.add_argument("--inp", required=True); p.add_argument("--out", required=True)
    p.add_argument("--threshold", default=""); p.set_defaults(fn=cmd_score)
    p = sub.add_parser("load"); p.add_argument("--inp", required=True); p.add_argument("--run-id", default="manual")
    p.set_defaults(fn=cmd_load)

    p = sub.add_parser("financials"); p.add_argument("--workdir", required=True); p.add_argument("--run-id", default="manual")
    p.add_argument("--force", type=int, default=0)
    for k in ("chargeback-fee", "review-cost", "fp-fixed-cost", "fp-revenue-rate"):
        p.add_argument(f"--{k}", type=float, default=None)
    p.set_defaults(fn=cmd_financials)

    p = sub.add_parser("wait-llm"); p.add_argument("--timeout-min", default=45); p.set_defaults(fn=cmd_wait_llm)
    p = sub.add_parser("summary"); p.add_argument("--workdir", required=True); p.add_argument("--force", type=int, default=0)
    p.add_argument("--attempts", default=3); p.set_defaults(fn=cmd_summary)
    p = sub.add_parser("validate"); p.add_argument("--workdir", required=True); p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("cleanup"); p.add_argument("--workdir", required=True); p.set_defaults(fn=cmd_cleanup)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
