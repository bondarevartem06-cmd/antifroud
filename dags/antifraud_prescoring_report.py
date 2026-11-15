"""
Единый DAG: прескоринг отложенной выборки -> финансовые показатели -> резюме от языковой модели -> проверенный отчёт.

extract_holdout >> score_holdout >> load_results >> calc_financials >> wait_for_llm >> generate_summary >> validate_report
validate_report >> [cleanup, report_published]
(report_published: листовая задача со строгим правилом all_success; если любая задача выше упала, запуск получает статус failed.
cleanup выполняется всегда, но сам статус запуска не маскирует.)

Идемпотентность: результаты пишутся UPSERT-ом по (model_version, trans_num), отчёт хранится по уникальному
report_key (версия модели, порог, допущения, размер выборки, LLM). Повторный запуск не создаёт дубликатов.
Расчётная логика лежит в /app (pipeline.py); задачи запускают её системным Python образа, где установлены
библиотеки модели, а сам Airflow работает в отдельном виртуальном окружении.
"""
import sys
from datetime import datetime, timedelta

sys.path.insert(0, "/app")

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

from assumptions import ASSUMPTIONS

PY = "python /app/pipeline.py "
WORK = "/opt/airflow/work/{{ ts_nodash }}"
RUN_ID = "'{{ run_id }}'"
THRESHOLD = "'{{ params.threshold if params.threshold is not none else \"\" }}'"
FORCE = "{{ 1 if params.force_regenerate else 0 }}"
ASSUME = ("--chargeback-fee {{ params.chargeback_fee }} --review-cost {{ params.review_cost }} "
          "--fp-fixed-cost {{ params.fp_fixed_cost }} --fp-revenue-rate {{ params.fp_revenue_rate }}")

with DAG(
    dag_id="antifraud_prescoring_report",
    description="Прескоринг отложенной выборки и бизнес-отчёт для руководства",
    start_date=datetime(2026, 1, 1),
    schedule="0 3 * * *",
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    default_args={"owner": "antifraud", "retries": 1, "retry_delay": timedelta(minutes=2)},
    params={
        "threshold": Param(None, type=["null", "number"], minimum=0, maximum=1,
                           description="Порог блокировки; пусто = порог из обучения модели"),
        "chargeback_fee": Param(ASSUMPTIONS["chargeback_fee"], type="number", minimum=0,
                                description="Расход на каждый пропущенный случай мошенничества, USD"),
        "review_cost": Param(ASSUMPTIONS["review_cost"], type="number", minimum=0,
                             description="Стоимость ручной проверки одной блокировки, USD"),
        "fp_fixed_cost": Param(ASSUMPTIONS["fp_fixed_cost"], type="number", minimum=0,
                               description="Фиксированная стоимость ложной блокировки, USD"),
        "fp_revenue_rate": Param(ASSUMPTIONS["fp_revenue_rate"], type="number", minimum=0, maximum=1,
                                 description="Доля суммы, потерянная при ложной блокировке"),
        "force_regenerate": Param(False, type="boolean",
                                  description="Пересоздать отчёт и резюме, даже если проверенный уже есть"),
    },
    tags=["antifraud", "batch"],
    doc_md=__doc__,
) as dag:

    extract = BashOperator(
        task_id="extract_holdout",
        bash_command=PY + "extract --out " + WORK + "/holdout.pkl",
    )
    score = BashOperator(
        task_id="score_holdout",
        bash_command=PY + "score --inp " + WORK + "/holdout.pkl --out " + WORK + "/scored.pkl --threshold " + THRESHOLD,
        execution_timeout=timedelta(hours=2),
    )
    load = BashOperator(
        task_id="load_results",
        bash_command=PY + "load --inp " + WORK + "/scored.pkl --run-id " + RUN_ID,
    )
    financials = BashOperator(
        task_id="calc_financials",
        bash_command=PY + "financials --workdir " + WORK + " --run-id " + RUN_ID + " " + ASSUME + " --force " + FORCE,
    )
    wait_llm = BashOperator(
        task_id="wait_for_llm",
        bash_command=PY + "wait-llm --timeout-min 45",
        execution_timeout=timedelta(minutes=50),
        retries=0,
    )
    summary = BashOperator(
        task_id="generate_summary",
        bash_command=PY + "summary --workdir " + WORK + " --force " + FORCE,
        execution_timeout=timedelta(minutes=45),
    )
    validate = BashOperator(
        task_id="validate_report",
        bash_command=PY + "validate --workdir " + WORK,
        retries=0,
    )
    cleanup = BashOperator(
        task_id="cleanup",
        bash_command="rm -rf " + WORK,
        trigger_rule=TriggerRule.ALL_DONE,
        retries=0,
    )

    published = EmptyOperator(task_id="report_published")  # успех запуска = отчёт проверен и опубликован

    extract >> score >> load >> financials >> wait_llm >> summary >> validate >> [cleanup, published]
