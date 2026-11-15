"""Отложенная выборка, результаты прескоринга и бизнес-отчёты

Revision ID: 0002
Revises: 0001
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

NOW = sa.text("now()")


def upgrade() -> None:
    # Отложенная выборка (загружается из fraudTest.csv скриптом load_holdout.py)
    op.create_table(
        "holdout_transactions",
        sa.Column("trans_num", sa.Text(), primary_key=True),
        sa.Column("trans_time", sa.DateTime(), nullable=False),
        sa.Column("cc_num", sa.Text(), nullable=False),
        sa.Column("merchant", sa.Text(), nullable=False),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("amt", sa.Numeric(12, 2), nullable=False),
        sa.Column("gender", sa.Text()),
        sa.Column("dob", sa.Date()),
        sa.Column("lat", postgresql.DOUBLE_PRECISION()),
        sa.Column("long", postgresql.DOUBLE_PRECISION()),
        sa.Column("merch_lat", postgresql.DOUBLE_PRECISION()),
        sa.Column("merch_long", postgresql.DOUBLE_PRECISION()),
        sa.Column("city_pop", sa.Integer()),
        sa.Column("state", sa.Text()),
        sa.Column("is_fraud", sa.Boolean(), nullable=False, comment="Фактическая метка"),
        sa.Column("loaded_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        comment="Отложенная выборка для пакетного прескоринга",
    )
    op.create_index("ix_holdout_trans_time", "holdout_transactions", ["trans_time"])

    # Результаты прескоринга. Ключ (версия модели, транзакция): повторный запуск перезаписывает, а не дублирует
    op.create_table(
        "prescoring_results",
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("trans_num", sa.Text(), sa.ForeignKey("holdout_transactions.trans_num", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("fraud_score", postgresql.DOUBLE_PRECISION(), nullable=False),
        sa.Column("threshold", postgresql.DOUBLE_PRECISION(), nullable=False),
        sa.Column("is_fraud_pred", sa.Boolean(), nullable=False),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.Column("run_id", sa.Text(), comment="Запуск DAG, записавший результат"),
        sa.PrimaryKeyConstraint("model_version", "trans_num", name="pk_prescoring_results"),
        comment="Результаты пакетного скоринга отложенной выборки",
    )
    op.create_index("ix_prescoring_trans_num", "prescoring_results", ["trans_num"])

    # Бизнес-отчёты. report_key уникален: один отчёт на набор входных данных, допущений и LLM
    op.create_table(
        "business_reports",
        sa.Column("report_key", sa.Text(), primary_key=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.Column("verified_at", sa.DateTime(timezone=True)),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("threshold", postgresql.DOUBLE_PRECISION(), nullable=False),
        sa.Column("holdout_rows", sa.Integer(), nullable=False),
        sa.Column("llm_model", sa.Text()),
        sa.Column("summary_source", sa.Text(), comment="llm или template_fallback"),
        sa.Column("summary", sa.Text()),
        sa.Column("assumptions", postgresql.JSONB(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(), nullable=False),
        sa.Column("validation", postgresql.JSONB()),
        sa.Column("dag_run_id", sa.Text()),
        sa.CheckConstraint("status IN ('metrics', 'drafted', 'verified', 'rejected')", name="ck_reports_status"),
        comment="Бизнес-отчёты по результатам прескоринга",
    )
    op.create_index("ix_reports_status_verified", "business_reports", ["status", "verified_at"])


def downgrade() -> None:
    op.drop_index("ix_reports_status_verified", table_name="business_reports")
    op.drop_table("business_reports")
    op.drop_index("ix_prescoring_trans_num", table_name="prescoring_results")
    op.drop_table("prescoring_results")
    op.drop_index("ix_holdout_trans_time", table_name="holdout_transactions")
    op.drop_table("holdout_transactions")
