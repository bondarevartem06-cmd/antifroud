"""Начальная схема: requests, decisions, labels

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

NOW = sa.text("now()")


def upgrade() -> None:
    # 1. Каждый запрос к сервису (успешный и ошибочный)
    op.create_table(
        "requests",
        sa.Column("request_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.Column("method", sa.Text(), nullable=False),
        sa.Column("endpoint", sa.Text(), nullable=False),
        sa.Column("status_code", sa.Integer()),
        sa.Column("ok", sa.Boolean(), comment="True, если status_code < 400"),
        sa.Column("latency_ms", postgresql.DOUBLE_PRECISION()),
        sa.Column("error_type", sa.Text()),
        sa.Column("error_detail", sa.Text()),
        sa.Column("payload", postgresql.JSONB(), comment="Тело запроса"),
        sa.Column("client_ip", sa.Text()),
        comment="Журнал запросов к API, включая ошибочные",
    )
    op.create_index("ix_requests_endpoint_created", "requests", ["endpoint", "created_at"])
    op.create_index("ix_requests_ok", "requests", ["ok"])

    # 2. Каждое решение модели
    op.create_table(
        "decisions",
        sa.Column("transaction_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("requests.request_id"),
                  nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.Column("trans_time", sa.DateTime(), comment="Время операции из запроса"),
        sa.Column("cc_num", sa.Text(), nullable=False),
        sa.Column("amt", sa.Numeric(12, 2), nullable=False),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("fraud_score", postgresql.DOUBLE_PRECISION(), nullable=False),
        sa.Column("is_fraud_pred", sa.Boolean(), nullable=False),
        sa.Column("threshold", postgresql.DOUBLE_PRECISION(), nullable=False, comment="Порог на момент решения"),
        sa.Column("reasons", postgresql.JSONB(), nullable=False),
        sa.CheckConstraint("fraud_score >= 0 AND fraud_score <= 1", name="ck_decisions_score_range"),
        comment="Решения модели по транзакциям",
    )
    op.create_index("ix_decisions_created_at", "decisions", ["created_at"])
    op.create_index("ix_decisions_cc_num", "decisions", ["cc_num"])
    op.create_index("ix_decisions_is_fraud_pred", "decisions", ["is_fraud_pred"])

    # 3. Фактические метки
    op.create_table(
        "labels",
        sa.Column("transaction_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("decisions.transaction_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("requests.request_id")),
        sa.Column("is_fraud", sa.Boolean(), nullable=False),
        sa.Column("labeled_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        comment="Подтверждённые метки фрода (одна на транзакцию)",
    )


def downgrade() -> None:
    op.drop_table("labels")
    op.drop_index("ix_decisions_is_fraud_pred", table_name="decisions")
    op.drop_index("ix_decisions_cc_num", table_name="decisions")
    op.drop_index("ix_decisions_created_at", table_name="decisions")
    op.drop_table("decisions")
    op.drop_index("ix_requests_ok", table_name="requests")
    op.drop_index("ix_requests_endpoint_created", table_name="requests")
    op.drop_table("requests")
