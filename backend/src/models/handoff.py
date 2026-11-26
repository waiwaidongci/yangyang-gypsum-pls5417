from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.config.database import Base


class Handoff(Base):
    __tablename__ = 'handoffs'

    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(String(140))
    base_run_id: Mapped[int] = mapped_column(ForeignKey('analysis_runs.id'), index=True)
    dataset_ids: Mapped[list] = mapped_column(JSON)
    previous_handoff_id: Mapped[int | None] = mapped_column(ForeignKey('handoffs.id'), nullable=True)
    superseded_by: Mapped[int | None] = mapped_column(ForeignKey('handoffs.id'), nullable=True)
    anchor_instrument: Mapped[str] = mapped_column(String(64))
    created_by: Mapped[int] = mapped_column(ForeignKey('accounts.id'))
    created_at: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), index=True)
    settings: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text)


class HandoffReview(Base):
    __tablename__ = 'handoff_reviews'

    handoff_id: Mapped[int] = mapped_column(ForeignKey('handoffs.id'), primary_key=True)
    reviewer_id: Mapped[int] = mapped_column(ForeignKey('accounts.id'))
    decision: Mapped[str] = mapped_column(String(16))
    note: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(32))
    # Immutable copy of the handoff's own verdict and basis at review time, so a
    # later revision never rewrites what was issued.
    basis: Mapped[dict] = mapped_column(JSON)
