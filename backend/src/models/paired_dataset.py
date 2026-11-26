from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from src.config.database import Base


class PairedDataset(Base):
    """A dual-instrument measurement ledger with one row per strip and instrument."""

    __tablename__ = 'paired_datasets'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(140), unique=True)
    checksum: Mapped[str] = mapped_column(String(64))
    rows: Mapped[list] = mapped_column(JSON)
    created_by: Mapped[int] = mapped_column(ForeignKey('accounts.id'))
