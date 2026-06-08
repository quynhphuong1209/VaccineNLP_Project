import datetime
import os
from typing import Optional
from sqlalchemy import create_engine, String, Text, Float, DateTime, func, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB

db_url = os.environ.get("DATABASE_URL", "sqlite:///:memory:")
JSON_TYPE = JSONB if db_url.startswith("postgresql") else JSON

if db_url.startswith("sqlite"):
    from sqlalchemy.pool import StaticPool
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
else:
    engine = create_engine(db_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class AnalysisHistory(Base):
    __tablename__ = "analysis_history"
    id: Mapped[int] = mapped_column(primary_key=True)
    source_text: Mapped[str] = mapped_column(Text)
    source_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    text_hash: Mapped[str] = mapped_column(String(64), index=True)

    misinfo_label: Mapped[str] = mapped_column(String(50))
    misinfo_score: Mapped[float] = mapped_column(Float)
    stance_label: Mapped[str] = mapped_column(String(50))
    stance_score: Mapped[float] = mapped_column(Float)
    sentiment_label: Mapped[str] = mapped_column(String(50))
    sentiment_score: Mapped[float] = mapped_column(Float)

    # MỚI: phân phối softmax ĐẦY ĐỦ theo từng trục, dạng {"misinfo":{"Fake":..,"Real":..}, ...}
    # Nullable để hàng cũ (trước migration) không vỡ; bản public dùng DB mới nên tự có cột này.
    phobert_probs: Mapped[Optional[dict]] = mapped_column(JSON_TYPE, nullable=True)

    consistency_flag: Mapped[str] = mapped_column(String(50), default="plausible")
    xai_status: Mapped[str] = mapped_column(String(20), default="pending")
    xai_explanation: Mapped[Optional[dict]] = mapped_column(JSON_TYPE, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
