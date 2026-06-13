from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    projects: Mapped[list["Project"]] = relationship("Project", back_populates="user")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    dataset_path: Mapped[Optional[str]] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    user: Mapped["User"] = relationship("User", back_populates="projects")
    feature_sets: Mapped[list["FeatureSet"]] = relationship(
        "FeatureSet", back_populates="project"
    )
    training_runs: Mapped[list["TrainingRun"]] = relationship(
        "TrainingRun", back_populates="project"
    )


class FeatureSet(Base):
    __tablename__ = "feature_sets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # {"features": [{"name": ..., "type": ..., "source": ..., ...}]}
    features_list: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    project: Mapped["Project"] = relationship("Project", back_populates="feature_sets")
    training_runs: Mapped[list["TrainingRun"]] = relationship(
        "TrainingRun", back_populates="feature_set"
    )
    history: Mapped[list["FeatureHistory"]] = relationship(
        "FeatureHistory", back_populates="feature_set"
    )


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    feature_set_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("feature_sets.id", ondelete="CASCADE"), nullable=False
    )
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)
    # {"accuracy": 0.95, "f1": 0.93, "roc_auc": 0.97, ...}
    metrics: Mapped[Optional[dict]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    project: Mapped["Project"] = relationship("Project", back_populates="training_runs")
    feature_set: Mapped["FeatureSet"] = relationship(
        "FeatureSet", back_populates="training_runs"
    )


class FeatureHistory(Base):
    __tablename__ = "feature_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    feature_set_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("feature_sets.id", ondelete="CASCADE"), nullable=False
    )
    feature_name: Mapped[str] = mapped_column(String(255), nullable=False)
    importance_score: Mapped[Optional[float]] = mapped_column(Float)

    feature_set: Mapped["FeatureSet"] = relationship(
        "FeatureSet", back_populates="history"
    )
