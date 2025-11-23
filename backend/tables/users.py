from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from config import Base
import datetime
import enum


# ---------- ENUM for Gender ----------
class GenderEnum(str, enum.Enum):
    male = "Male"
    female = "Female"
    other = "Other"


# ---------- CareTaker Table ----------
class CareTaker(Base):
    __tablename__ = "caretakers"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    phone_number = Column(String(10), unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationship: one caretaker → many recipients
    care_recipients = relationship("CareRecipient", back_populates="caretaker", cascade="all, delete-orphan")


# ---------- CareRecipient Table ----------
class CareRecipient(Base):
    __tablename__ = "care_recipients"

    id = Column(Integer, primary_key=True, index=True)
    caretaker_id = Column(Integer, ForeignKey("caretakers.id", ondelete="CASCADE"), nullable=False)

    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    phone_number = Column(String(10), unique=True, index=True, nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(Enum(GenderEnum), nullable=True)
    respiratory_condition_status = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationship back to caretaker
    caretaker = relationship("CareTaker", back_populates="care_recipients")

    # medical reports relation
    medical_reports = relationship('MedicalReport', back_populates='care_recipient', cascade='all, delete-orphan')
