from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from config import Base
import datetime


class MedicalReport(Base):
    __tablename__ = 'medical_reports'

    id = Column(Integer, primary_key=True, index=True)
    care_recipient_id = Column(Integer, ForeignKey('care_recipients.id', ondelete='CASCADE'), nullable=False)
    filename = Column(String, nullable=False)
    mime_type = Column(String, nullable=True)
    data = Column(LargeBinary, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)

    # relationship back to recipient
    care_recipient = relationship('CareRecipient', back_populates='medical_reports')
