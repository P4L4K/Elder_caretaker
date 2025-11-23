from tables.medical_reports import MedicalReport


def create_medical_report(db, care_recipient_id: int, filename: str, mime_type: str, data: bytes):
    report = MedicalReport(
        care_recipient_id=care_recipient_id,
        filename=filename,
        mime_type=mime_type,
        data=data,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def list_reports_for_recipient(db, care_recipient_id: int):
    return db.query(MedicalReport).filter(MedicalReport.care_recipient_id == care_recipient_id).all()
