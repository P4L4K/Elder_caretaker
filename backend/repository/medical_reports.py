from tables.medical_reports import MedicalReport
import config


def create_medical_report(db, care_recipient_id: int, filename: str, mime_type: str, data: bytes):
    report = MedicalReport(
        care_recipient_id=care_recipient_id,
        filename=filename,
        mime_type=mime_type,
        data=data,
    )
    try:
        print(f"[repo.medical_reports] DB URL: {getattr(config, 'DATABASE_URL', 'unknown')}")
        db.add(report)
        db.commit()
        db.refresh(report)
        # confirm count for this recipient
        try:
            cnt = db.query(MedicalReport).filter(MedicalReport.care_recipient_id == care_recipient_id).count()
            print(f"[repo.medical_reports] After insert, count for recipient {care_recipient_id} = {cnt}")
        except Exception as qex:
            print("[repo.medical_reports] Could not query count after insert:", qex)
        return report
    except Exception as e:
        print("[repo.medical_reports] Exception while creating report:", e)
        try:
            db.rollback()
        except Exception:
            pass
        raise


def list_reports_for_recipient(db, care_recipient_id: int):
    return db.query(MedicalReport).filter(MedicalReport.care_recipient_id == care_recipient_id).all()
