from fastapi import APIRouter, Depends, File, UploadFile, Header, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import Optional

from models.users import ResponseSchema
from tables.users import CareRecipient, CareTaker
from config import get_db
from repository.users import UsersRepo
from repository.medical_reports import create_medical_report, list_reports_for_recipient
from utils.summarizer import extract_text_from_bytes, summarize_text_via_gemini
from fastapi.responses import StreamingResponse
from io import BytesIO

router = APIRouter(tags=["Recipients"])


def _get_username_from_auth(auth_header: Optional[str]):
    if not auth_header:
        return None
    try:
        parts = auth_header.split()
        if len(parts) != 2:
            return None
        token = parts[1]
        from repository.users import JWTRepo
        decoded = JWTRepo.decode_token(token)
        return decoded.get('sub') if isinstance(decoded, dict) else None
    except Exception:
        return None


@router.post('/recipients/{recipient_id}/reports', response_model=ResponseSchema)
async def upload_medical_report(recipient_id: int, file: UploadFile = File(...), authorization: Optional[str] = Header(None), db: Session = Depends(get_db), request: Request = None):
    # Auth
    # Debug: log some request header info
    try:
        hdrs = dict(request.headers)
        print(f"[recipients] Request headers: Authorization={'present' if hdrs.get('authorization') else 'missing'}, Content-Type={hdrs.get('content-type')}")
    except Exception:
        pass

    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Ensure recipient belongs to caretaker
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    # Basic validation
    content = await file.read()
    max_size = 5 * 1024 * 1024  # 5 MB
    if len(content) > max_size:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")

    # include application/octet-stream to handle some browsers that don't set a specific mime
    allowed = ['application/pdf', 'image/png', 'image/jpeg', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/octet-stream']
    mime = file.content_type or 'application/octet-stream'
    if mime not in allowed:
        # allow some generic types but prefer strict list
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported file type: {mime}")

    # Save to DB
    try:
        # Debug: log mime and size
        try:
            print(f"[recipients] Incoming upload: recipient_id={recipient_id} filename={file.filename} mime={mime} size={len(content)} bytes")
        except Exception:
            print(f"[recipients] Incoming upload: recipient_id={recipient_id} filename={file.filename} (could not determine size)")

        report = create_medical_report(db, recipient_id, file.filename, mime, content)
        print(f"[recipients] Uploaded report id={report.id} recipient_id={recipient_id} filename={file.filename}")

        # After saving, aggregate all report texts for this recipient and generate a short summary
        try:
            reports = list_reports_for_recipient(db, recipient_id)
            texts = []
            for r in reports:
                if r.data:
                    t = extract_text_from_bytes(r.data, r.mime_type)
                    if t:
                        texts.append(t)
            combined = '\n\n'.join(texts)
            # Limit size before sending to external API
            combined = combined[:200000]
            summary = summarize_text_via_gemini(combined)
            # update recipient
            recipient.report_summary = summary
            db.add(recipient)
            db.commit()
        except Exception as se:
            # Do not fail the upload if summarization fails
            print('Summarization failed:', se)

        return ResponseSchema(code=200, status='success', message='Report uploaded', result={'report_id': report.id, 'filename': report.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save report: {str(e)}")


@router.get('/recipients/{recipient_id}/reports', response_model=ResponseSchema)
def list_reports(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    reports = list_reports_for_recipient(db, recipient_id)
    print(f"[recipients] Listing {len(reports)} reports for recipient_id={recipient_id} (requested by {username})")
    out = [{'id': r.id, 'filename': r.filename, 'mime_type': r.mime_type, 'uploaded_at': r.uploaded_at.isoformat()} for r in reports]
    return ResponseSchema(code=200, status='success', message='Reports fetched', result={'reports': out})


@router.get('/debug/medical_reports/inspect')
def debug_inspect_reports(recipient_id: Optional[int] = None, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Temporary debug endpoint: returns DB url and counts. Requires authentication."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")
    try:
        import config
        from tables.medical_reports import MedicalReport
        total = db.query(MedicalReport).count()
        recip_count = None
        if recipient_id is not None:
            recip_count = db.query(MedicalReport).filter(MedicalReport.care_recipient_id == recipient_id).count()
        return {
            'db_url': str(getattr(config, 'DATABASE_URL', 'unknown')),
            'engine_url': str(getattr(getattr(config, 'engine', ''), 'url', 'unknown')),
            'total_reports': total,
            'recipient_reports': recip_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug inspect failed: {e}")


@router.post('/recipients/{recipient_id}/summarize', response_model=ResponseSchema)
def summarize_recipient_reports(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Re-run summarization for an existing recipient's uploaded reports and save the summary."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    try:
        reports = list_reports_for_recipient(db, recipient_id)
        texts = []
        for r in reports:
            if r.data:
                t = extract_text_from_bytes(r.data, r.mime_type)
                if t:
                    texts.append(t)

        if not texts:
            return ResponseSchema(code=200, status='success', message='No textual content found in reports', result={'summary': ''})

        combined = '\n\n'.join(texts)[:200000]
        summary = summarize_text_via_gemini(combined, target_words=250)

        recipient.report_summary = summary
        db.add(recipient)
        db.commit()

        return ResponseSchema(code=200, status='success', message='Summary generated', result={'summary': summary})
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to summarize reports: {e}")


@router.get('/recipients/{recipient_id}/reports/{report_id}/download')
def download_report(recipient_id: int, report_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Stream an individual report back to the authenticated caretaker if they own the recipient."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    report = db.query(__import__('tables.medical_reports', fromlist=['MedicalReport']).MedicalReport).filter_by(id=report_id, care_recipient_id=recipient_id).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    data = report.data or b''
    stream = BytesIO(data)
    headers = {
        'Content-Disposition': f'attachment; filename="{report.filename}"'
    }
    return StreamingResponse(stream, media_type=report.mime_type or 'application/octet-stream', headers=headers)


@router.post('/recipients/{recipient_id}/reports/base64', response_model=ResponseSchema)
def upload_report_base64(recipient_id: int, payload: dict, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Accept a JSON payload with base64-encoded file to simplify client uploads.
    Payload: { filename: str, mime_type: str, b64: str }
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    try:
        print(f"[recipients] Received base64 upload request for recipient_id={recipient_id} filename={payload.get('filename')}")
        filename = payload.get('filename')
        mime = payload.get('mime_type') or 'application/octet-stream'
        b64 = payload.get('b64')
        if not filename or not b64:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Missing filename or b64 payload')
        import base64
        data = base64.b64decode(b64)
        report = create_medical_report(db, recipient_id, filename, mime, data)

        # Aggregate texts and summarize (same as multipart flow)
        try:
            reports = list_reports_for_recipient(db, recipient_id)
            texts = []
            for r in reports:
                if r.data:
                    t = extract_text_from_bytes(r.data, r.mime_type)
                    if t:
                        texts.append(t)
            combined = '\n\n'.join(texts)
            combined = combined[:200000]
            summary = summarize_text_via_gemini(combined, target_words=250)
            recipient.report_summary = summary
            db.add(recipient)
            db.commit()
        except Exception as se:
            print('Summarization failed (base64):', se)

        return ResponseSchema(code=200, status='success', message='Report uploaded', result={'report_id': report.id, 'filename': report.filename})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save report: {str(e)}")
