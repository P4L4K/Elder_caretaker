import os
from io import BytesIO
import re

try:
    import requests
except Exception:
    requests = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    import docx
except Exception:
    docx = None

try:
    from PIL import Image
    import pytesseract
except Exception:
    Image = None
    pytesseract = None


def _simple_text_from_pdf_bytes(b: bytes):
    if not PdfReader:
        return ''
    try:
        reader = PdfReader(BytesIO(b))
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or '')
            except Exception:
                parts.append('')
        return '\n'.join(parts)
    except Exception:
        return ''


def _simple_text_from_docx_bytes(b: bytes):
    if not docx:
        return ''
    try:
        doc = docx.Document(BytesIO(b))
        paragraphs = [p.text for p in doc.paragraphs]
        return '\n'.join(paragraphs)
    except Exception:
        return ''


def _simple_text_from_image_bytes(b: bytes):
    if not Image or not pytesseract:
        return ''
    try:
        img = Image.open(BytesIO(b))
        text = pytesseract.image_to_string(img)
        return text
    except Exception:
        return ''


def extract_text_from_bytes(b: bytes, mime: str = None) -> str:
    """Attempt to extract text from bytes based on mime type. Returns empty string if unable."""
    mime = (mime or '').lower()
    if 'pdf' in mime:
        return _simple_text_from_pdf_bytes(b)
    if 'word' in mime or 'officedocument' in mime or mime.endswith('.docx'):
        return _simple_text_from_docx_bytes(b)
    if mime.startswith('image'):
        return _simple_text_from_image_bytes(b)

    # Last-ditch: try to decode as utf-8 text
    try:
        text = b.decode('utf-8')
        # if text looks binary, return empty
        if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text):
            return ''
        return text
    except Exception:
        return ''


def summarize_text_via_gemini(text: str) -> str:
    """Call out to an external Gemini endpoint (config via env). If not configured, fall back to a simple local summarizer."""
    if not text:
        return ''

    endpoint = os.environ.get('GEMINI_API_ENDPOINT')
    api_key = os.environ.get('GEMINI_API_KEY')

    # Basic local fallback summarizer: take first 3 sentences
    def local_summary(s: str) -> str:
        # split into sentences naively
        sentences = re.split(r'(?<=[.!?])\s+', s.strip())
        summary = ' '.join(sentences[:3])
        return summary[:2000]

    if not endpoint or not api_key or not requests:
        return local_summary(text)

    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'input': text,
            'task': 'summarize',
            'max_tokens': 512
        }
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            # Expecting a 'summary' field; fallback to raw text
            summary = data.get('summary') or data.get('output') or data.get('result')
            if isinstance(summary, list):
                summary = ' '.join(summary)
            if not summary:
                return local_summary(text)
            return summary[:2000]
        else:
            return local_summary(text)
    except Exception:
        return local_summary(text)
