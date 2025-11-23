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
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import pdf2image
except Exception:
    pdf2image = None

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
    """Extract text from PDF bytes, with OCR fallback."""
    text = ''
    # 1. Try with pdfplumber - often good for structured PDFs
    if pdfplumber:
        try:
            with pdfplumber.open(BytesIO(b)) as pdf:
                parts = [p.extract_text() or '' for p in pdf.pages]
                text = '\n'.join(parts).strip()
        except Exception:
            text = ''  # ignore failures

    # 2. If no text, try PyPDF2
    if not text and PdfReader:
        try:
            reader = PdfReader(BytesIO(b))
            parts = [page.extract_text() or '' for page in reader.pages]
            text = '\n'.join(parts).strip()
        except Exception:
            text = ''  # ignore failures

    # 3. If still no text, and this is a PDF, try OCR
    if not text and pdf2image:
        print('[summarizer] PDF text extraction failed, trying OCR fallback...')
        try:
            images = pdf2image.convert_from_bytes(b)
            parts = []
            for i, image in enumerate(images):
                try:
                    # TODO: consider adding language hints if known
                    parts.append(pytesseract.image_to_string(image) or '')
                except Exception as e:
                    print(f'[summarizer] OCR on page {i} failed: {e}')
            text = '\n'.join(parts).strip()
            print(f'[summarizer] OCR fallback produced {len(text)} chars')
        except Exception as e:
            # This can happen if poppler is not installed
            print(f'[summarizer] OCR fallback failed entirely: {e}')
            text = ''

    return text


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


def summarize_text_via_gemini(text: str, target_words: int = 250) -> str:
    """Call out to an external Gemini endpoint (config via env) to get a summary of approximately target_words.
    If not configured, fall back to a simple local summarizer.
    """
    if not text:
        return ''

    endpoint = os.environ.get('GEMINI_API_ENDPOINT')
    api_key = os.environ.get('GEMINI_API_KEY')

    # Basic local fallback summarizer: take sentences until target_words reached
    def local_summary(s: str) -> str:
        if not s:
            return ''
        sentences = re.split(r'(?<=[.!?])\s+', s.strip())
        out_words = []
        for sent in sentences:
            parts = sent.split()
            if not parts:
                continue
            # if adding this sentence goes over target, stop and break
            if len(out_words) + len(parts) > target_words:
                break
            out_words.extend(parts)
            if len(out_words) >= target_words:
                break
        if not out_words:
            # fallback: just take the first N words from the raw text
            return ' '.join(s.split()[:target_words])
        return ' '.join(out_words)

    if not endpoint or not api_key or not requests:
        print('[summarizer] Gemini not configured or requests missing; using local summary fallback')
        return local_summary(text)

    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        # Use a general prompt wrapper in case the Gemini endpoint expects prompt-based input
        prompt = f"Please provide a concise medical summary of the following documents in approximately {target_words} words. Focus on diagnoses, medications, findings, and relevant dates. Use short paragraphs.\n\nDocuments:\n{text}"
        payload = {
            'prompt': prompt,
            'max_tokens': 1500
        }
        print(f'[summarizer] Calling Gemini endpoint {endpoint} (payload words approx {len(payload["prompt"].split())})')
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
        print(f'[summarizer] Gemini response status: {getattr(resp, "status_code", "?" )}')
        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                data = {}
            # Try several common response keys
            summary = data.get('summary') or data.get('output') or data.get('result') or data.get('text') or data.get('response')
            if isinstance(summary, list):
                summary = ' '.join(summary)
            if not summary:
                print('[summarizer] Gemini returned empty summary payload, falling back to local summary')
                return local_summary(text)
            # Truncate to target_words
            words = summary.split()
            out = ' '.join(words[:min(len(words), target_words)])
            print(f'[summarizer] Gemini produced {len(out.split())} words (truncated to {target_words})')
            return out
        else:
            print('[summarizer] Gemini call failed, status not 200 — using local summary')
            return local_summary(text)
    except Exception:
        return local_summary(text)
