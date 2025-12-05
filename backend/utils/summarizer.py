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


def extract_clinical_findings(medical_text: str) -> str:
    """Extract key clinical information from medical documents using Gemini.
    
    Focuses on:
    - Diagnoses / diseases
    - Symptoms and complaints
    - Abnormal test results
    - Important vitals (if high/low)
    - Medications prescribed (with dose if mentioned)
    - Relevant dates
    - Doctor assessments or impressions
    - Follow-up or recommended care
    
    Returns extracted findings without filler language.
    """
    if not medical_text:
        return ''

    api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key or not requests:
        print('[summarizer] Gemini API key not configured; cannot extract clinical findings')
        return ''

    clinical_extraction_prompt = """You are a clinical information extractor. Summarize the following medical documents into clear, precise medical findings.

Focus ONLY on:
• Diagnoses / diseases the patient has
• Symptoms and complaints
• Abnormal test results
• Important vitals (if high/low)
• Medications prescribed (with dose if mentioned)
• Relevant dates
• Doctor assessments or impressions
• Any follow-up or recommended care

Do NOT include filler language.
Do NOT rewrite entire paragraphs.
Do NOT add content not found in the report.
Just extract key medical facts and present them clearly.

Medical Document:
{text}"""

    try:
        prompt = clinical_extraction_prompt.format(text=medical_text)
        
        # Use Google Generative AI library if available, otherwise fall back to REST API
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            print('[summarizer] Clinical extraction via genai library successful')
            return response.text if response.text else ''
        except Exception as e:
            print(f'[summarizer] genai library not available ({e}), trying REST API...')
            
            # Fall back to REST API
            headers = {
                'Content-Type': 'application/json'
            }
            payload = {
                'contents': [
                    {
                        'parts': [
                            {
                                'text': prompt
                            }
                        ]
                    }
                ]
            }
            
            url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}'
            print(f'[summarizer] Calling Gemini REST API for clinical extraction')
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            print(f'[summarizer] Gemini response status: {resp.status_code}')
            
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    # Extract text from Gemini response
                    if 'candidates' in data and len(data['candidates']) > 0:
                        candidate = data['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            parts = candidate['content']['parts']
                            if len(parts) > 0 and 'text' in parts[0]:
                                return parts[0]['text']
                except Exception as e:
                    print(f'[summarizer] Failed to parse Gemini response: {e}')
                    return ''
            else:
                print(f'[summarizer] Gemini API returned status {resp.status_code}')
                return ''
                
    except Exception as e:
        print(f'[summarizer] Clinical extraction failed: {e}')
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
        # Build the URL with API key as a query parameter
        url = f"{endpoint}?key={api_key}"
        
        # Set up headers and payload according to Gemini API spec
        headers = {
            'Content-Type': 'application/json'
        }
        
        # Log the input text for debugging
        print(f'[summarizer] Input text length: {len(text)} characters')
        if len(text) > 200:
            print(f'[summarizer] Text preview: {text[:100]}...{text[-100:]}')
        else:
            print(f'[summarizer] Text: {text}')

        # Format the prompt with the actual text - CONCISE VERSION
        prompt = """You are a clinical information extractor and health-risk analyst.

Extract ONLY the most critical medical information from the document.  
Summarize concisely, suitable for quick understanding.

Provide a short list of the key points under each heading:

• Diagnoses: list the main disease(s) only.  
• Symptoms / Complaints: list only the most significant symptoms.  
• Important Vitals: only abnormal values (high/low/critical).  
• Medications: list main medications with dosage if available.  
• Follow-up Advice: summarize the most important advice in one line per item.

Rules:
- Keep it short, 1–2 bullet points per section.
- Do NOT include filler or detailed explanations.
- Skip any category with no information.
- Don't include unnecessary symbols like "*" in the final answer keep it professional 


Based on the *Diagnoses*, provide ONLY these three recommendations in **short format**:

1. Recommended Temperature Range (°C)  
2. Recommended Relative Humidity Range (%)  
3. Recommended Indoor Air Quality (PM2.5 / AQI)

Rules:
- Give general safe ranges if diagnosis is unclear.
- Keep it extremely concise (1 line per item).
- Don't include unnecessary symbols like "*" in the final answer keep it professional 
========================
Medical Document:
{text}""".format(text=text.strip())
        
        # Structure the payload according to Gemini API spec
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 2000,
                "temperature": 0.2,
                "topP": 0.8,
                "topK": 40
            }
        }
        
        print(f'[summarizer] Calling Gemini endpoint {url} (payload words approx {len(prompt.split())})')
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        print(f'[summarizer] Gemini response status: {getattr(resp, "status_code", "?")}')
        
        if resp.status_code == 200:
            try:
                data = resp.json()
                # Extract text from Gemini response
                if 'candidates' in data and data['candidates']:
                    candidate = data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if parts and 'text' in parts[0]:
                            full_response = parts[0]['text']
                            
                            # Split response into summary and thresholds
                            if "ENVIRONMENTAL THRESHOLDS" in full_response:
                                summary, thresholds_section = full_response.split("ENVIRONMENTAL THRESHOLDS", 1)
                                thresholds = parse_environmental_thresholds("ENVIRONMENTAL THRESHOLDS" + thresholds_section)
                                
                                # Add thresholds to summary
                                summary += "\n\nENVIRONMENTAL RECOMMENDATIONS:\n"
                                if thresholds["temperature"]:
                                    summary += f"• Temperature: {thresholds['temperature']}\n"
                                if thresholds["humidity"]:
                                    summary += f"• Humidity: {thresholds['humidity']}\n"
                                if thresholds["air_quality"]:
                                    summary += f"• Air Quality: {thresholds['air_quality']}\n"
                                
                                for rec in thresholds["additional_recommendations"]:
                                    summary += f"• {rec}\n"
                            else:
                                summary = full_response
                            
                            # Return the full response without truncation
                            print(f'[summarizer] Gemini produced {len(summary.split())} words')
                            
                            # Ensure proper line breaks for bullet points
                            summary = summary.replace('•', '\n•')  # Add newline before each bullet
                            summary = '\n'.join(line.strip() for line in summary.split('\n'))  # Clean up whitespace
                            
                            return summary
            except Exception as e:
                print(f'[summarizer] Failed to parse Gemini response: {e}')
                return local_summary(text)
        
        print(f'[summarizer] Gemini call failed with status {resp.status_code} — using local summary')
        if hasattr(resp, 'text'):
            print(f'[summarizer] Response text: {resp.text[:500]}...')  # Log first 500 chars of response
        return local_summary(text)
        
    except Exception as e:
        print(f'[summarizer] Error in summarize_text_via_gemini: {str(e)}')
        return local_summary(text)

def parse_environmental_thresholds(thresholds_text: str) -> dict:
    """Parse environmental thresholds from the Gemini response.
    
    Args:
        thresholds_text: Raw text containing the environmental thresholds section
        
    Returns:
        dict: Parsed thresholds with keys: temperature, humidity, air_quality, additional_recommendations
    """
    thresholds = {
        "temperature": None,
        "humidity": None,
        "air_quality": None,
        "additional_recommendations": []
    }
    
    if not thresholds_text:
        return thresholds
    
    # For the new concise format with numbered items
    if '1. Recommended Temperature' in thresholds_text:
        lines = thresholds_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('1.'):
                temp = line.split(':', 1)[-1].strip()
                if '°C' not in temp and 'C' in temp:  # Handle case where ° is missing
                    temp = temp.replace('C', '°C')
                thresholds["temperature"] = temp
            elif line.startswith('2.'):
                thresholds["humidity"] = line.split(':', 1)[-1].strip()
            elif line.startswith('3.'):
                thresholds["air_quality"] = line.split(':', 1)[-1].strip()
    else:
        # Fallback to old format parsing
        # Extract temperature (supports formats like "20-24°C" or "20°C to 24°C")
        temp_match = re.search(r'(\d+)\s*°?C?\s*[\-\s]+\s*(\d+)\s*°?C', thresholds_text)
        if temp_match:
            min_temp, max_temp = temp_match.groups()
            thresholds["temperature"] = f"{min_temp}°C - {max_temp}°C"
        
        # Extract humidity (supports formats like "40-55%" or "40% to 55%")
        hum_match = re.search(r'(\d+)\s*[%]\s*[-\s]+\s*(\d+)\s*%', thresholds_text)
        if hum_match:
            min_hum, max_hum = hum_match.groups()
            thresholds["humidity"] = f"{min_hum}% - {max_hum}%"
        
        # Extract air quality (looks for AQI or PM2.5 values)
        aqi_match = re.search(r'(?:AQI|PM2\.5)\s*[:\-]?\s*([^.\n]+)', thresholds_text, re.IGNORECASE)
        if aqi_match:
            thresholds["air_quality"] = aqi_match.group(1).strip()
    
    return thresholds
