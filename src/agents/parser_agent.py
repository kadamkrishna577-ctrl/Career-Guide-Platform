import io
import pymupdf
import pytesseract
from PIL import Image
from src.config import get_llm
from src.schemas import ParsedResume


def extract_pdf_text_with_ocr(pdf_path:str) -> tuple[str, bool]:
    
    """Extracts text from a PDF using PyMuPDF.
    If a page has < 80 characters (tricky/scanned PDF), falls back to Tesseract OCR."""
    
    doc = pymupdf.open(pdf_path)
    extracted_pages = []
    ocr_used = False
    
    for page in doc:
        native_text = page.get_text("text").strip()
        
        if len(native_text) >= 80:
            extracted_pages.append(native_text)
        else:
            
            ocr_used = True
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(img)
            extracted_pages.append(ocr_text)
            
    doc.close()
    
    return "\n\n".join(extracted_pages), ocr_used

def resume_parser_node(state: dict) -> dict:
    """LangGraph Node for Agent1 . Reads 'pdf_path' from state, extracts structured data via Gemini,
    and returns updates to the shared graph state."""
    
    pdf_path = state["pdf_path"]
    raw_text, ocr_triggred = extract_pdf_text_with_ocr(pdf_path)
    
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(ParsedResume)
    
    prompt = f"""You are Agent 1 (Resume Parser). Extract all candidate details, technical skills, 
    education, work experience, and project bullet points from the raw resume text below into 
    the exact Pydantic schema. Do not fabricate or hallucinate any information.

    RAW RESUME TEXT:
    {raw_text}"""
    
    parsed_resume: ParsedResume = structured_llm.invoke(prompt)
    
    return {
        "raw_resume_text" : raw_text,
        "ocr_used": ocr_triggred,
        "parsed_resume": parsed_resume
    }