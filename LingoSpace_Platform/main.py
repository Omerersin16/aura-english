import os
import json
import re
import psycopg2
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
import PyPDF2
from groq import Groq

# ==========================================
# 1. AYARLAR VE YAPILANDIRMA
# ==========================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip("[]'\"")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip("[]'\"")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip("[]'\"")

app = FastAPI(title="AuraEnglish")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ==========================================
# 2. VERİTABANI MODÜLÜ
# ==========================================
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    if not DATABASE_URL:
        return
    try:
        with get_db_connection() as conn:
            with conn.cursor() as c:
                c.execute('''CREATE TABLE IF NOT EXISTS history
                             (id SERIAL PRIMARY KEY, type TEXT, topic TEXT, content TEXT, feedback TEXT, created_at TEXT)''')
            conn.commit()
    except Exception as e:
        print(f"DB Başlatma Hatası: {e}")

init_db()

def save_to_history(type_val: str, topic: str, content: str, feedback: str):
    if not DATABASE_URL:
        return
    try:
        with get_db_connection() as conn:
            with conn.cursor() as c:
                c.execute("INSERT INTO history (type, topic, content, feedback, created_at) VALUES (%s, %s, %s, %s, %s)",
                          (type_val, topic, content, feedback, datetime.now().isoformat()))
            conn.commit()
    except Exception as e:
        print(f"Geçmiş Kaydetme Hatası: {e}")

# ==========================================
# 3. YAPAY ZEKA (AI) MODÜLÜ - REST API MİMARİSİ
# ==========================================
def parse_ai_json(text: str) -> dict:
    text = text.replace("```json", "").replace("```", "").strip()
    # Markdown blok temizliği
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0]
    text = text.strip()
    
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError as e:
        try:
            fixed_text = re.sub(r'"\s*\n\s*"', '",\n"', text)
            if not fixed_text.endswith("}"): 
                fixed_text += "}"
            return json.loads(fixed_text, strict=False)
        except Exception:
            print(f"JSON Parse Hatası Alınan Ham Metin:\n{text}")
            raise HTTPException(status_code=500, detail="Yapay zeka geçerli bir JSON üretemedi.")

def call_gemini_rest_api(prompt: str) -> str:
    """Google SDK kullanmadan, doğrudan API'ye istek atar."""
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API Key bulunamadı.")
    
    # 2026 Uyumlu Güncel Model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 8192  # Uzun metinler için artırıldı
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        print(f"\n--- GOOGLE API HATASI ---")
        print(f"Kodu: {response.status_code} | Detay: {response.text}\n")
        raise Exception(f"Gemini API Hatası: {response.status_code}")
        
    data = response.json()
    return data['candidates'][0]['content']['parts'][0]['text']

async def get_ai_response(prompt: str) -> dict:
    prompt += """
    \n\nCRITICAL JSON RULES:
    1. Output ONLY valid JSON starting with '{' and ending with '}'. No markdown wrappers outside, no extra text.
    2. Ensure ALL brackets and braces are properly closed.
    3. Escape internal quotes or use single quotes for inner texts.
    """

    # Adım 1: Doğrudan HTTP İsteği ile Gemini (3.5-flash)
    try:
        print("⚡ Gemini-3.5-flash'a İstek Atılıyor...")
        response_text = call_gemini_rest_api(prompt)
        return parse_ai_json(response_text)
        
    except Exception as e:
        print(f"⚠️ Gemini Yanıt Veremedi: {e}. Görev Groq'a Devrediliyor...")
        
        # Adım 2: Yedek Groq Sistemi
        try:
            if not groq_client: raise ValueError("Groq client yok.")
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                response_format={"type": "json_object"},
                max_tokens=8192
            )
            return parse_ai_json(chat_completion.choices[0].message.content)
            
        except Exception as e2:
            print(f"❌ Tüm servisler çöktü: {e2}")
            raise HTTPException(status_code=500, detail="Yapay zeka servisleri şu an yanıt vermiyor.")

# ==========================================
# 4. VERİ MODELLERİ (SCHEMAS)
# ==========================================
class ReadingRequest(BaseModel):
    level: str
    topic: str

class WritingRequest(BaseModel):
    topic: str
    user_text: str

# ==========================================
# 5. API YÖNLENDİRMELERİ (ROUTERS)
# ==========================================
@app.post("/api/reading/generate")
async def generate_reading(req: ReadingRequest):
    topic_instruction = f"Topic: {req.topic}" if req.topic.strip() else "Topic: Choose a random, engaging topic."
    
    prompt = f"""
    Target Level: {req.level}
    {topic_instruction}
    
    Task:
    1. Generate a HIGH-QUALITY, LONG reading passage in English. The passage MUST be at least 250-300 words and divided into 4-5 paragraphs. DO NOT put any blanks inside this main passage. Use '\\n\\n' to separate paragraphs.
    2. Extract exactly 10 CHALLENGING vocabulary words (adjectives, adverbs, complex verbs) from the passage.
    3. For each extracted word, create a separate standalone "fill in the blank" sentence testing that word. The blank must be represented by '___'.
    4. Provide 3 options, the correct index, and a Turkish explanation for each blank.
    5. Generate EXACTLY 5 CHALLENGING multiple-choice comprehension questions based on the text. Focus on inference and main idea.
    
    Output STRICTLY in this JSON format:
    {{
      "title": "Title in English",
      "passage": "Paragraph 1 text...\\n\\nParagraph 2 text...\\n\\nParagraph 3 text...",
      "fill_in_blanks": [
        {{
          "sentence_with_blank": "The explorer wanted to ___ the ancient ruins.",
          "options": ["discover", "destroy", "ignore"],
          "correct_index": 0,
          "explanation_tr": "Doğru cevap 'discover' (keşfetmek). 'destroy' ve 'ignore' cümleye anlamsız olur."
        }}
      ],
      "questions": [
        {{
          "question": "A challenging inference question?",
          "options": ["A", "B", "C", "D"],
          "correct_index": 0,
          "explanation_tr": "Doğru cevabın mantıksal Türkçe açıklaması."
        }}
      ]
    }}
    """
    result = await get_ai_response(prompt)
    save_to_history("reading", req.topic or result.get("title", "Rastgele Okuma"), json.dumps(result), "{}")
    return result

@app.post("/api/writing/evaluate")
async def evaluate_writing(req: WritingRequest):
    chosen_topic = req.topic.strip() if req.topic and req.topic.strip() else "A memorable day in your life"
    prompt = f"""
    Topic Given: {chosen_topic}
    Student's Text: "{req.user_text}"
    Analyze the text for grammar and vocabulary.
    Output STRICTLY in this JSON format:
    {{
      "topic": "{chosen_topic}",
      "overall_score": 85,
      "feedback_tr": "Feedback in Turkish.",
      "corrections": [
        {{
          "original": "wrong",
          "corrected": "right",
          "explanation_tr": "Explanation in Turkish."
        }}
      ],
      "improved_text": "Improved text."
    }}
    """
    result = await get_ai_response(prompt)
    save_to_history("writing", chosen_topic, req.user_text, json.dumps(result))
    return result

@app.post("/api/file/analyze")
async def analyze_file(file: UploadFile = File(...)):
    content = ""
    try:
        file_bytes = await file.read()
        filename_lower = file.filename.lower()
        
        if filename_lower.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    content += extracted + "\n"
        else:
            # TXT ve diğer metin dosyaları için güvenli okuma (UTF-8 ve ISO-8859-1 desteği)
            try:
                content = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                content = file_bytes.decode("latin-1", errors="ignore")
                
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dosya okunamadı: {str(e)}")

    if not content.strip():
        raise HTTPException(status_code=400, detail="Dosya boş veya okunabilir metin bulunamadı.")

    content = content[:10000]

    prompt = f"""
    Analyze the following text extracted from a file:
    "{content}"
    
    Task:
    1. Reformat the text into a readable English passage (at least 250-300 words, using '\\n\\n' for paragraphs).
    2. Replace exactly 10 advanced vocabulary words in the passage with numbered blanks formatted strictly as [1], [2] up to [10].
    3. Provide the options and Turkish explanation for each blank ID.
    4. Generate EXACTLY 5 challenging comprehension questions.
    
    Output STRICTLY in this JSON format:
    {{
      "title": "A Suitable Title",
      "passage": "Cleaned up text with [1], [2] blanks...\\n\\nNext paragraph...",
      "blanks": [
        {{
          "id": 1,
          "options": ["opt1", "opt2", "opt3"],
          "correct_index": 0,
          "explanation_tr": "Türkçe detaylı açıklama."
        }}
      ],
      "questions": [
        {{
          "question": "A challenging question?",
          "options": ["A", "B", "C", "D"],
          "correct_index": 0,
          "explanation_tr": "Açıklama"
        }}
      ]
    }}
    """
    result = await get_ai_response(prompt)
    save_to_history("file", file.filename, json.dumps(result), "{}")
    return result

@app.get("/api/history")
async def get_history():
    if not DATABASE_URL:
        return {"history": []}
        
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, type, topic, content, feedback, created_at FROM history ORDER BY id DESC LIMIT 30")
    rows = c.fetchall()
    conn.close()
    
    history_list = []
    for r in rows:
        history_list.append({
            "id": r[0], 
            "type": r[1], 
            "topic": r[2], 
            "content": r[3], 
            "feedback": r[4], 
            "created_at": r[5][:16].replace("T", " ")
        })
    return {"history": history_list}

app.mount("/", StaticFiles(directory="static", html=True), name="static")