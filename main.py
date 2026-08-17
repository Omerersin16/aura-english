import os
import io
import json
import re
import psycopg2
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
                
                # Yıldızlama özelliği için yeni sütunu güvenle ekle
                c.execute("SELECT column_name FROM information_schema.columns WHERE table_name='history' AND column_name='is_starred'")
                if not c.fetchone():
                    c.execute("ALTER TABLE history ADD COLUMN is_starred BOOLEAN DEFAULT FALSE")
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
                c.execute("INSERT INTO history (type, topic, content, feedback, created_at, is_starred) VALUES (%s, %s, %s, %s, %s, FALSE)",
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
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json"
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
                max_tokens=4096
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

class StarRequest(BaseModel):
    is_starred: bool

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
    1. Generate a HIGH-QUALITY reading passage in English (approx. 500-600 words).
    2. SENTENCE STRUCTURE (CRITICAL): Use a natural mix of short and long sentences. DO NOT write massive run-on sentences. Use proper punctuation.
    3. PARAGRAPH FORMATTING: Divide the text clearly into 5 distinct paragraphs. Use EXACTLY '\\n\\n' to separate paragraphs. DO NOT put any line breaks inside a paragraph.
    4. Extract exactly 10 CHALLENGING vocabulary words from the passage.
    5. For each extracted word, create a separate standalone "fill in the blank" sentence. The blank MUST be represented by '___' (three underscores).
    6. Provide EXACTLY 3 options for the blank, the correct_index, and a detailed Turkish explanation.
    7. Generate EXACTLY 5 multiple-choice comprehension questions with EXACTLY 4 options each.
    
    CRITICAL JSON RULES:
    - NEVER use double quotes (") inside your string values. Use single quotes (') instead.
    - Output ONLY valid JSON without any markdown or extra characters.
    
    Output STRICTLY in this JSON format:
    {{
      "title": "Your Generated Title Here",
      "passage": "Paragraph 1 text goes here.\\n\\nParagraph 2 text goes here.\\n\\nParagraph 3 text goes here.",
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
          "question": "What is the main idea?",
          "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
          "correct_index": 0,
          "explanation_tr": "Burada yazar şunu ifade etmektedir..."
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
            # TXT ve diğer metin dosyaları için güvenli okuma
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
    1. Reformat the text into a readable English passage (approx. 500 words, using '\\n\\n' for paragraphs).
    2. Replace exactly 10 advanced vocabulary words in the passage with numbered blanks formatted strictly as [1], [2] up to [10].
    3. Provide exactly 3 options and a Turkish explanation for each blank ID.
    4. Generate EXACTLY 5 challenging comprehension questions with 4 options each.
    
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

@app.post("/api/speaking/evaluate")
async def evaluate_speaking(audio: UploadFile = File(...), topic: str = Form("")):
    # 1. Aşama: Ses Dosyasını Groq Whisper ile Metne Çevir
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API yapılandırılmamış, ses analizi çalışamaz.")
        
    try:
        audio_bytes = await audio.read()
        print("🎙️ Groq Whisper'a ses gönderiliyor...")
        
        # Groq API'nin beklediği format: (dosya_adi, byte_verisi)
        transcription = groq_client.audio.transcriptions.create(
            file=(audio.filename, audio_bytes),
            model="whisper-large-v3",
            response_format="verbose_json"
        )
        user_text = transcription.text.strip()
        
        if not user_text:
            raise HTTPException(status_code=400, detail="Seste herhangi bir konuşma algılanamadı. Lütfen tekrar deneyin.")
            
    except Exception as e:
        print(f"Ses işleme hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Ses metne çevrilirken hata oluştu: {str(e)}")

    print(f"📝 Kullanıcının Söylediği: {user_text}")

    # 2. Aşama: Elde edilen metni Gemini ile analiz et (Telaffuz, Gramer, Akıcılık)
    chosen_topic = topic.strip() if topic.strip() else "Serbest Konuşma (Free Speaking)"
    
    prompt = f"""
    Topic Context: {chosen_topic}
    Student's Transcribed Speech: "{user_text}"
    
    Task:
    1. You are an expert English speaking coach. Evaluate the speech transcribed by an AI.
    2. IMPORTANT: If there are weirdly used words, it is highly likely the student mispronounced a word (e.g., saying 'sink' instead of 'think', or 'bad' instead of 'bed'). Point these pronunciation errors out!
    3. Analyze the text for grammar, vocabulary, and fluency (excessive use of filler words like 'um', 'uh').
    4. Provide an overall speaking score out of 100.
    
    CRITICAL JSON RULES:
    - NEVER use double quotes (") inside your string values. Use single quotes (') instead.
    - Output ONLY valid JSON without any markdown or extra characters.
    
    Output STRICTLY in this JSON format:
    {{
      "topic": "{chosen_topic}",
      "transcribed_text": "{user_text.replace('"', "'")}",
      "overall_score": 85,
      "feedback_tr": "Konuşman genel olarak akıcıydı ancak bazı kelimelerin telaffuzunda hatalar var...",
      "corrections": [
        {{
          "original": "I sink we should go",
          "corrected": "I think we should go",
          "explanation_tr": "Burada 'think' demek istedin ancak telaffuzun 'sink' (batmak) gibi duyulmuş. 'th' sesini çıkarırken dilini dişlerinin arasına almalısın."
        }}
      ],
      "improved_text": "An improved, natural-sounding version of what the student tried to say."
    }}
    """
    
    result = await get_ai_response(prompt)
    
    # Her ihtimale karşı transkripti orijinal haliyle JSON'a garanti olarak ekleyelim
    result["transcribed_text"] = user_text
    
    save_to_history("speaking", chosen_topic, user_text, json.dumps(result))
    return result

@app.get("/api/history")
async def get_history(page: int = 1, limit: int = 10, starred: bool = False):
    if not DATABASE_URL:
        return {"history": [], "total_pages": 1, "current_page": 1}
        
    conn = get_db_connection()
    c = conn.cursor()
    offset = (page - 1) * limit
    
    count_query = "SELECT COUNT(*) FROM history"
    if starred: count_query += " WHERE is_starred = TRUE"
    c.execute(count_query)
    total_items = c.fetchone()[0]
    total_pages = (total_items + limit - 1) // limit if total_items > 0 else 1
    
    query = "SELECT id, type, topic, content, feedback, created_at, is_starred FROM history"
    if starred: query += " WHERE is_starred = TRUE"
    query += f" ORDER BY id DESC LIMIT {limit} OFFSET {offset}"
    
    c.execute(query)
    rows = c.fetchall()
    conn.close()
    
    history_list = []
    for r in rows:
        history_list.append({
            "id": r[0], "type": r[1], "topic": r[2], "content": r[3], 
            "feedback": r[4], "created_at": r[5][:16].replace("T", " "),
            "is_starred": bool(r[6] if len(r) > 6 else False)
        })
    return {"history": history_list, "total_pages": total_pages, "current_page": page}

@app.delete("/api/history/{item_id}")
async def delete_history(item_id: int):
    if not DATABASE_URL: return {"success": False}
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute("SELECT is_starred FROM history WHERE id = %s", (item_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Kayıt bulunamadı.")
    if row[0]: # is_starred == True
        conn.close()
        raise HTTPException(status_code=400, detail="Yıldızlı kayıtlar silinemez! Önce yıldızı kaldırın.")
        
    c.execute("DELETE FROM history WHERE id = %s", (item_id,))
    conn.commit()
    conn.close()
    return {"success": True}

@app.put("/api/history/{item_id}/star")
async def toggle_star(item_id: int, req: StarRequest):
    if not DATABASE_URL: return {"success": False}
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE history SET is_starred = %s WHERE id = %s", (req.is_starred, item_id))
    conn.commit()
    conn.close()
    return {"success": True}

app.mount("/", StaticFiles(directory="static", html=True), name="static")