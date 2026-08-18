import os
import io
import json
import re
import random
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
import asyncio
import tempfile
import uuid
import edge_tts

# ==========================================
# 1. AYARLAR VE YAPILANDIRMA
# ==========================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip("[]'\"")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip("[]'\"")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip("[]'\"")
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

RANDOM_TOPIC_POOL = [
    "The Psychology of Decision Making in Modern Video Games",
    "Artificial Intelligence and the Future of Autonomous Space Exploration",
    "The Mysterious Disappearance of Ancient Maritime Trade Routes",
    "How Cryptography Shaped the Outcome of World War II",
    "The Science of Sleep and Why Dreams Remain a Mystery",
    "The Rise of Cyberpunk Culture in Architecture and Fashion",
    "Deep Sea Ecosystems and Creatures That Glow in the Dark",
    "The Economics of E-Sports and Professional Gaming Leagues",
    "The History and Evolution of Coffee Culture Around the World",
    "Quantum Computing: How It Will Change the Digital World Forever",
    "The Architecture of Frank Lloyd Wright and Harmony with Nature",
    "The Psychology Behind Viral Marketing and Internet Memes",
    "Surviving Extreme Environments: How the Human Body Reacts to High Altitudes",
    "The Secret History of Renaissance Art Forgeries",
    "The Impact of Virtual Reality on Modern Education and Training"
]

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
                
                c.execute("SELECT column_name FROM information_schema.columns WHERE table_name='history' AND column_name='is_starred'")
                if not c.fetchone():
                    c.execute("ALTER TABLE history ADD COLUMN is_starred BOOLEAN DEFAULT FALSE")
                
                c.execute('''CREATE TABLE IF NOT EXISTS vocabulary
                             (id SERIAL PRIMARY KEY, word TEXT, translation TEXT, is_learned BOOLEAN DEFAULT FALSE, created_at TEXT)''')
            conn.commit()
    except Exception as e:
        print(f"DB Başlatma Hatası: {e}")

init_db()

os.makedirs("static/audio", exist_ok=True)

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
# 3. YENİLMEZ (6 KADEMELİ) YAPAY ZEKA MİMARİSİ
# ==========================================
def parse_ai_json(text: str) -> dict:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = text.replace("```json", "").replace("```", "").strip()
    
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    clean_json_text = json_match.group(0) if json_match else text

    try:
        return json.loads(clean_json_text, strict=False)
    except json.JSONDecodeError:
        try:
            fixed_text = re.sub(r'"\s*\n\s*"', '",\n"', clean_json_text)
            if fixed_text.count('{') > fixed_text.count('}'):
                fixed_text += '}' * (fixed_text.count('{') - fixed_text.count('}'))
            if fixed_text.count('[') > fixed_text.count(']'):
                fixed_text += ']' * (fixed_text.count('[') - fixed_text.count(']'))
            return json.loads(fixed_text, strict=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Yapay zeka geçerli JSON üretemedi. Hata: {str(e)}")

def call_gemini_rest_api(prompt: str, model_name: str) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API Key bulunamadı.")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json"
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Gemini API Hatası ({model_name} - {response.status_code}): {response.text}")
        
    data = response.json()
    return data['candidates'][0]['content']['parts'][0]['text']

def call_groq_api(prompt: str) -> str:
    if not groq_client:
        raise ValueError("Groq client yapılandırılmamış.")
    
    model_name = "openai/gpt-oss-120b"
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a strict JSON generator. Output ONLY valid JSON starting with '{' and ending with '}'."},
            {"role": "user", "content": prompt}
        ],
        model=model_name,
        max_tokens=4000,
        temperature=0.7
    )
    raw_content = chat_completion.choices[0].message.content or ""
    if not raw_content.strip():
        raise ValueError("Groq boş yanıt döndürdü.")
    return raw_content

def call_openrouter_api(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("OpenRouter API anahtarı ayarlanmamış.")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "AuraEnglish"
    }
    payload = {
        "model": "openrouter/free",
        "messages": [
            {"role": "system", "content": "You are a strict JSON generator. Output ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 4000
    }
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"OpenRouter Hatası ({response.status_code}): {response.text}")
        
    data = response.json()
    return data['choices'][0]['message']['content']

async def get_ai_response(prompt: str) -> dict:
    prompt += """
    \n\nCRITICAL JSON RULES:
    1. Output ONLY valid JSON starting with '{' and ending with '}'. No extra text, no markdown.
    2. Ensure ALL brackets and braces are properly closed.
    """
    
    # 6 Kademeli Şelale (Cascade) Sistemi
    gemini_models = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
    last_error = ""

    # KADEME 1-4: Gemini Modelleri
    for model in gemini_models:
        try:
            print(f"⚡ [{model}] Modeline İstek Atılıyor...")
            response_text = call_gemini_rest_api(prompt, model)
            return parse_ai_json(response_text)
        except Exception as e:
            print(f"⚠️ {model} Başarısız ({e}). Sıradaki modele geçiliyor...")
            last_error = str(e)
            
    # KADEME 5: Groq (GPT-OSS-120B)
    try:
        print("⚡ [Groq: openai/gpt-oss-120b] Deneniyor...")
        response_text = call_groq_api(prompt)
        return parse_ai_json(response_text)
    except Exception as e:
        print(f"⚠️ Groq Başarısız ({e}). Son kale OpenRouter'a geçiliyor...")
        last_error = str(e)
        
    # KADEME 6: OpenRouter (Son Kale)
    try:
        print("⚡ [OpenRouter: openrouter/free] Deneniyor...")
        response_text = call_openrouter_api(prompt)
        return parse_ai_json(response_text)
    except Exception as e:
        print(f"⚠️ OpenRouter Başarısız ({e}).")
        last_error = str(e)

    raise HTTPException(status_code=500, detail=f"Tüm 6 Yapay Zeka Servisi de Çöktü! Son Hata: {last_error}")

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

class ListeningRequest(BaseModel):
    level: str
    topic: str
    speakers: int

class VocabularyRequest(BaseModel):
    word: str
    translation: str

class VocabStatusRequest(BaseModel):
    is_learned: bool

# ==========================================
# 5. API YÖNLENDİRMELERİ (ROUTERS)
# ==========================================

# ------------------------------------------
# YENİ: KARŞILIKLI SOHBET UÇ NOKTASI (6 Kademeli)
# ------------------------------------------
@app.post("/api/chat/voice")
async def voice_chat(audio: UploadFile = File(...), history: str = Form("[]")):
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API yapılandırılmamış.")

    # 1. Kullanıcının Sesini Metne Çevir (Whisper her zaman Groq'ta çalışır)
    try:
        audio_bytes = await audio.read()
        print("🎙️ Groq Whisper'a sohbet sesi gönderiliyor...")
        transcription = groq_client.audio.transcriptions.create(
            file=(audio.filename, audio_bytes),
            model="whisper-large-v3",
            response_format="verbose_json"
        )
        user_text = transcription.text.strip()
        if not user_text:
            raise HTTPException(status_code=400, detail="Seste konuşma algılanamadı.")
    except Exception as e:
        print(f"Sohbet Ses işleme hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Ses anlaşılamadı: {str(e)}")

    print(f"🗣️ Sen: {user_text}")

    # 2. Geçmişi Parse Et ve Sohbet Bağlamını Hazırla
    try:
        chat_history = json.loads(history)
    except:
        chat_history = []

    system_prompt = """You are a highly friendly, engaging, and empathetic English conversation partner.
    You are having a real-life casual voice call with the user.
    RULES:
    1. Act like a real human friend. Use natural conversational fillers (e.g., "Oh wow", "That's interesting", "Hmm", "Actually").
    2. Keep your responses VERY SHORT (1 to 3 sentences maximum). It's a fast-paced spoken conversation, not a long essay.
    3. Always end your response by asking a simple, natural question to keep the conversation flowing.
    4. The user is at a B1-B2 English level.
    5. NEVER use markdown, bold text (*), lists, or formatting. Just pure spoken text.
    6. Never break character. Do not act like an AI."""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        role = "assistant" if msg["role"] == "ai" else "user"
        messages.append({"role": role, "content": msg["text"]})
    messages.append({"role": "user", "content": user_text})

    ai_text = ""
    gemini_models = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
    
    # Sohbet için Gemini Serisi
    for model in gemini_models:
        try:
            print(f"⚡ [Sohbet] Gemini ({model}) deneniyor...")
            gemini_prompt = f"{system_prompt}\n\nSohbet Geçmişi:\n"
            for m in messages[1:]:
                gemini_prompt += f"{m['role'].upper()}: {m['content']}\n"
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
            payload = {"contents": [{"parts": [{"text": gemini_prompt}]}]}
            res = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
            if res.status_code == 200:
                ai_text = res.json()['candidates'][0]['content']['parts'][0]['text'].strip()
                break
        except Exception as e:
            print(f"⚠️ Sohbet Gemini ({model}) Başarısız: {e}")

    # Sohbet için Groq
    if not ai_text:
        try:
            print("⚡ [Sohbet] Groq (openai/gpt-oss-120b) deneniyor...")
            chat_completion = groq_client.chat.completions.create(
                messages=messages,
                model="openai/gpt-oss-120b",
                max_tokens=150, 
                temperature=0.7
            )
            ai_text = chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Sohbet Groq Başarısız: {e}")

    # Sohbet için OpenRouter
    if not ai_text:
        try:
            print("⚡ [Sohbet] OpenRouter (openrouter/free) deneniyor...")
            headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": "openrouter/free", "messages": messages, "max_tokens": 150}
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            if res.status_code == 200:
                ai_text = res.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"⚠️ Sohbet OpenRouter Başarısız: {e}")

    if not ai_text:
        raise HTTPException(status_code=500, detail="Tüm 6 AI modeli de sohbet için çöktü.")

    # AI metnini olası markdown (*) kalıntılarından temizle
    ai_text = ai_text.replace("*", "").replace("#", "")
    print(f"🤖 AI: {ai_text}")

    # 4. Metni Sese Çevir (Edge-TTS)
    print("🔊 Ses sentezleniyor (Edge-TTS)...")
    file_name = f"chat_{uuid.uuid4().hex[:8]}.mp3"
    file_path = os.path.join("static", "audio", file_name)
    
    try:
        communicate = edge_tts.Communicate(ai_text, "en-US-SteffanNeural", rate="+5%")
        await communicate.save(file_path)
    except Exception as e:
        print(f"TTS Hatası: {e}")

    return {
        "user_text": user_text,
        "ai_text": ai_text,
        "audio_url": f"/audio/{file_name}"
    }

# ------------------------------------------
# DİĞER STANDART UÇ NOKTALAR
# ------------------------------------------

@app.post("/api/reading/generate")
async def generate_reading(req: ReadingRequest):
    if not req.topic or not req.topic.strip():
        chosen_topic = random.choice(RANDOM_TOPIC_POOL)
    else:
        chosen_topic = req.topic.strip()

    topic_instruction = f"Topic: {chosen_topic}"
    
    prompt = f"""
    Target Level: {req.level}
    {topic_instruction}
    
    Task:
    1. Generate a HIGH-QUALITY reading passage in English (approx. 500-600 words) strictly about the given topic. Make it fascinating, engaging, and unique.
    2. SENTENCE STRUCTURE: Use a natural mix of short and complex sentences appropriate for the target level.
    3. PARAGRAPH FORMATTING: Divide the text into EXACTLY 5 distinct paragraphs. Use '\\n\\n' to separate them.
    4. Extract EXACTLY 10 CHALLENGING vocabulary words (B2-C1 level) from the passage.
    5. For each extracted word, create a separate standalone "fill in the blank" sentence. The blank MUST be '___'.
    6. Provide EXACTLY 3 options for the blank, the correct_index, and a concise Turkish explanation.
    7. Generate EXACTLY 5 multiple-choice comprehension questions with EXACTLY 4 options each. The questions should be challenging (e.g., testing inference, main idea, and author's purpose), not just simple fact retrieval.
    
    Output STRICTLY in this JSON format:
    {{
      "title": "Your Generated Title Here",
      "passage": "Paragraph 1 text.\\n\\nParagraph 2 text.\\n\\nParagraph 3 text.\\n\\nParagraph 4 text.\\n\\nParagraph 5 text.",
      "fill_in_blanks": [
        {{
          "sentence_with_blank": "The explorer wanted to ___ the ruins.",
          "options": ["discover", "destroy", "ignore"],
          "correct_index": 0,
          "explanation_tr": "Doğru cevap discover. Keşfetmek anlamına gelir."
        }}
      ],
      "questions": [
        {{
          "question": "What is the main idea?",
          "options": ["Opt 1", "Opt 2", "Opt 3", "Opt 4"],
          "correct_index": 0,
          "explanation_tr": "Detaylı Türkçe açıklama."
        }}
      ]
    }}
    """
    result = await get_ai_response(prompt)
    save_to_history("reading", chosen_topic, json.dumps(result), "{}")
    return result

@app.post("/api/writing/evaluate")
async def evaluate_writing(req: WritingRequest):
    chosen_topic = req.topic.strip() if req.topic and req.topic.strip() else random.choice(RANDOM_TOPIC_POOL)
    prompt = f"""
    Topic Given: {chosen_topic}
    Student's Text: "{req.user_text}"
    Analyze the text for grammar, advanced vocabulary usage, and cohesiveness.
    Output STRICTLY in this JSON format:
    {{
      "topic": "{chosen_topic}",
      "overall_score": 85,
      "feedback_tr": "Detaylı ve öğretici genel değerlendirme.",
      "corrections": [
        {{
          "original": "wrong",
          "corrected": "right",
          "explanation_tr": "Hatanın detaylı Türkçe dilbilgisi açıklaması."
        }}
      ],
      "improved_text": "A polished, advanced version of the text."
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
    1. Reformat the text into a highly readable English passage (approx. 500-600 words, using '\\n\\n' for EXACTLY 5 paragraphs).
    2. Replace exactly 10 advanced vocabulary words in the passage with numbered blanks formatted strictly as [1], [2] up to [10].
    3. Provide exactly 3 options and a concise Turkish explanation for each blank ID.
    4. Generate EXACTLY 5 challenging comprehension questions (inference, tone, main idea) with 4 options each.
    
    Output STRICTLY in this JSON format:
    {{
      "title": "A Suitable Title",
      "passage": "Cleaned up text with [1], [2] blanks...\\n\\nNext paragraph...",
      "blanks": [
        {{
          "id": 1,
          "options": ["opt1", "opt2", "opt3"],
          "correct_index": 0,
          "explanation_tr": "Detaylı Türkçe açıklama."
        }}
      ],
      "questions": [
        {{
          "question": "A challenging question?",
          "options": ["A", "B", "C", "D"],
          "correct_index": 0,
          "explanation_tr": "Detaylı açıklama"
        }}
      ]
    }}
    """
    result = await get_ai_response(prompt)
    save_to_history("file", file.filename, json.dumps(result), "{}")
    return result

@app.post("/api/speaking/evaluate")
async def evaluate_speaking(audio: UploadFile = File(...), topic: str = Form("")):
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API yapılandırılmamış.")
        
    try:
        audio_bytes = await audio.read()
        print("🎙️ Groq Whisper'a ses gönderiliyor...")
        
        transcription = groq_client.audio.transcriptions.create(
            file=(audio.filename, audio_bytes),
            model="whisper-large-v3",
            response_format="verbose_json"
        )
        user_text = transcription.text.strip()
        
        if not user_text:
            raise HTTPException(status_code=400, detail="Seste konuşma algılanamadı.")
            
    except Exception as e:
        print(f"Ses işleme hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Ses metne çevrilirken hata oluştu. Detay: {str(e)}")

    print(f"📝 Kullanıcının Söylediği: {user_text}")

    chosen_topic = topic.strip() if topic.strip() else random.choice(RANDOM_TOPIC_POOL)
    
    prompt = f"""
    Topic Context: {chosen_topic}
    Student's Transcribed Speech: "{user_text}"
    
    Task:
    1. Evaluate the speech critically. Point out specific pronunciation or phrasing errors.
    2. Analyze for grammar, advanced vocabulary usage, and natural fluency.
    3. Provide an overall speaking score out of 100.
    
    Output STRICTLY in this JSON format:
    {{
      "topic": "{chosen_topic}",
      "transcribed_text": "{user_text.replace('"', "'")}",
      "overall_score": 85,
      "feedback_tr": "Öğretici ve detaylı değerlendirme.",
      "corrections": [
        {{
          "original": "I sink we should go",
          "corrected": "I think we should go",
          "explanation_tr": "Telaffuz veya gramer kuralı açıklaması."
        }}
      ],
      "improved_text": "A significantly improved, native-sounding version."
    }}
    """
    
    result = await get_ai_response(prompt)
    result["transcribed_text"] = user_text
    
    save_to_history("speaking", chosen_topic, user_text, json.dumps(result))
    return result

@app.post("/api/listening/generate")
async def generate_listening(req: ListeningRequest):
    chosen_topic = req.topic.strip() if req.topic and req.topic.strip() else random.choice(RANDOM_TOPIC_POOL)
    topic_instruction = f"Topic: {chosen_topic}"
    
    if req.speakers == 1:
        format_instruction = "A single narrator giving an advanced monologue."
    elif req.speakers == 2:
        format_instruction = "A sophisticated dialogue between a host and an expert guest."
    else:
        format_instruction = "A high-level podcast panel with host1, host2, and an expert guest."

    prompt = f"""
    Target Level: {req.level} 
    {topic_instruction}
    Format: {format_instruction}
    
    Task:
    1. Write a highly realistic, engaging English audio script (approx. 200-250 words total).
    2. Break it into a JSON array called 'dialogue' with 'speaker' and 'text'.
       - Allowed speakers: "host1", "host2", "guest".
    3. Generate EXACTLY 5 challenging comprehension questions that require deep understanding.
    
    Output STRICTLY in this JSON format:
    {{
      "title": "Title of the Audio Segment",
      "dialogue": [
        {{ "speaker": "host1", "text": "Welcome to the show..." }}
      ],
      "questions": [
        {{
          "question": "What is the implied meaning of the guest's statement?",
          "options": ["A", "B", "C", "D"],
          "correct_index": 0,
          "explanation_tr": "Çıkarımın Türkçe açıklaması."
        }}
      ]
    }}
    """
    
    result = await get_ai_response(prompt)
    
    voice_map = {
        "host1": "en-GB-RyanNeural",
        "host2": "en-GB-SoniaNeural",
        "guest": "en-US-SteffanNeural"
    }
    
    dialogue = result.get("dialogue", [])
    if not dialogue:
        raise HTTPException(status_code=500, detail="Senaryo oluşturulamadı.")

    combined_audio = b""
    for line in dialogue:
        speaker = line.get("speaker", "host1")
        text = line.get("text", "")
        voice = voice_map.get(speaker, "en-US-AriaNeural")
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_path = temp_audio.name
            
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(temp_path)
        
        with open(temp_path, "rb") as f:
            combined_audio += f.read()
            
        os.remove(temp_path)
        
    file_name = f"listening_{uuid.uuid4().hex[:8]}.mp3"
    file_path = os.path.join("static", "audio", file_name)
    
    with open(file_path, "wb") as f:
        f.write(combined_audio)
        
    result["audio_url"] = f"/audio/{file_name}"
    
    full_transcript = "\n".join([f"{d['speaker'].capitalize()}: {d['text']}" for d in dialogue])
    save_to_history("listening", chosen_topic, full_transcript, json.dumps(result))
    
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
    if row[0]:
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

# ==========================================
# 6. KELİME KASASI (VOCABULARY) UÇ NOKTALARI
# ==========================================
@app.post("/api/vocabulary")
async def add_vocabulary(req: VocabularyRequest):
    if not DATABASE_URL: return {"success": False}
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("INSERT INTO vocabulary (word, translation, created_at) VALUES (%s, %s, %s)",
                  (req.word, req.translation, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Veritabanı Kayıt Hatası: {str(e)}")

@app.get("/api/vocabulary")
async def get_vocabulary():
    if not DATABASE_URL: return {"vocabulary": []}
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, word, translation, is_learned FROM vocabulary ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    vocab_list = [{"id": r[0], "word": r[1], "translation": r[2], "is_learned": bool(r[3])} for r in rows]
    return {"vocabulary": vocab_list}

@app.put("/api/vocabulary/{word_id}/status")
async def update_vocab_status(word_id: int, req: VocabStatusRequest):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE vocabulary SET is_learned = %s WHERE id = %s", (req.is_learned, word_id))
    conn.commit()
    conn.close()
    return {"success": True}

@app.delete("/api/vocabulary/{word_id}")
async def delete_vocabulary(word_id: int):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM vocabulary WHERE id = %s", (word_id,))
    conn.commit()
    conn.close()
    return {"success": True}

app.mount("/", StaticFiles(directory="static", html=True), name="static")