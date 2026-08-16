import os
import requests
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip("[]'\"")

if not GEMINI_API_KEY:
    print("API Anahtarı bulunamadı! .env dosyasını kontrol et.")
else:
    print("Google sunucularına bağlanılıyor...\n")
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        models = response.json().get("models", [])
        print("🎉 SENİN API ANAHTARINLA KULLANABİLECEĞİN MODELLER:")
        print("-" * 50)
        for m in models:
            # Sadece metin üretimi (generateContent) destekleyen modelleri listele
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print(f"✅ Model Adı: {m['name'].replace('models/', '')}")
        print("-" * 50)
    else:
        print(f"Hata Kodu: {response.status_code}")
        print(f"Detay: {response.text}")
        