<<<<<<< HEAD
# aura-english
A web site for learning english
=======
# LingoSpace - English Practice Platform

Kız arkadaşın için hazırladığın özel İngilizce pratik platformu!

## Özellikler
- **Modern Arayüz:** Vue 3 ve TailwindCSS ile tasarlandı, yumuşak (gül/mor) tonlarında, göze hitap eden ve mobilde bile harika çalışan bir UI.
- **Akıllı API Geçişi (Fallback):** Google Gemini 1.5 Flash kullanır. Eğer ücretsiz kotası biterse hiçbir hata vermeden otomatik olarak Groq (Llama 3) API'sine geçer!
- **Dahili Veritabanı:** Yapılan tüm testler ve yazma pratikleri anında SQLite veritabanına kaydedilir, 'Geçmiş' sekmesinden bakılabilir.

## Nasıl Kurulur ve Çalıştırılır?

1. Gerekli kütüphaneleri yükleyin:
   `pip install -r requirements.txt`

2. API Anahtarlarını Alın ve Ekleyin:
   - Google AI Studio'dan (Gemini) API anahtarı alın.
   - Groq Cloud'dan (Llama 3 için) API anahtarı alın.
   - `main.py` dosyasını açın ve en üstte bulunan `GEMINI_API_KEY` ve `GROQ_API_KEY` değişkenlerine bu şifreleri yapıştırın.

3. Sunucuyu Başlatın:
   `python main.py`
   (Veya `uvicorn main:app --reload` komutunu kullanabilirsiniz.)

4. Tarayıcınızdan http://localhost:8000 adresine gidin. Platform hazır!

## Ücretsiz Canlıya Alma (Deployment)
Bu projeyi internette yayınlamak istersen **Render.com** üzerinden sadece GitHub deponu bağlayıp `uvicorn main:app --host 0.0.0.0 --port $PORT` komutuyla tamamen ücretsiz bir şekilde canlıya alabilirsin!
>>>>>>> e84af68 (AuraEnglish first launch)
