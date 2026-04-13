# AI-powered-Teleconsultation-Triage-System.
### An AI-powered triage and teleconsultation system that prioritizes patients based on severity using voice input and sensor data, generates structured summaries for doctors, and enables efficient rural healthcare delivery with integrated medicine dispensing.  
### AI INTEGRATION PART ONLY
---

## 1. SETUP
### 1.1 Installing chocolatey
Inside the terminal:
```console
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```
### 1.2 Installing ffmpeg
Inside the terminal:
```console
choco install ffmpeg
```
### 1.3 Setting up Python virtual environment
Inside the terminal:
```console
cd <project_driectory>
python -m venv venv
pip install -r requirements.txt
```
### 1.4 Launching the API
Inside the terminal:
```console
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
## 2. AI INTEGRATION
### 2.1 Pulling the models
Inside the terminal
```console
ollama pull phi3
ollama pull qwen2.5:7b
```
### 2.2 Setting up OpenAI-Whisper
Inside the terminal
```console
pip install openai-whisper
```
### 2.3 Pipeline of Execution of AI Models one after another
Step 1: Audio of the patient goes into the `speech.py` module which translates the audio to English and return a `transcript.txt`file.

Step 2: The `transcript.txt`, `vitals.json`, `medicines.json` go into the `structured_report_and_queue.py` module which will produce a medical report in the form of `report<number>.json`.

Step 3: The `transcript.txt`, `vitals.json`, `medicines.json` go into the `recommendations.py` module to give a list of recommended medicines for the patient in the form of `recommendations.json`.

## 3. API REFERENCE

Base URL: `http://<your-ip>:8000`

| Method | Endpoint  | Description                        |
|--------|-----------|------------------------------------|
| GET    | `/health` | Liveness probe — returns `{"status": "ok"}` |
| POST   | `/triage` | Full triage pipeline — accepts vitals JSON, returns transcript, report, recommendations, and WAV URL |
| GET    | `/static/patient_summary.wav` | Stream the latest TTS audio summary |
| GET    | `/docs`   | Interactive Swagger UI             |

**Example request:**
```json
POST /triage
{
  "vitals": {
    "heart_rate": 95,
    "blood_pressure": "120/80",
    "temperature": 37.5,
    "spo2": 97,
    "respiratory_rate": 18
  }
}
```

**Example response:**
```json
{
  "transcript": "Patient reports headache and mild fever for two days...",
  "report": {
    "triage_priority": "medium",
    "patient_summary": "...",
    "diagnosis": "..."
  },
  "recommendations": {
    "recommendations": [...]
  },
  "tts_audio_path": "/absolute/path/static/patient_summary.wav",
  "tts_audio_url": "http://192.168.1.x:8000/static/patient_summary.wav"
}
```

---

