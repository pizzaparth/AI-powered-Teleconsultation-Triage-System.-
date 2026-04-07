# AI-powered-Teleconsultation-Triage-System.
### An AI-powered triage and teleconsultation system that prioritizes patients based on severity using voice input and sensor data, generates structured summaries for doctors, and enables efficient rural healthcare delivery with integrated medicine dispensing.  
### HARDWARE + SOFTWARE + AI INTEGRATION
---
## 1. REPOSITORY TREE
## 2. HARDWARE
## 3. SETUP
### 3.1 Installing chocolatey
Inside the terminal:
```console
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```
### 3.2 Installing ffmpeg
Inside the terminal:
```console
choco install ffmpeg
```
### 3.3 Setting up Python virtual environment
Inside the terminal:
```console
cd <project_driectory>
python -m venv venv
pip install -r requirements.txt
```
### 3.4 Launching the API
Inside the terminal:
```console
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
## 4. AI INTEGRATION
### 4.1 Pulling the models
Inside the terminal
```console
ollama pull phi3
ollama pull qwen2.5:7b
```
### 4.2 Setting up OpenAI-Whisper
Inside the terminal
```console
pip install openai-whisper
```
### 4.3 Pipeline of Execution of AI Models one after another
Step 1: Audio of the patient goes into the `speech.py` module which translates the audio to English and return a `transcript.txt`file.

Step 2: The `transcript.txt`, `vitals.json`, `medicines.json` go into the `structured_report_and_queue.py` module which will produce a medical report in the form of `report<number>.json`.

Step 3: The `transcript.txt`, `vitals.json`, `medicines.json` go into the `recommendations.py` module to give a list of recommended medicines for the patient in the form of `recommendations.json`.

## 5. WEBSITE
