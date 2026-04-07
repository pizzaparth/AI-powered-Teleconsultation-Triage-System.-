# api.py
# FastAPI backend for the AI-powered rural healthcare triage platform.
# Run with:  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# Docs at:   http://<your-ip>:8000/docs

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Ensure modules/ is importable regardless of working directory ──────────────
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from speech                      import generate_transcript
from recommendation_model        import generate_recommendations
from structured_report_and_queue import generate_structured_report
from tts                         import generate_tts

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("api.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("api")

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Rural Healthcare Triage API",
    description=(
        "AI-powered triage and teleconsultation platform. "
        "Orchestrates speech transcription, structured report generation, "
        "and medicine recommendations in a single pipeline call."
    ),
    version="1.0.0",
)

# ── File paths (project-relative) ─────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
VITALS_PATH     = BASE_DIR / "vitals.json"
TRANSCRIPT_PATH = BASE_DIR / "transcript.txt"
MEDICINES_PATH  = BASE_DIR / "data" / "medicines.json"
REPORTS_DIR     = BASE_DIR / "reports"
STATIC_DIR      = BASE_DIR / "static"             # serves all WAV files
TTS_OUTPUT_PATH = STATIC_DIR / "patient_summary.wav"

# Ensure output directories exist at startup
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ── Static file serving — WAV reachable at /static/<filename> ─────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Pydantic models ────────────────────────────────────────────────────────────

# ── Default vitals (used when no body is sent) ─────────────────────────────────
DEFAULT_VITALS: dict[str, Any] = {
    "heart_rate"      : 95,
    "blood_pressure"  : "120/80",
    "temperature"     : 37.5,
    "spo2"            : 97,
    "respiratory_rate": 18,
}


class VitalsPayload(BaseModel):
    """
    Hardware sensor vitals submitted by the IoT device.
    All fields are optional — omitted fields fall back to DEFAULT_VITALS.
    Send an empty body `{}` or no body at all to use all defaults.

    Example
    -------
    {
        "vitals": {
            "heart_rate": 95,
            "blood_pressure": "120/80",
            "temperature": 37.5,
            "spo2": 97,
            "respiratory_rate": 18
        }
    }
    """
    vitals: dict[str, Any] = DEFAULT_VITALS


class TriageResponse(BaseModel):
    """Unified response returned after the full pipeline completes."""
    transcript     : str
    report         : dict[str, Any]
    recommendations: dict[str, Any]
    tts_audio_path : str   # absolute path on server
    tts_audio_url  : str   # downloadable URL e.g. http://192.168.1.x:8000/static/patient_summary.wav


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """
    Simple liveness probe.
    Returns {"status": "ok"} when the server is running.
    """
    return {"status": "ok"}


# ── Main triage endpoint ───────────────────────────────────────────────────────

@app.post("/triage", response_model=TriageResponse, tags=["Triage"])
def triage(payload: VitalsPayload, request: Request) -> TriageResponse:
    """
    End-to-end triage pipeline.

    **Steps executed in order:**
    1. Receive vitals from the hardware sensor and persist to `vitals.json`.
    2. Record patient audio via microphone and transcribe with Whisper → `transcript.txt`.
    3. Generate a structured triage report (qwen2.5:7b via Ollama, rule-based fallback).
    4. Generate medicine recommendations (phi3 via Ollama).
    5. Generate TTS as WAV and return a URL to download/stream it.

    **Request body:**
    ```json
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
    """
    log.info("POST /triage — vitals received: %s", payload.vitals)

    # ── Step 1: Persist vitals ─────────────────────────────────────────────────
    try:
        with VITALS_PATH.open("w", encoding="utf-8") as fh:
            json.dump(payload.vitals, fh, indent=2, ensure_ascii=False)
        log.info("Vitals saved → %s", VITALS_PATH)
    except OSError as exc:
        log.error("Failed to save vitals: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Could not write vitals.json: {exc}",
        ) from exc

    # ── Step 2: Record audio and transcribe ────────────────────────────────────
    try:
        log.info("Starting speech recording …")
        transcript_text = generate_transcript(
            wav_path  = BASE_DIR / "recorded_audio.wav",
            txt_path  = TRANSCRIPT_PATH,
            json_path = BASE_DIR / "transcript.json",
        )
        log.info("Transcription complete (%d chars)", len(transcript_text))
    except Exception as exc:
        log.error("Speech pipeline failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Speech pipeline error: {exc}",
        ) from exc

    # ── Step 3: Generate structured report ────────────────────────────────────
    try:
        log.info("Generating structured triage report …")
        report = generate_structured_report(
            transcript_file = str(TRANSCRIPT_PATH),
            vitals_file     = str(VITALS_PATH),
            medicines_file  = str(MEDICINES_PATH),
            report_dir      = str(REPORTS_DIR),
        )
        log.info(
            "Report generated — priority: %s",
            report.get("triage_priority", "N/A").upper(),
        )
    except FileNotFoundError as exc:
        log.error("Report generation — input file missing: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        log.error("Report generation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Report generation error: {exc}",
        ) from exc

    # ── Step 4: Generate medicine recommendations ──────────────────────────────
    try:
        log.info("Generating medicine recommendations …")
        recommendations = generate_recommendations(
            transcript_file = str(TRANSCRIPT_PATH),
            vitals_file     = str(VITALS_PATH),
            medicines_file  = str(MEDICINES_PATH),
            output_file     = str(BASE_DIR / "recommendations.json"),
        )
        log.info(
            "Recommendations generated — %d medicines matched",
            len(recommendations.get("recommendations", [])),
        )
    except FileNotFoundError as exc:
        log.error("Recommendations — input file missing: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        log.error("Recommendations generation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Recommendations error: {exc}",
        ) from exc

    # ── Step 5: Text-to-speech — generate WAV and build download URL ───────────
    tts_audio_path = ""
    tts_audio_url  = ""
    try:
        patient_summary = report.get("patient_summary", "")
        if patient_summary:
            log.info("Generating TTS WAV for patient summary …")
            tts_audio_path = generate_tts(
                text        = patient_summary,
                output_path = str(TTS_OUTPUT_PATH),
                auto_play   = True,   # plays automatically on the server device
            )

            # Build a URL the calling device can use to fetch the WAV file.
            # request.base_url reflects the host:port the client actually hit,
            # so it works on any IP — localhost or LAN.
            base_url      = str(request.base_url).rstrip("/")
            filename      = Path(tts_audio_path).name
            tts_audio_url = f"{base_url}/static/{filename}"

            log.info("TTS complete → %s | URL → %s", tts_audio_path, tts_audio_url)
        else:
            log.warning("No patient_summary found in report — skipping TTS.")
    except Exception as exc:
        # TTS failure must never crash the pipeline
        log.warning("TTS generation failed (non-fatal): %s", exc)

    # ── Step 6: Return unified response ───────────────────────────────────────
    log.info("POST /triage — pipeline complete")
    return TriageResponse(
        transcript      = transcript_text,
        report          = report,
        recommendations = recommendations,
        tts_audio_path  = tts_audio_path,
        tts_audio_url   = tts_audio_url,
    )


# ── Dev entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)