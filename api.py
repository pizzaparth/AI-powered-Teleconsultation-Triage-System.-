# api.py
# FastAPI backend for the AI-powered rural healthcare triage platform.
# Run with:  uvicorn api:app --reload
# Docs at:   http://localhost:8000/docs

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
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
TTS_OUTPUT_PATH = BASE_DIR / "patient_summary.mp3"

# Ensure output directories exist at startup
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Pydantic models ────────────────────────────────────────────────────────────

class VitalsPayload(BaseModel):
    """
    Hardware sensor vitals submitted by the IoT device.

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
    vitals: dict[str, Any]


class TriageResponse(BaseModel):
    """Unified response returned after the full pipeline completes."""
    transcript     : str
    report         : dict[str, Any]
    recommendations: dict[str, Any]
    tts_audio_path : str


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
def triage(payload: VitalsPayload) -> TriageResponse:
    """
    End-to-end triage pipeline.

    **Steps executed in order:**
    1. Receive vitals from the hardware sensor and persist to `vitals.json`.
    2. Record patient audio via microphone and transcribe with Whisper → `transcript.txt`.
    3. Generate a structured triage report (qwen2.5:7b via Ollama, rule-based fallback).
    4. Generate medicine recommendations (phi3 via Ollama).
    5. Return transcript, report, and recommendations as a single JSON response.

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

    # ── Step 5: Text-to-speech — speak the patient summary ────────────────────
    tts_audio_path = ""
    try:
        patient_summary = report.get("patient_summary", "")
        if patient_summary:
            log.info("Generating TTS for patient summary …")
            tts_audio_path = generate_tts(
                text        = patient_summary,
                output_path = str(TTS_OUTPUT_PATH),
                auto_play   = True,   # plays automatically on the server device
            )
            log.info("TTS complete → %s", tts_audio_path)
        else:
            log.warning("No patient_summary found in report — skipping TTS.")
    except Exception as exc:
        # TTS failure must never crash the pipeline — report and recs are more critical
        log.warning("TTS generation failed (non-fatal): %s", exc)

    # ── Step 6: Return unified response ───────────────────────────────────────
    log.info("POST /triage — pipeline complete")
    return TriageResponse(
        transcript      = transcript_text,
        report          = report,
        recommendations = recommendations,
        tts_audio_path  = tts_audio_path,
    )


# ── Dev entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)