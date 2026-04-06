# modules/recommendation_model.py
# Parth Pancholi — refactored for FastAPI pipeline

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

# ── Ollama config (env-overridable) ────────────────────────────────────────────
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME      = os.getenv("RECOMMENDATION_MODEL", "phi3")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_files(
    transcript_file: str | Path,
    vitals_file: str | Path,
    medicines_file: str | Path,
) -> tuple[str, dict, list[dict]]:
    """Load and validate all three input files."""

    # transcript.txt
    tf = Path(transcript_file)
    if not tf.exists():
        raise FileNotFoundError(f"Transcript not found: {tf}")
    transcript = tf.read_text(encoding="utf-8").strip()
    if not transcript:
        raise ValueError(f"Transcript file is empty: {tf}")

    # vitals.json
    vf = Path(vitals_file)
    if not vf.exists():
        raise FileNotFoundError(f"Vitals file not found: {vf}")
    with vf.open(encoding="utf-8") as fh:
        vitals = json.load(fh)
    if not isinstance(vitals, dict):
        raise ValueError("vitals.json must be a JSON object.")

    # medicines.json
    mf = Path(medicines_file)
    if not mf.exists():
        raise FileNotFoundError(f"Medicines file not found: {mf}")
    with mf.open(encoding="utf-8") as fh:
        medicines = json.load(fh)
    if not isinstance(medicines, list) or len(medicines) == 0:
        raise ValueError("medicines.json must be a non-empty JSON array.")

    log.info(
        "Files loaded → transcript (%d chars) | %d vitals | %d medicines",
        len(transcript), len(vitals), len(medicines),
    )
    return transcript, vitals, medicines


def _build_prompt(transcript: str, vitals: dict, medicines: list[dict]) -> str:
    """Construct the clinical decision-support prompt."""

    vitals_block = "\n".join(f"  {k}: {v}" for k, v in vitals.items())

    medicine_lines = []
    for idx, med in enumerate(medicines, start=1):
        name  = med.get("name", "Unknown")
        syms  = med.get("symptoms", med.get("indications", []))
        dose  = med.get("dosage", "N/A")
        contra = med.get("contraindications", [])
        medicine_lines.append(
            f"  [{idx}] {name}\n"
            f"      Symptoms/Indications : {', '.join(syms)}\n"
            f"      Dosage               : {dose}\n"
            f"      Contraindications    : {', '.join(contra)}"
        )
    medicines_block = "\n".join(medicine_lines)

    scorecard_rows = [
        f'  | {m.get("name",""):<40} | symptoms_match: YES/NO | '
        f'contraindicated: YES/NO | confidence_score: 0.0 |'
        for m in medicines
    ]
    scorecard_template = "\n".join(scorecard_rows)

    json_slots = [
        f'    {{\n'
        f'      "name": "{m.get("name","")}",\n'
        f'      "reason": "<your reasoning for {m.get("name","")}>",\n'
        f'      "confidence_score": 0.0\n'
        f'    }}'
        for m in medicines
    ]
    json_skeleton = ",\n".join(json_slots)

    medicine_names = ", ".join(m.get("name", "") for m in medicines)
    n = len(medicines)

    few_shot = """\
EXAMPLE (format reference only — do not copy values)
══════════════════════════════════════════════════════
Patient: high fever, runny nose, mild headache, slight cough.
Scorecard:
  | MedicineA | symptoms_match: YES | contraindicated: NO  | confidence_score: 0.85 |
  | MedicineB | symptoms_match: YES | contraindicated: NO  | confidence_score: 0.70 |
  | MedicineC | symptoms_match: YES | contraindicated: NO  | confidence_score: 0.60 |
  | MedicineD | symptoms_match: NO  | contraindicated: NO  | confidence_score: 0.00 |
Final JSON:
{
  "recommendations": [
    { "name": "MedicineA", "reason": "Treats fever and headache.", "confidence_score": 0.85 },
    { "name": "MedicineB", "reason": "Addresses runny nose.",      "confidence_score": 0.70 },
    { "name": "MedicineC", "reason": "Suppresses mild cough.",     "confidence_score": 0.60 }
  ]
}
══════════════════════════════════════════════════════"""

    return f"""You are a clinical decision-support assistant acting like a medical vending machine.
Given a patient's transcript and vitals, evaluate EVERY medicine in the inventory
and recommend ALL medicines that are relevant.

{few_shot}

════════════════════════════════════════
SECTION 1 – PATIENT TRANSCRIPT
════════════════════════════════════════
{transcript}

════════════════════════════════════════
SECTION 2 – PATIENT VITALS
════════════════════════════════════════
{vitals_block}

════════════════════════════════════════
SECTION 3 – MEDICINES INVENTORY  ({n} medicines total)
════════════════════════════════════════
{medicines_block}

════════════════════════════════════════
YOUR TASK — follow these steps in order
════════════════════════════════════════

STEP 1 ▸ FILL IN THE SCORECARD (ALL {n} rows required)
{scorecard_template}

STEP 2 ▸ SELECTION RULE
Select every medicine where:
  • confidence_score > 0.0  AND
  • contraindicated = NO (or does not apply to this patient)

STEP 3 ▸ PRODUCE FINAL JSON
• Set confidence_score = 0.0 for non-matching medicines.
• Remove 0.0-score entries from the final output.
• Return ONLY valid JSON — no markdown, no extra text.

ALLOWED MEDICINE NAMES (exact spelling): {medicine_names}

OUTPUT SKELETON — fill this in:
{{
  "recommendations": [
{json_skeleton}
  ]
}}

FINAL REMINDER: return ONLY the JSON object — nothing before or after it.
"""


def _check_ollama() -> None:
    """Verify that the Ollama service is reachable."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        log.info("Ollama service is running at %s", OLLAMA_BASE_URL)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama service is not running. Start it with: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama service timed out.")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not reach Ollama service: {exc}") from exc


def _query_model(prompt: str) -> dict:
    """Send the prompt to Ollama and return the validated recommendations dict."""
    try:
        import ollama as ollama_lib
    except ImportError:
        raise RuntimeError(
            "ollama package not installed. Run: pip install ollama"
        )

    log.info("Querying model '%s' …", MODEL_NAME)
    system_msg = (
        "You are a clinical decision-support assistant. "
        "Respond ONLY with valid JSON matching the exact schema requested. "
        "Do not include markdown, explanations, or text outside the JSON object."
    )

    response = ollama_lib.chat(
        model=MODEL_NAME,
        format="json",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ],
    )

    raw = response["message"]["content"]
    log.debug("Raw model response (first 300 chars): %s", raw[:300])

    result = json.loads(raw)

    if "recommendations" not in result:
        raise ValueError("Model response missing 'recommendations' key.")

    validated = []
    for entry in result["recommendations"]:
        name  = entry.get("name", "").strip()
        reason = entry.get("reason", "").strip()
        try:
            score = float(entry.get("confidence_score", 0.0))
            score = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            score = 0.0
        if name:
            validated.append({
                "name": name,
                "reason": reason,
                "confidence_score": round(score, 4),
            })

    validated.sort(key=lambda x: x["confidence_score"], reverse=True)
    return {"recommendations": validated}


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_recommendations(
    transcript_file: str | Path = "transcript.txt",
    vitals_file: str | Path = "vitals.json",
    medicines_file: str | Path = "data/medicines.json",
    output_file: str | Path = "recommendations.json",
) -> dict[str, Any]:
    """
    Full recommendation pipeline:
      1. Load transcript, vitals, and medicines.
      2. Build clinical prompt.
      3. Query the Ollama phi3 model.
      4. Save and return recommendations.json.

    Parameters
    ----------
    transcript_file : Path to transcript.txt
    vitals_file     : Path to vitals.json
    medicines_file  : Path to medicines.json (static)
    output_file     : Where to save recommendations.json

    Returns
    -------
    dict : { "recommendations": [ { name, reason, confidence_score }, ... ] }
    """
    transcript, vitals, medicines = _load_files(
        transcript_file, vitals_file, medicines_file
    )
    prompt = _build_prompt(transcript, vitals, medicines)
    _check_ollama()
    result = _query_model(prompt)

    out_path = Path(output_file)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    log.info("Recommendations saved → %s  (%d items)", out_path, len(result["recommendations"]))

    return result


# ── Stand-alone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    recs = generate_recommendations()
    print(json.dumps(recs, indent=2))
