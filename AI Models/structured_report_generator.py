# Made by Parth Pancholi

from __future__ import annotations

import glob
import json
import logging
import os
import re
import sys
import time
from typing import Any

import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("report_generator.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL",    "qwen2.5:7b")
OLLAMA_TIMEOUT    = int(os.getenv("OLLAMA_TIMEOUT", "120"))   # seconds
MAX_RETRIES       = int(os.getenv("MAX_RETRIES",   "3"))
RETRY_DELAY_S     = float(os.getenv("RETRY_DELAY_S", "2.0"))

TRANSCRIPT_FILE   = os.getenv("TRANSCRIPT_FILE", "transcript.txt")
VITALS_FILE       = os.getenv("VITALS_FILE",     "vitals.json")
REPORT_DIR        = os.getenv("REPORT_DIR",      ".")           # output directory


REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "patient_summary"     : str,
    "reported_symptoms"   : list,
    "symptom_duration"    : str,
    "risk_factors"        : list,
    "vitals"              : dict,
    "clinical_flags"      : list,
    "triage_priority"     : str,
    "reason_for_priority" : str,
    "confidence"          : (int, float),
}

REQUIRED_VITALS_FIELDS = {
    "heart_rate", "blood_pressure", "temperature", "spo2", "respiratory_rate"
}

VALID_PRIORITIES = {"normal", "urgent", "highly_urgent"}




def get_next_report_filename(directory: str = ".") -> str:
    """
    Scan *directory* for files matching report_<number>.json,
    find the highest existing serial number, and return the next filename.

    Examples
    --------
    Directory empty           → "report_1.json"
    Contains report_1, 2, 3  → "report_4.json"
    Contains report_1, 5, 3  → "report_6.json"  (gaps are skipped safely)
    """
    pattern = os.path.join(directory, "report_*.json")
    existing = glob.glob(pattern)

    max_serial = 0
    for filepath in existing:
        basename = os.path.basename(filepath)
        match = re.fullmatch(r"report_(\d+)\.json", basename)
        if match:
            num = int(match.group(1))
            if num > max_serial:
                max_serial = num

    next_serial = max_serial + 1
    filename = f"report_{next_serial}.json"
    log.info("Serial detection: highest=%d → next file='%s'", max_serial, filename)
    return filename




def read_transcript(path: str) -> str:
    """Read and return the patient transcript, stripping excess whitespace."""
    log.info("Reading transcript from '%s'", path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Transcript file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        text = fh.read().strip()
    if not text:
        raise ValueError(f"Transcript file is empty: {path}")
    log.info("Transcript loaded (%d chars)", len(text))
    return text


def read_vitals(path: str) -> dict[str, Any]:
    """Load and minimally validate the vitals JSON file."""
    log.info("Reading vitals from '%s'", path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Vitals file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    missing = REQUIRED_VITALS_FIELDS - set(data.keys())
    if missing:
        log.warning("Vitals JSON is missing fields: %s", missing)
    log.info("Vitals loaded: %s", data)
    return data




def build_prompt(transcript: str, vitals: dict[str, Any]) -> str:
    """
    Construct the strict JSON-forcing prompt sent to qwen2.5:7b.
    The instruction block makes it clear that the model MUST return only
    a single JSON object — no markdown, no prose.
    """
    vitals_str = json.dumps(vitals, indent=2)

    prompt = f"""You are a clinical decision-support AI for a rural telemedicine triage system.
Your ONLY task is to analyse the patient transcript and vitals below and return a
single, valid JSON object — nothing else.

STRICT RULES:
- Return ONLY a JSON object. No markdown, no code fences, no explanation text.
- Do NOT wrap the JSON in ```json ... ``` or any other delimiters.
- Every field listed in the schema is REQUIRED.
- "triage_priority" MUST be exactly one of: "normal", "urgent", or "highly_urgent".
- "confidence" MUST be a float between 0.0 and 1.0.

TRIAGE RULES (apply these in priority order):
  HIGHLY URGENT → SpO2 < 90, severe breathing difficulty, chest pain with abnormal heart rate
  URGENT        → Fever > 39 °C, heart_rate > 120, persistent vomiting, severe pain
  NORMAL        → Mild symptoms, vitals within normal range

PATIENT TRANSCRIPT:
{transcript}

PATIENT VITALS (JSON):
{vitals_str}

OUTPUT JSON SCHEMA (fill every field):
{{
  "patient_summary"     : "<2-3 sentence clinical summary>",
  "reported_symptoms"   : ["<symptom1>", "<symptom2>", "..."],
  "symptom_duration"    : "<e.g. 2 days>",
  "risk_factors"        : ["<risk1>", "..."],
  "vitals"              : {{
      "heart_rate"       : <number>,
      "blood_pressure"   : "<string, e.g. 130/85>",
      "temperature"      : <number>,
      "spo2"             : <number>,
      "respiratory_rate" : <number>
  }},
  "clinical_flags"      : ["<flag1>", "..."],
  "triage_priority"     : "normal | urgent | highly_urgent",
  "reason_for_priority" : "<clear clinical reasoning>",
  "confidence"          : <float 0.0–1.0>
}}

Return ONLY the completed JSON object now:"""

    return prompt




def call_ollama(prompt: str, model: str | None = None) -> str:
    model = model or OLLAMA_MODEL
    
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model"  : model,
        "prompt" : prompt,
        "stream" : False,
        "options": {
            "temperature" : 0.1,   # near-deterministic for clinical output
            "top_p"       : 0.9,
            "num_predict" : 1024,
        },
    }

    log.info("Calling Ollama model '%s' at %s", model, url)
    response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()

    result = response.json()
    raw_text: str = result.get("response", "").strip()
    log.debug("Raw model response (first 300 chars): %s", raw_text[:300])
    return raw_text



def extract_json_from_text(text: str) -> dict[str, Any]:
    
    # Strategy 1 — direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2 — strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3 — find first balanced JSON object via brace matching
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError("No valid JSON object found in model response.")


def validate_report(report: dict[str, Any]) -> list[str]:
   
    errors: list[str] = []

    # Top-level field presence & types
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in report:
            errors.append(f"Missing required field: '{field}'")
            continue
        value = report[field]
        if not isinstance(value, expected_type):
            errors.append(
                f"Field '{field}' has wrong type "
                f"(expected {expected_type}, got {type(value).__name__})"
            )

    # Vitals sub-fields
    if "vitals" in report and isinstance(report["vitals"], dict):
        missing_v = REQUIRED_VITALS_FIELDS - set(report["vitals"].keys())
        for f in missing_v:
            errors.append(f"Missing vitals sub-field: '{f}'")

    # triage_priority enum
    if "triage_priority" in report:
        if report["triage_priority"] not in VALID_PRIORITIES:
            errors.append(
                f"Invalid triage_priority '{report['triage_priority']}'. "
                f"Must be one of {VALID_PRIORITIES}."
            )

    # confidence range
    if "confidence" in report and isinstance(report["confidence"], (int, float)):
        if not (0.0 <= report["confidence"] <= 1.0):
            errors.append(
                f"'confidence' value {report['confidence']} is out of range [0.0, 1.0]."
            )

    return errors



def rule_based_triage(transcript: str, vitals: dict[str, Any]) -> dict[str, Any]:
    log.warning("Falling back to rule-based triage engine.")

    transcript_lower = transcript.lower()

    # Extract numeric vitals safely
    hr    = float(vitals.get("heart_rate",       0))
    temp  = float(vitals.get("temperature",      0))
    spo2  = float(vitals.get("spo2",           100))
    rr    = float(vitals.get("respiratory_rate", 0))
    bp    = str(vitals.get("blood_pressure",    ""))

    # Keyword sets
    breathing_kws   = {"breathing difficulty", "shortness of breath", "can't breathe",
                       "cannot breathe", "breathless", "dyspnoea", "dyspnea"}
    chest_pain_kws  = {"chest pain", "chest tightness", "chest pressure", "angina"}
    vomit_kws       = {"vomit", "vomiting", "nausea and vomit", "persistent vomit"}
    severe_pain_kws = {"severe pain", "unbearable pain", "excruciating", "intense pain"}

    def any_kw(kw_set: set[str]) -> bool:
        return any(kw in transcript_lower for kw in kw_set)

    # Symptom extraction (simple keyword scan)
    symptom_map = {
        "fever"              : ["fever", "temperature", "hot"],
        "headache"           : ["headache", "head pain", "migraine"],
        "weakness"           : ["weak", "fatigue", "tired", "lethargy"],
        "cough"              : ["cough"],
        "chest pain"         : list(chest_pain_kws),
        "breathing difficulty": list(breathing_kws),
        "vomiting"           : list(vomit_kws),
        "severe pain"        : list(severe_pain_kws),
        "dizziness"          : ["dizzy", "dizziness", "lightheaded"],
    }
    detected_symptoms = [
        sym for sym, kws in symptom_map.items()
        if any(kw in transcript_lower for kw in kws)
    ]

    # Duration extraction (simple regex)
    duration_match = re.search(
        r"(\d+\s*(?:day|days|hour|hours|week|weeks|month|months))", transcript_lower
    )
    duration = duration_match.group(1) if duration_match else "unknown"

    # Clinical flags
    clinical_flags: list[str] = []
    if spo2 < 90:
        clinical_flags.append(f"Critical SpO2: {spo2}%")
    elif spo2 < 95:
        clinical_flags.append(f"Low SpO2: {spo2}%")
    if hr > 120:
        clinical_flags.append(f"Tachycardia: HR {hr} bpm")
    elif hr > 100:
        clinical_flags.append(f"Elevated HR: {hr} bpm")
    if temp > 39:
        clinical_flags.append(f"High fever: {temp}°C")
    elif temp > 37.5:
        clinical_flags.append(f"Low-grade fever: {temp}°C")
    if rr > 25:
        clinical_flags.append(f"Tachypnoea: RR {rr}/min")

    # Priority decision tree
    priority: str
    reason: str
    confidence: float

    # HIGHLY URGENT
    if spo2 < 90:
        priority   = "highly_urgent"
        reason     = f"SpO2 critically low at {spo2}% — indicates hypoxaemia requiring immediate intervention."
        confidence = 0.97
    elif any_kw(breathing_kws) and rr > 25:
        priority   = "highly_urgent"
        reason     = "Severe breathing difficulty with elevated respiratory rate detected."
        confidence = 0.93
    elif any_kw(chest_pain_kws) and hr > 100:
        priority   = "highly_urgent"
        reason     = f"Chest pain combined with elevated heart rate ({hr} bpm) suggests possible cardiac event."
        confidence = 0.91
    # URGENT
    elif temp > 39:
        priority   = "urgent"
        reason     = f"High fever ({temp}°C) exceeds urgent threshold of 39°C."
        confidence = 0.88
    elif hr > 120:
        priority   = "urgent"
        reason     = f"Significant tachycardia (HR {hr} bpm) requires prompt evaluation."
        confidence = 0.85
    elif any_kw(vomit_kws):
        priority   = "urgent"
        reason     = "Persistent vomiting detected — risk of dehydration and electrolyte imbalance."
        confidence = 0.80
    elif any_kw(severe_pain_kws):
        priority   = "urgent"
        reason     = "Severe pain reported by patient requiring prompt pain management and assessment."
        confidence = 0.82
    # NORMAL
    else:
        priority   = "normal"
        reason     = "Symptoms are mild and vitals are within acceptable ranges. Routine assessment recommended."
        confidence = 0.75

    summary = (
        f"Rule-based assessment (AI model unavailable). "
        f"Patient presents with {', '.join(detected_symptoms) or 'non-specific symptoms'} "
        f"for approximately {duration}. "
        f"Vitals: HR {hr} bpm, Temp {temp}°C, SpO2 {spo2}%."
    )

    return {
        "patient_summary"     : summary,
        "reported_symptoms"   : detected_symptoms,
        "symptom_duration"    : duration,
        "risk_factors"        : [],
        "vitals"              : {
            "heart_rate"       : vitals.get("heart_rate",       0),
            "blood_pressure"   : vitals.get("blood_pressure",   ""),
            "temperature"      : vitals.get("temperature",      0),
            "spo2"             : vitals.get("spo2",             100),
            "respiratory_rate" : vitals.get("respiratory_rate", 0),
        },
        "clinical_flags"      : clinical_flags,
        "triage_priority"     : priority,
        "reason_for_priority" : reason,
        "confidence"          : confidence,
        "_generated_by"       : "rule_based_fallback",   # metadata tag
    }



def generate_report(
    transcript_path: str = TRANSCRIPT_FILE,
    vitals_path    : str = VITALS_FILE,
    report_dir     : str = REPORT_DIR,
    model          : str | None = None,
) -> str:
    
    # -- 1. Load inputs -------------------------------------------------------
    log.info("=" * 60)
    log.info("Starting structured report generation")
    log.info("=" * 60)

    transcript = read_transcript(transcript_path)
    vitals     = read_vitals(vitals_path)

    # -- 2. Build prompt ------------------------------------------------------
    prompt = build_prompt(transcript, vitals)
    log.debug("Prompt length: %d characters", len(prompt))

    # -- 3 & 4. Call model with retries & validate ----------------------------
    report: dict[str, Any] | None = None
    last_error: str = ""

    for attempt in range(1, MAX_RETRIES + 1):
        log.info("Ollama attempt %d / %d", attempt, MAX_RETRIES)
        try:
            raw_output = call_ollama(prompt, model=model)
            parsed     = extract_json_from_text(raw_output)
            errors     = validate_report(parsed)

            if errors:
                last_error = "; ".join(errors)
                log.warning("Validation failed (attempt %d): %s", attempt, last_error)
                if attempt < MAX_RETRIES:
                    log.info("Retrying in %.1f s …", RETRY_DELAY_S)
                    time.sleep(RETRY_DELAY_S)
                continue

            report = parsed
            log.info("Model returned valid report on attempt %d.", attempt)
            break

        except requests.exceptions.ConnectionError:
            last_error = "Cannot connect to Ollama service."
            log.error("Attempt %d — %s", attempt, last_error)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)

        except requests.exceptions.Timeout:
            last_error = f"Ollama request timed out after {OLLAMA_TIMEOUT}s."
            log.error("Attempt %d — %s", attempt, last_error)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)

        except requests.exceptions.HTTPError as exc:
            last_error = f"HTTP error from Ollama: {exc}"
            log.error("Attempt %d — %s", attempt, last_error)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)

        except ValueError as exc:
            last_error = str(exc)
            log.warning("Attempt %d — JSON parse error: %s", attempt, last_error)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)

    
    if report is None:
        log.error(
            "All %d Ollama attempts failed. Last error: %s", MAX_RETRIES, last_error
        )
        log.warning("Activating rule-based fallback triage.")
        report = rule_based_triage(transcript, vitals)

    
    filename    = get_next_report_filename(report_dir)
    output_path = os.path.join(report_dir, filename)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    log.info("Report saved → %s", os.path.abspath(output_path))
    log.info("Triage priority : %s", report.get("triage_priority", "N/A").upper())
    log.info("Confidence      : %.2f", report.get("confidence", 0.0))
    log.info("=" * 60)

    return os.path.abspath(output_path)



def main() -> None:
    """Entry point when executed as a script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a structured triage report from transcript + vitals."
    )
    parser.add_argument(
        "--transcript", default=TRANSCRIPT_FILE,
        help=f"Path to transcript text file (default: {TRANSCRIPT_FILE})"
    )
    parser.add_argument(
        "--vitals", default=VITALS_FILE,
        help=f"Path to vitals JSON file (default: {VITALS_FILE})"
    )
    parser.add_argument(
        "--report-dir", default=REPORT_DIR,
        help=f"Directory to save report files (default: {REPORT_DIR})"
    )
    parser.add_argument(
        "--model", default=OLLAMA_MODEL,
        help=f"Ollama model name (default: {OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level logging"
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        saved_path = generate_report(
            transcript_path = args.transcript,
            vitals_path     = args.vitals,
            report_dir      = args.report_dir,
            model           = args.model,
        )
        print(f"\n✅  Report generated: {saved_path}")
    except FileNotFoundError as exc:
        log.error("Input file error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.exception("Unexpected error during report generation: %s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
