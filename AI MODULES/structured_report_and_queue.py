# modules/structured_report_and_queue.py
# Parth Pancholi — refactored for FastAPI pipeline

from __future__ import annotations

import glob
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

# ── Ollama config (env-overridable) ────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("REPORT_MODEL",     "qwen2.5:7b")
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT",  "120"))
MAX_RETRIES     = int(os.getenv("MAX_RETRIES",     "3"))
RETRY_DELAY_S   = float(os.getenv("RETRY_DELAY_S", "2.0"))

# ── Validation constants ───────────────────────────────────────────────────────
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


# ── Serial number helper ───────────────────────────────────────────────────────

def _get_next_report_filename(directory: str = ".") -> str:
    """
    Scan *directory* for report_<N>.json files and return the next filename.
    Examples:
      Empty dir          → "report_1.json"
      Has 1, 2, 3        → "report_4.json"
      Has 1, 5, 3 (gaps) → "report_6.json"
    """
    pattern  = os.path.join(directory, "report_*.json")
    existing = glob.glob(pattern)

    max_serial = 0
    for fp in existing:
        m = re.fullmatch(r"report_(\d+)\.json", os.path.basename(fp))
        if m:
            num = int(m.group(1))
            if num > max_serial:
                max_serial = num

    filename = f"report_{max_serial + 1}.json"
    log.info("Serial detection: highest=%d → next='%s'", max_serial, filename)
    return filename


# ── File readers ───────────────────────────────────────────────────────────────

def _read_transcript(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Transcript not found: {p}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Transcript file is empty: {p}")
    log.info("Transcript loaded (%d chars)", len(text))
    return text


def _read_vitals(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Vitals file not found: {p}")
    with p.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("vitals.json must contain a JSON object.")
    missing = REQUIRED_VITALS_FIELDS - set(data.keys())
    if missing:
        log.warning("Vitals missing fields: %s", missing)
    log.info("Vitals loaded: %s", data)
    return data


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(transcript: str, vitals: dict[str, Any]) -> str:
    vitals_str = json.dumps(vitals, indent=2)
    return f"""You are a clinical decision-support AI for a rural telemedicine triage system.
Your ONLY task is to analyse the patient transcript and vitals below and return a
single, valid JSON object — nothing else.

STRICT RULES:
- Return ONLY a JSON object. No markdown, no code fences, no explanation text.
- Every field listed in the schema is REQUIRED.
- "triage_priority" MUST be exactly one of: "normal", "urgent", or "highly_urgent".
- "confidence" MUST be a float between 0.0 and 1.0.

TRIAGE RULES (apply in priority order):
  HIGHLY URGENT → SpO2 < 90, severe breathing difficulty, chest pain + abnormal HR
  URGENT        → Fever > 39°C, heart_rate > 120, persistent vomiting, severe pain
  NORMAL        → Mild symptoms, vitals within normal range

PATIENT TRANSCRIPT:
{transcript}

PATIENT VITALS (JSON):
{vitals_str}

OUTPUT JSON SCHEMA (fill every field):
{{
  "patient_summary"     : "<2-3 sentence clinical summary>",
  "reported_symptoms"   : ["<symptom1>", "<symptom2>"],
  "symptom_duration"    : "<e.g. 2 days>",
  "risk_factors"        : ["<risk1>"],
  "vitals"              : {{
      "heart_rate"       : <number>,
      "blood_pressure"   : "<string, e.g. 130/85>",
      "temperature"      : <number>,
      "spo2"             : <number>,
      "respiratory_rate" : <number>
  }},
  "clinical_flags"      : ["<flag1>"],
  "triage_priority"     : "normal | urgent | highly_urgent",
  "reason_for_priority" : "<clear clinical reasoning>",
  "confidence"          : <float 0.0–1.0>
}}

Return ONLY the completed JSON object now:"""


# ── Ollama helpers ─────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, model: str | None = None) -> str:
    model = model or OLLAMA_MODEL
    url   = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model"  : model,
        "prompt" : prompt,
        "stream" : False,
        "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 1024},
    }
    log.info("Calling Ollama model '%s' …", model)
    response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()
    raw: str = response.json().get("response", "").strip()
    log.debug("Raw response (first 300 chars): %s", raw[:300])
    return raw


def _extract_json(text: str) -> dict[str, Any]:
    """Three-strategy JSON extraction: direct → strip fences → brace matching."""
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

    # Strategy 3 — find first balanced { ... } block
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
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError("No valid JSON object found in model response.")


def _validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field, expected in REQUIRED_FIELDS.items():
        if field not in report:
            errors.append(f"Missing field: '{field}'")
            continue
        if not isinstance(report[field], expected):
            errors.append(
                f"Wrong type for '{field}': expected {expected}, "
                f"got {type(report[field]).__name__}"
            )

    if isinstance(report.get("vitals"), dict):
        for f in REQUIRED_VITALS_FIELDS - set(report["vitals"].keys()):
            errors.append(f"Missing vitals sub-field: '{f}'")

    if report.get("triage_priority") not in VALID_PRIORITIES:
        errors.append(
            f"Invalid triage_priority '{report.get('triage_priority')}'. "
            f"Must be one of {VALID_PRIORITIES}."
        )

    conf = report.get("confidence")
    if isinstance(conf, (int, float)) and not (0.0 <= conf <= 1.0):
        errors.append(f"'confidence' {conf} out of range [0.0, 1.0].")

    return errors


# ── Rule-based fallback ────────────────────────────────────────────────────────

def _rule_based_triage(transcript: str, vitals: dict[str, Any]) -> dict[str, Any]:
    """Deterministic triage when Ollama is unavailable."""
    log.warning("Falling back to rule-based triage engine.")
    tl = transcript.lower()

    hr   = float(vitals.get("heart_rate",       0))
    temp = float(vitals.get("temperature",      0))
    spo2 = float(vitals.get("spo2",           100))
    rr   = float(vitals.get("respiratory_rate", 0))

    breathing_kws   = {"breathing difficulty", "shortness of breath", "can't breathe",
                       "cannot breathe", "breathless", "dyspnoea", "dyspnea"}
    chest_pain_kws  = {"chest pain", "chest tightness", "chest pressure", "angina"}
    vomit_kws       = {"vomit", "vomiting", "persistent vomit"}
    severe_pain_kws = {"severe pain", "unbearable pain", "excruciating", "intense pain"}

    def any_kw(kw_set: set[str]) -> bool:
        return any(kw in tl for kw in kw_set)

    symptom_map = {
        "fever"              : ["fever", "temperature", "hot"],
        "headache"           : ["headache", "head pain", "migraine"],
        "weakness"           : ["weak", "fatigue", "tired"],
        "cough"              : ["cough"],
        "chest pain"         : list(chest_pain_kws),
        "breathing difficulty": list(breathing_kws),
        "vomiting"           : list(vomit_kws),
        "severe pain"        : list(severe_pain_kws),
        "dizziness"          : ["dizzy", "dizziness", "lightheaded"],
    }
    detected = [sym for sym, kws in symptom_map.items() if any(kw in tl for kw in kws)]

    dur_m = re.search(r"(\d+\s*(?:day|days|hour|hours|week|weeks|month|months))", tl)
    duration = dur_m.group(1) if dur_m else "unknown"

    clinical_flags: list[str] = []
    if spo2 < 90:   clinical_flags.append(f"Critical SpO2: {spo2}%")
    elif spo2 < 95: clinical_flags.append(f"Low SpO2: {spo2}%")
    if hr > 120:    clinical_flags.append(f"Tachycardia: HR {hr} bpm")
    elif hr > 100:  clinical_flags.append(f"Elevated HR: {hr} bpm")
    if temp > 39:   clinical_flags.append(f"High fever: {temp}°C")
    elif temp > 37.5: clinical_flags.append(f"Low-grade fever: {temp}°C")
    if rr > 25:     clinical_flags.append(f"Tachypnoea: RR {rr}/min")

    if spo2 < 90:
        priority, reason, confidence = (
            "highly_urgent",
            f"SpO2 critically low at {spo2}% — immediate intervention required.",
            0.97,
        )
    elif any_kw(breathing_kws) and rr > 25:
        priority, reason, confidence = (
            "highly_urgent",
            "Severe breathing difficulty with elevated respiratory rate.",
            0.93,
        )
    elif any_kw(chest_pain_kws) and hr > 100:
        priority, reason, confidence = (
            "highly_urgent",
            f"Chest pain + elevated HR ({hr} bpm) — possible cardiac event.",
            0.91,
        )
    elif temp > 39:
        priority, reason, confidence = (
            "urgent",
            f"High fever ({temp}°C) exceeds urgent threshold.",
            0.88,
        )
    elif hr > 120:
        priority, reason, confidence = (
            "urgent",
            f"Significant tachycardia (HR {hr} bpm).",
            0.85,
        )
    elif any_kw(vomit_kws):
        priority, reason, confidence = (
            "urgent",
            "Persistent vomiting — risk of dehydration.",
            0.80,
        )
    elif any_kw(severe_pain_kws):
        priority, reason, confidence = (
            "urgent",
            "Severe pain requiring prompt assessment.",
            0.82,
        )
    else:
        priority, reason, confidence = (
            "normal",
            "Mild symptoms; vitals within acceptable range. Routine assessment.",
            0.75,
        )

    summary = (
        f"Rule-based assessment (AI model unavailable). "
        f"Patient presents with {', '.join(detected) or 'non-specific symptoms'} "
        f"for approximately {duration}. "
        f"Vitals: HR {hr} bpm, Temp {temp}°C, SpO2 {spo2}%."
    )

    return {
        "patient_summary"     : summary,
        "reported_symptoms"   : detected,
        "symptom_duration"    : duration,
        "risk_factors"        : [],
        "vitals"              : {
            "heart_rate"       : vitals.get("heart_rate",       0),
            "blood_pressure"   : vitals.get("blood_pressure",   ""),
            "temperature"      : vitals.get("temperature",      0),
            "spo2"             : vitals.get("spo2",            100),
            "respiratory_rate" : vitals.get("respiratory_rate", 0),
        },
        "clinical_flags"      : clinical_flags,
        "triage_priority"     : priority,
        "reason_for_priority" : reason,
        "confidence"          : confidence,
        "_generated_by"       : "rule_based_fallback",
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_structured_report(
    transcript_file: str | Path = "transcript.txt",
    vitals_file: str | Path = "vitals.json",
    medicines_file: str | Path = "data/medicines.json",   # accepted but unused
    report_dir: str | Path = "reports",
    model: str | None = None,
) -> dict[str, Any]:
    """
    Full structured report pipeline:
      1. Load transcript and vitals.
      2. Build clinical prompt.
      3. Query Ollama (qwen2.5:7b) with retries; fall back to rule-based triage.
      4. Save report_<serial>.json to *report_dir*.
      5. Return the report dict.

    Parameters
    ----------
    transcript_file : Path to transcript.txt
    vitals_file     : Path to vitals.json
    medicines_file  : Accepted for API compatibility (not used in report generation)
    report_dir      : Directory where report_<N>.json files are saved
    model           : Override the Ollama model name

    Returns
    -------
    dict : Structured triage report object
    """
    log.info("=" * 60)
    log.info("Starting structured report generation")
    log.info("=" * 60)

    transcript = _read_transcript(transcript_file)
    vitals     = _read_vitals(vitals_file)
    prompt     = _build_prompt(transcript, vitals)

    report: dict[str, Any] | None = None
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        log.info("Ollama attempt %d / %d", attempt, MAX_RETRIES)
        try:
            raw    = _call_ollama(prompt, model=model)
            parsed = _extract_json(raw)
            errors = _validate_report(parsed)

            if errors:
                last_error = "; ".join(errors)
                log.warning("Validation failed (attempt %d): %s", attempt, last_error)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_S)
                continue

            report = parsed
            log.info("Valid report received on attempt %d.", attempt)
            break

        except requests.exceptions.ConnectionError:
            last_error = "Cannot connect to Ollama service."
            log.error("Attempt %d — %s", attempt, last_error)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)

        except requests.exceptions.Timeout:
            last_error = f"Ollama timed out after {OLLAMA_TIMEOUT}s."
            log.error("Attempt %d — %s", attempt, last_error)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)

        except (requests.exceptions.HTTPError, ValueError) as exc:
            last_error = str(exc)
            log.warning("Attempt %d — error: %s", attempt, last_error)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)

    if report is None:
        log.error("All %d attempts failed (%s). Using rule-based fallback.", MAX_RETRIES, last_error)
        report = _rule_based_triage(transcript, vitals)

    # ── Save to report_<N>.json ────────────────────────────────────────────────
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename   = _get_next_report_filename(str(out_dir))
    out_path   = out_dir / filename

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    log.info("Report saved → %s", out_path.resolve())
    log.info("Triage priority : %s", report.get("triage_priority", "N/A").upper())
    log.info("Confidence      : %.2f", report.get("confidence", 0.0))
    log.info("=" * 60)

    return report


# ── Stand-alone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate structured triage report.")
    parser.add_argument("--transcript", default="transcript.txt")
    parser.add_argument("--vitals",     default="vitals.json")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--model",      default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    report = generate_structured_report(
        transcript_file=args.transcript,
        vitals_file=args.vitals,
        report_dir=args.report_dir,
        model=args.model,
    )
    print(json.dumps(report, indent=2))
