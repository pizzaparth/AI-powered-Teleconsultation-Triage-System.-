# Made by Parth Pancholi

import json
import sys
import os
import requests

# Ollama Python library
try:
    import ollama
except ImportError:
    print("Error: 'ollama' package not found. Install it with: pip install ollama")
    sys.exit(1)


# Constants
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT_FILE  = os.path.join(SCRIPT_DIR, "transcript.txt")
VITALS_FILE      = os.path.join(SCRIPT_DIR, "vitals.json")
MEDICINES_FILE   = os.path.join(SCRIPT_DIR, "medicines.json")
OUTPUT_FILE      = os.path.join(SCRIPT_DIR, "recommendations.json")
MODEL_NAME       = "phi3"
OLLAMA_BASE_URL  = "http://localhost:11434"


# File Loading 
def load_files() -> tuple[str, dict, list[dict]]:
    """
    Load and return:
      - transcript (str)        from transcript.txt
      - vitals     (dict)       from vitals.json
      - medicines  (list[dict]) from medicines.json

    Exits with a clear message if any file is missing or contains invalid JSON.
    """

    # --- transcript.txt ---
    if not os.path.exists(TRANSCRIPT_FILE):
        print(f"Error: Missing file → {TRANSCRIPT_FILE}")
        sys.exit(1)
    try:
        with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as fh:
            transcript = fh.read().strip()
        if not transcript:
            print("Error: transcript.txt is empty.")
            sys.exit(1)
    except OSError as exc:
        print(f"Error reading transcript.txt: {exc}")
        sys.exit(1)

    # --- vitals.json ---
    if not os.path.exists(VITALS_FILE):
        print(f"Error: Missing file → {VITALS_FILE}")
        sys.exit(1)
    try:
        with open(VITALS_FILE, "r", encoding="utf-8") as fh:
            vitals = json.load(fh)
        if not isinstance(vitals, dict):
            print("Error: vitals.json must contain a JSON object (dict).")
            sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Error: vitals.json contains invalid JSON → {exc}")
        sys.exit(1)
    except OSError as exc:
        print(f"Error reading vitals.json: {exc}")
        sys.exit(1)

    # --- medicines.json ---
    if not os.path.exists(MEDICINES_FILE):
        print(f"Error: Missing file → {MEDICINES_FILE}")
        sys.exit(1)
    try:
        with open(MEDICINES_FILE, "r", encoding="utf-8") as fh:
            medicines = json.load(fh)
        if not isinstance(medicines, list) or len(medicines) == 0:
            print("Error: medicines.json must contain a non-empty JSON array.")
            sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Error: medicines.json contains invalid JSON → {exc}")
        sys.exit(1)
    except OSError as exc:
        print(f"Error reading medicines.json: {exc}")
        sys.exit(1)

    print(f"[✓] Files loaded  →  transcript ({len(transcript)} chars) | "
          f"{len(vitals)} vital signs | {len(medicines)} medicines")
    return transcript, vitals, medicines


# Prompt Construction
def build_prompt(transcript: str, vitals: dict, medicines: list[dict]) -> str:

    # Vitals block
    vitals_block = "\n".join(f"  {k}: {v}" for k, v in vitals.items())

    # Per-medicine inventory block
    medicine_lines = []
    for idx, med in enumerate(medicines, start=1):
        name              = med.get("name", "Unknown")
        symptoms          = med.get("symptoms", med.get("indications", []))
        dosage            = med.get("dosage", "N/A")
        contraindications = med.get("contraindications", [])
        medicine_lines.append(
            f"  [{idx}] {name}\n"
            f"      Symptoms/Indications : {', '.join(symptoms)}\n"
            f"      Dosage               : {dosage}\n"
            f"      Contraindications    : {', '.join(contraindications)}"
        )
    medicines_block = "\n".join(medicine_lines)

    scorecard_rows = []
    for med in medicines:
        name = med.get("name", "Unknown")
        scorecard_rows.append(
            f'  | {name:<40} | symptoms_match: YES/NO | '
            f'contraindicated: YES/NO | confidence_score: 0.0 |'
        )
    scorecard_template = "\n".join(scorecard_rows)

    json_slots = []
    for med in medicines:
        name = med.get("name", "Unknown")
        json_slots.append(
            f'    {{\n'
            f'      "name": "{name}",\n'
            f'      "reason": "<your reasoning for {name}>",\n'
            f'      "confidence_score": 0.0\n'
            f'    }}'
        )
    json_skeleton = ",\n".join(json_slots)

    # Comma-separated names guard against hallucination
    medicine_names = ", ".join(m.get("name", "") for m in medicines)

    few_shot_example = """\
EXAMPLE (do not copy — for format reference only)
══════════════════════════════════════════════════
Patient has: high fever, runny nose, mild headache, slight cough.

Scorecard:
  | MedicineA   | symptoms_match: YES | contraindicated: NO  | confidence_score: 0.85 |
  | MedicineB   | symptoms_match: YES | contraindicated: NO  | confidence_score: 0.70 |
  | MedicineC   | symptoms_match: YES | contraindicated: NO  | confidence_score: 0.60 |
  | MedicineD   | symptoms_match: NO  | contraindicated: NO  | confidence_score: 0.00 |

Final JSON (all medicines with confidence_score > 0.0 are included):
{
  "recommendations": [
    { "name": "MedicineA", "reason": "Directly treats fever and headache.", "confidence_score": 0.85 },
    { "name": "MedicineB", "reason": "Addresses runny nose symptoms.",      "confidence_score": 0.70 },
    { "name": "MedicineC", "reason": "Helps suppress the mild cough.",      "confidence_score": 0.60 }
  ]
}
══════════════════════════════════════════════════"""

    prompt = f"""You are a clinical decision-support assistant acting like a medical vending machine.
Given a patient's transcript and vitals, you MUST evaluate EVERY medicine in the inventory
and dispense ALL medicines that are relevant — not just one.

{few_shot_example}

════════════════════════════════════════
SECTION 1 – PATIENT TRANSCRIPT
════════════════════════════════════════
{transcript}

════════════════════════════════════════
SECTION 2 – PATIENT VITALS
════════════════════════════════════════
{vitals_block}

════════════════════════════════════════
SECTION 3 – MEDICINES INVENTORY  ({len(medicines)} medicines total)
════════════════════════════════════════
{medicines_block}

════════════════════════════════════════
YOUR TASK  — follow these steps in order
════════════════════════════════════════

STEP 1 ▸ FILL IN THE SCORECARD
Examine every medicine one by one. Complete the scorecard below.
Replace YES/NO and 0.0 with your actual assessments.
You MUST fill ALL {len(medicines)} rows before moving to Step 2.

{scorecard_template}

STEP 2 ▸ SELECTION RULE
From the completed scorecard, select every medicine where:
  • confidence_score > 0.0   AND
  • contraindicated = NO (or contraindication does not apply to this patient)

There is NO upper limit on how many medicines you may select.
NEVER return fewer medicines than actually match.

STEP 3 ▸ PRODUCE FINAL JSON
Fill in the pre-built skeleton below.
• Set confidence_score to 0.0 for medicines that do NOT match.
• Write a clear reason for every medicine you keep (score > 0).
• Remove entries with confidence_score = 0.0 from the final output.
• The output must be valid JSON — no markdown, no extra text.

ALLOWED MEDICINE NAMES (use exact spelling):
{medicine_names}

════════════════════════════════════════
OUTPUT SKELETON — fill this in exactly
════════════════════════════════════════
{{
  "recommendations": [
{json_skeleton}
  ]
}}

FINAL REMINDER
• A medical vending machine DISPENSES EVERYTHING that matches.
• If {len(medicines)} medicines all match → return all {len(medicines)}.
• If only 1 matches → return 1.
• NEVER truncate the list to save space.
• Return ONLY the JSON object — no commentary before or after it.
"""
    return prompt


# Ollama Service Check
def check_ollama() -> None:
    """
    Verify that the Ollama service is reachable before attempting a model call.
    Exits with a helpful message if the service is down.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        print(f"[✓] Ollama service is running at {OLLAMA_BASE_URL}")
    except requests.exceptions.ConnectionError:
        print("Error: Ollama service is not running. Start it with: ollama serve")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("Error: Ollama service timed out. Ensure it is running and responsive.")
        sys.exit(1)
    except requests.exceptions.RequestException as exc:
        print(f"Error: Could not reach Ollama service → {exc}")
        sys.exit(1)


# Model Query
def query_model(prompt: str) -> dict:
    print(f"[→] Sending prompt to model '{MODEL_NAME}' (this may take a moment)…")

    # System message enforces the assistant role and output contract
    system_message = (
        "You are a clinical decision-support assistant. "
        "You MUST respond with ONLY valid JSON that matches the exact schema requested. "
        "Do NOT include any explanation, markdown, or text outside the JSON object."
    )

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            format="json",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": prompt},
            ],
        )
    except ollama.ResponseError as exc:
        print(f"Error: Ollama model error → {exc}")
        sys.exit(1)
    except Exception as exc:
        # Catches network/connection failures at the ollama library level
        print(f"Error: Failed to communicate with Ollama → {exc}")
        sys.exit(1)

    # Extract raw text content from the response
    raw_content = response["message"]["content"]
    print("[✓] Response received from model.")

    # Parse and validate the JSON returned by the model
    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        print(f"Error: Model returned invalid JSON → {exc}")
        print(f"Raw response:\n{raw_content}")
        sys.exit(1)

    # Ensure the expected top-level key exists
    if "recommendations" not in result:
        print("Error: Model response is missing the 'recommendations' key.")
        print(f"Parsed response: {result}")
        sys.exit(1)

    # Validate each recommendation entry
    validated = []
    for entry in result["recommendations"]:
        name   = entry.get("name", "").strip()
        reason = entry.get("reason", "").strip()
        try:
            score = float(entry.get("confidence_score", 0.0))
            score = max(0.0, min(1.0, score))   # clamp to [0, 1]
        except (TypeError, ValueError):
            score = 0.0

        if name:   # skip empty-name entries
            validated.append({
                "name":             name,
                "reason":           reason,
                "confidence_score": round(score, 4),
            })

    # Sort by confidence descending for readability
    validated.sort(key=lambda x: x["confidence_score"], reverse=True)
    return {"recommendations": validated}


# Output Handling 
def save_and_print(result: dict) -> None:
    recommendations = result.get("recommendations", [])

    print("\n" + "═" * 60)
    print("  CLINICAL RECOMMENDATIONS")
    print("═" * 60)

    if not recommendations:
        print("  No medicines matched the patient's profile.")
    else:
        for idx, rec in enumerate(recommendations, start=1):
            print(f"\n  [{idx}] {rec['name']}")
            print(f"       Confidence : {rec['confidence_score']:.2f}")
            print(f"       Reason     : {rec['reason']}")

    print("═" * 60)

    # Save to recommendations.json
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print(f"\n[✓] Recommendations saved → {OUTPUT_FILE}")
    except OSError as exc:
        print(f"Error: Could not write recommendations.json → {exc}")
        sys.exit(1)


# Entry Point 
def main() -> None:
    print("\n═══════════════════════════════════════════")
    print("   Medical Suggestion Pipeline  (phi3)")
    print("═══════════════════════════════════════════\n")

    # Step 1 – Load all input files
    transcript, vitals, medicines = load_files()

    # Step 2 – Build the structured clinical prompt
    prompt = build_prompt(transcript, vitals, medicines)

    # Step 3 – Verify Ollama is reachable
    check_ollama()

    # Step 4 – Query the model
    result = query_model(prompt)

    # Step 5 – Display and persist the results
    save_and_print(result)


if __name__ == "__main__":
    main()