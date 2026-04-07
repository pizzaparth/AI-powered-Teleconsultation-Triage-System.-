# modules/tts.py
# Parth Pancholi — Text-to-Speech module
# Converts patient summary to audio, saves as WAV, and auto-plays it.

from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT = "static/patient_summary.wav"
TTS_RATE       = 150    # words per minute (lower = clearer for rural use)
TTS_VOLUME     = 1.0    # 0.0 – 1.0


# ── Internal helpers ───────────────────────────────────────────────────────────

def _synthesize(text: str, output_path: Path) -> None:
    """
    Convert *text* to speech and save as a WAV file using pyttsx3.
    Fully offline — no internet required, works on Windows / macOS / Linux.
    """
    try:
        import pyttsx3
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyttsx3 is not installed. Run: pip install pyttsx3"
        ) from exc

    log.info("Synthesizing speech for %d chars of text …", len(text))

    engine = pyttsx3.init()
    engine.setProperty("rate",   TTS_RATE)
    engine.setProperty("volume", TTS_VOLUME)

    engine.save_to_file(text, str(output_path))
    engine.runAndWait()

    log.info("Audio saved → %s", output_path)


def _play_audio(path: Path) -> None:
    """
    Auto-play the WAV file without requiring any user interaction.
    Runs in background so it doesn't block the API response.
    """
    system   = platform.system()
    abs_path = str(path.resolve())

    try:
        if system == "Windows":
            subprocess.Popen(
                ["powershell", "-c", f'Start-Process "{abs_path}"'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        elif system == "Darwin":  # macOS
            subprocess.Popen(
                ["afplay", abs_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        else:  # Linux — prefer WAV-native ALSA players
            for player in ["aplay", "paplay", "ffplay", "sox"]:
                if subprocess.call(
                    ["which", player],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ) == 0:
                    subprocess.Popen(
                        [player, abs_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    log.info("Playing audio with '%s'", player)
                    return
            log.warning(
                "No audio player found on Linux. "
                "Install one with: sudo apt install alsa-utils"
            )
            return

        log.info("Audio playback started (%s) → %s", system, abs_path)

    except Exception as exc:
        log.warning("Audio playback failed (non-fatal): %s", exc)


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_tts(
    text: str,
    output_path: str | Path = DEFAULT_OUTPUT,
    auto_play: bool = True,
) -> str:
    """
    Convert *text* to speech, save as WAV, and optionally auto-play it.

    Parameters
    ----------
    text        : The text to synthesize (typically the patient_summary field).
    output_path : Where to save the WAV file (default: patient_summary.wav).
    auto_play   : If True, play the audio automatically after saving.

    Returns
    -------
    str : Absolute path to the saved WAV file.
    """
    if not text or not text.strip():
        raise ValueError("TTS input text is empty — nothing to synthesize.")

    out = Path(output_path)

    # Enforce .wav extension regardless of what was passed in
    if out.suffix.lower() != ".wav":
        log.warning("Output path had extension '%s' — changing to .wav", out.suffix)
        out = out.with_suffix(".wav")

    out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Synthesize and save
    _synthesize(text.strip(), out)

    # 2. Auto-play
    if auto_play:
        _play_audio(out)
    else:
        log.info("Auto-play disabled — skipping playback.")

    return str(out.resolve())