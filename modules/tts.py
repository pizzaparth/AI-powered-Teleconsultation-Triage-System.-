# modules/tts.py
# Parth Pancholi — Text-to-Speech module
# Converts patient summary to audio, saves as MP3, and auto-plays it.

from __future__ import annotations

import logging
import os
import platform
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT   = "patient_summary.mp3"
TTS_LANGUAGE     = "en"       # gTTS language code
TTS_SLOW         = False      # False = normal speed, True = slower (clearer for rural use)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _synthesize(text: str, output_path: Path) -> None:
    """
    Convert *text* to speech and save as an MP3 file using gTTS.
    gTTS uses Google's TTS engine — requires internet on first use,
    but is the most reliable cross-platform option with no C++ deps.
    """
    try:
        from gtts import gTTS
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "gTTS is not installed. Run: pip install gtts"
        ) from exc

    log.info("Synthesizing speech for %d chars of text …", len(text))
    tts = gTTS(text=text, lang=TTS_LANGUAGE, slow=TTS_SLOW)
    tts.save(str(output_path))
    log.info("Audio saved → %s", output_path)


def _play_audio(path: Path) -> None:
    """
    Auto-play the audio file without requiring any user interaction.
    Uses the correct system command for Windows / macOS / Linux.
    Runs in background so it doesn't block the API response.
    """
    system = platform.system()
    abs_path = str(path.resolve())

    try:
        if system == "Windows":
            # PowerShell Media.SoundPlayer doesn't support MP3 —
            # use the built-in wmplayer or start (which picks the default app)
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

        else:  # Linux
            # Try common players in order of preference
            for player in ["mpg123", "mpg321", "ffplay", "aplay"]:
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
                "Install one with: sudo apt install mpg123"
            )
            return

        log.info("Audio playback started (%s) → %s", system, abs_path)

    except Exception as exc:
        # Playback failure should never crash the API pipeline
        log.warning("Audio playback failed (non-fatal): %s", exc)


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_tts(
    text: str,
    output_path: str | Path = DEFAULT_OUTPUT,
    auto_play: bool = True,
) -> str:
    """
    Convert *text* to speech, save as MP3, and optionally auto-play it.

    Parameters
    ----------
    text        : The text to synthesize (typically the patient_summary field).
    output_path : Where to save the MP3 file (default: patient_summary.mp3).
    auto_play   : If True, play the audio automatically after saving.

    Returns
    -------
    str : Absolute path to the saved MP3 file.
    """
    if not text or not text.strip():
        raise ValueError("TTS input text is empty — nothing to synthesize.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Synthesize and save
    _synthesize(text.strip(), out)

    # 2. Auto-play
    if auto_play:
        _play_audio(out)
    else:
        log.info("Auto-play disabled — skipping playback.")

    return str(out.resolve())