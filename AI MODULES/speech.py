# modules/speech.py
# Parth Pancholi — Silero VAD silence detection, no Enter key required

from __future__ import annotations

import json
import logging
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from scipy.io.wavfile import write as wav_write

log = logging.getLogger(__name__)

# ── Audio constants ────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000   # Hz — Whisper + Silero both expect 16 kHz
CHANNELS    = 1        # mono
DTYPE       = "float32"  # Silero needs float32 (converted to int16 only when saving)

# ── Whisper constants ──────────────────────────────────────────────────────────
MODEL_SIZE = "small"
TASK       = "translate"

# ── VAD tuning parameters ──────────────────────────────────────────────────────
# Silero processes 512-sample chunks at 16 kHz = 32 ms per chunk
VAD_CHUNK_SAMPLES = 512
SPEECH_THRESHOLD  = 0.5    # probability above which a chunk is considered speech (0–1)
SILENCE_TIMEOUT_S = 2.5    # seconds of silence before recording stops
MIN_SPEECH_S      = 1.0    # minimum speech before silence can trigger stop
MAX_RECORDING_S   = 120    # hard ceiling in seconds


# ── Silero VAD loader ──────────────────────────────────────────────────────────

def _load_silero_vad():
    """Download (first run) or load cached Silero VAD model."""
    try:
        model, _ = torch.hub.load(
            repo_or_dir = "snakers4/silero-vad",
            model       = "silero_vad",
            force_reload = False,
            onnx        = False,
            verbose     = False,
            skip_validation = True,   # avoids GitHub rate limit errors
        )
        model.eval()
        log.info("Silero VAD model loaded.")
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Silero VAD model: {exc}\n"
            "Make sure you have internet access on first run, or run:\n"
            "  pip install torch torchaudio silero-vad"
        ) from exc


# ── Recording ─────────────────────────────────────────────────────────────────

def _record_until_silence(
    silence_timeout: float = SILENCE_TIMEOUT_S,
    min_speech_s:    float = MIN_SPEECH_S,
    max_duration:    float = MAX_RECORDING_S,
) -> np.ndarray:
    """
    Record from microphone and stop automatically after *silence_timeout*
    seconds of continuous silence, using Silero VAD.

    Algorithm
    ---------
    1. Stream audio in 512-sample (32 ms) chunks.
    2. Run each chunk through Silero VAD to get a speech probability (0–1).
    3. Chunk is speech if probability >= SPEECH_THRESHOLD.
    4. Stop when:
       - Consecutive silence exceeds silence_timeout, AND
       - Total detected speech >= min_speech_s.
    5. Hard stop after max_duration seconds.

    Returns
    -------
    np.ndarray : Flat int16 audio array at 16 kHz.
    """
    vad_model = _load_silero_vad()

    # Thresholds in chunk counts
    chunks_per_sec    = SAMPLE_RATE / VAD_CHUNK_SAMPLES        # 31.25 chunks/sec
    max_silent_chunks = int(silence_timeout * chunks_per_sec)
    min_speech_chunks = int(min_speech_s    * chunks_per_sec)
    max_total_chunks  = int(max_duration    * chunks_per_sec)

    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            log.warning("[sounddevice] %s", status)
        audio_q.put(indata.copy())

    all_chunks:    list[np.ndarray] = []
    leftover:      np.ndarray = np.array([], dtype=np.float32)
    silent_count:  int = 0
    speech_count:  int = 0
    total_count:   int = 0

    log.info(
        "Silero VAD recording — silence=%.1fs | min_speech=%.1fs | max=%.0fs",
        silence_timeout, min_speech_s, max_duration,
    )
    print("🎙  Listening … (will stop automatically after silence)")

    stream = sd.InputStream(
        samplerate = SAMPLE_RATE,
        channels   = CHANNELS,
        dtype      = DTYPE,
        blocksize  = VAD_CHUNK_SAMPLES,
        callback   = _callback,
    )

    with stream:
        while True:
            try:
                chunk = audio_q.get(timeout=1.0).flatten()
            except queue.Empty:
                log.warning("Audio queue empty — microphone may have stalled.")
                continue

            all_chunks.append(chunk)
            total_count += 1

            # Combine leftover samples with new chunk, process in VAD_CHUNK_SAMPLES blocks
            buf = np.concatenate([leftover, chunk])

            while len(buf) >= VAD_CHUNK_SAMPLES:
                frame   = buf[:VAD_CHUNK_SAMPLES]
                buf     = buf[VAD_CHUNK_SAMPLES:]

                # Silero expects a float32 torch tensor
                tensor  = torch.from_numpy(frame).unsqueeze(0)
                with torch.no_grad():
                    prob = vad_model(tensor, SAMPLE_RATE).item()

                is_speech = prob >= SPEECH_THRESHOLD

                if is_speech:
                    speech_count += 1
                    silent_count  = 0
                else:
                    silent_count += 1

            leftover = buf  # carry unprocessed samples forward

            # ── Stop conditions ────────────────────────────────────────────────
            if silent_count >= max_silent_chunks and speech_count >= min_speech_chunks:
                log.info(
                    "Silence threshold reached — speech_chunks=%d silent_chunks=%d",
                    speech_count, silent_count,
                )
                break

            if total_count >= max_total_chunks:
                log.warning("Max duration (%.0fs) reached — forcing stop.", max_duration)
                break

    print("⏹  Recording stopped.")

    if not all_chunks:
        raise RuntimeError("No audio was captured. Check your microphone.")

    # Concatenate float32, convert to int16 for WAV saving
    audio_f32 = np.concatenate(all_chunks)
    audio_i16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)

    speech_s = speech_count / chunks_per_sec
    total_s  = total_count  / chunks_per_sec
    log.info(
        "Recording complete — total=%.1fs | speech=%.1fs | samples=%d",
        total_s, speech_s, len(audio_i16),
    )

    if speech_count < min_speech_chunks:
        log.warning(
            "Very little speech detected (%.1fs). Transcript may be inaccurate.", speech_s
        )

    return audio_i16


# ── WAV writer ─────────────────────────────────────────────────────────────────

def _save_wav(audio: np.ndarray, path: Path) -> None:
    """Write a 16-bit PCM WAV file."""
    wav_write(str(path), SAMPLE_RATE, audio)
    log.info("Audio saved → %s", path)


# ── Whisper transcription ──────────────────────────────────────────────────────

def _transcribe(wav_path: Path) -> dict:
    """Load Whisper and transcribe the given WAV file."""
    try:
        import whisper
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        ) from exc

    log.info("Loading Whisper model '%s' …", MODEL_SIZE)
    model = whisper.load_model(MODEL_SIZE)

    log.info("Transcribing %s …", wav_path)
    result = model.transcribe(str(wav_path), task=TASK)

    return {
        "language": result.get("language", "unknown"),
        "text":     result.get("text", "").strip(),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_transcript(
    wav_path:        str | Path = "recorded_audio.wav",
    txt_path:        str | Path = "transcript.txt",
    json_path:       str | Path = "transcript.json",
    silence_timeout: float      = SILENCE_TIMEOUT_S,
    min_speech_s:    float      = MIN_SPEECH_S,
    max_duration:    float      = MAX_RECORDING_S,
) -> str:
    """
    Full speech pipeline — no user interaction required:
      1. Record patient audio via Silero VAD (auto-stops on silence).
      2. Transcribe with Whisper (translates to English).
      3. Persist transcript.txt and transcript.json.
      4. Return the transcript string.

    Parameters
    ----------
    wav_path        : Where to save the raw WAV recording.
    txt_path        : Where to save the plain-text transcript.
    json_path       : Where to save the JSON result {language, text}.
    silence_timeout : Seconds of silence that trigger stop (default 2.5 s).
    min_speech_s    : Minimum speech before silence can stop recording (default 1 s).
    max_duration    : Hard ceiling on recording length in seconds (default 120 s).

    Returns
    -------
    str : The transcribed text.
    """
    wav_path  = Path(wav_path)
    txt_path  = Path(txt_path)
    json_path = Path(json_path)

    # 1. Record with Silero VAD
    audio = _record_until_silence(
        silence_timeout = silence_timeout,
        min_speech_s    = min_speech_s,
        max_duration    = max_duration,
    )
    _save_wav(audio, wav_path)

    # 2. Transcribe
    result          = _transcribe(wav_path)
    transcript_text = result["text"]

    # 3. Persist outputs
    txt_path.write_text(transcript_text, encoding="utf-8")
    log.info("Transcript saved → %s", txt_path)

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Transcript JSON saved → %s", json_path)

    log.info(
        "Speech pipeline complete | language=%s | length=%d chars",
        result["language"], len(transcript_text),
    )
    return transcript_text


# ── Stand-alone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    text = generate_transcript()
    print(f"\nTranscript: {text}\n")