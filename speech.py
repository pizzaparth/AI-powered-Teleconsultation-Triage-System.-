# Parth Pancholi

import json
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

# constants
SAMPLE_RATE = 16_000  # Hz – Whisper expects 16 kHz
CHANNELS = 1  # mono
DTYPE = "int16"  # 16-bit PCM
WAV_PATH = Path("recorded_audio.wav")
TXT_PATH = Path("transcript.txt")
JSON_PATH = Path("transcript.json")
MODEL_SIZE = "small"
TASK = "translate"

# audio recording


def record_until_enter() -> np.ndarray:
    chunks: list[np.ndarray] = []
    stop_event = threading.Event()

    def _stream_callback(
        indata: np.ndarray, frames: int, time_info, status
    ) -> None:  # noqa: ANN001
        if status:
            print(f"[sounddevice] {status}", file=sys.stderr)
        chunks.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=_stream_callback,
    )

    def _wait_for_enter() -> None:
        input()  # blocks until the user presses Enter
        stop_event.set()

    enter_thread = threading.Thread(target=_wait_for_enter, daemon=True)

    print("🎙  Recording … press Enter to stop.")
    with stream:
        enter_thread.start()
        stop_event.wait()  # block main thread until Enter is pressed

    print("⏹  Recording stopped.")

    if not chunks:
        raise RuntimeError("No audio was captured. Check your microphone.")

    return np.concatenate(chunks, axis=0).flatten()


def save_wav(audio: np.ndarray, path: Path) -> None:
    """Write a 16-bit PCM WAV file."""
    wav_write(str(path), SAMPLE_RATE, audio)
    print(f"💾  Audio saved → {path}")


# transcription


def transcribe(wav_path: Path) -> dict:
    try:
        import whisper  # iss line ko mat hatana lol.
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "openai-whisper is not installed.\n" "Run:  pip install openai-whisper"
        ) from exc

    print(f"🔄  Loading Whisper model '{MODEL_SIZE}' …")
    model = whisper.load_model(MODEL_SIZE)

    print(f"📝  Transcribing {wav_path} …")
    result = model.transcribe(str(wav_path), task=TASK)

    return {
        "language": result.get("language", "unknown"),
        "text": result.get("text", "").strip(),
    }


# output writer


def save_txt(data: dict, path: Path) -> None:
    path.write_text(data["text"], encoding="utf-8")
    print(f"📄  Transcript saved → {path}")


def save_json(data: dict, path: Path) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"📋  JSON result saved → {path}")


# main


def main() -> None:
    # 1. Record
    audio = record_until_enter()
    save_wav(audio, WAV_PATH)

    # 2. Transcribe
    result = transcribe(WAV_PATH)

    # 3. Save outputs
    save_txt(result, TXT_PATH)
    save_json(result, JSON_PATH)

    # 4. Print summary
    print("\n─── Result ────────────────────────────────────────────────")
    print(f"Detected language : {result['language']}")
    print(f"Transcript        : {result['text']}")
    print("────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
