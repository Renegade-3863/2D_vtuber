"""End-to-end smoke test for XTTS voice cloning + forced alignment.

Run from project root:
    python ai_runtime/_test_xtts_e2e.py
"""
import os
import sys
import time
from pathlib import Path

# Make ai_runtime importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai_runtime.xtts_client import XTTSClient
from ai_runtime.forced_aligner import ForcedAligner


def main():
    ref = Path(__file__).resolve().parents[1] / "data" / "voice" / "keqing.mp3"
    if not ref.exists():
        print(f"ERROR: reference voice not found: {ref}")
        sys.exit(1)
    print(f"Reference voice: {ref}")

    print("\n[1/3] Loading XTTS v2 model (first run downloads ~2 GB) ...")
    t0 = time.time()
    tts = XTTSClient(speaker_wav=str(ref), language="zh")
    # Force model load now (lazy by default)
    tts._get_model()
    print(f"  XTTS ready in {time.time()-t0:.1f}s")

    text = "你好，我是刻晴。今天天气真不错，要不要一起去散散步？"
    print(f"\n[2/3] Synthesizing: {text!r}")
    t0 = time.time()
    audio_path, duration = tts.generate_audio(text)
    print(f"  generated in {time.time()-t0:.1f}s -> {audio_path} ({duration:.2f}s)")
    if not audio_path:
        print("ERROR: synthesis failed")
        sys.exit(2)

    print("\n[3/3] Running WhisperX forced alignment ...")
    t0 = time.time()
    aligner = ForcedAligner()
    boundaries = aligner.align(audio_path, text, language="zh")
    elapsed = time.time() - t0
    print(f"  aligned {len(boundaries)} word boundaries in {elapsed:.2f}s")
    for wb in boundaries[:10]:
        print(f"    {wb.offset:6.2f}-{wb.end:6.2f}s  '{wb.text}'")
    if len(boundaries) > 10:
        print(f"    ... ({len(boundaries)-10} more)")

    # Optional playback
    print(f"\nAudio file kept at: {audio_path}")
    print("OK")


if __name__ == "__main__":
    main()
