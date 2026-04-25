"""Quick listen test for local GPT-SoVITS API voice cloning.

Run from project root:
    set GPT_SOVITS_API_URL=http://127.0.0.1:9880/tts
    set GPT_SOVITS_REFERENCE_WAV=data/voice/keqing.mp3
    python ai_runtime/_test_gpt_sovits_listen.py
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_runtime.gpt_sovits_client import GPTSoVITSClient

REF = os.environ.get(
    "GPT_SOVITS_REFERENCE_WAV",
    str(ROOT / "data" / "voice" / "keqing.mp3"),
)

SENTENCES = [
    "你好，我是刻晴，今天要继续高效推进这个项目。",
    "延迟已经降下来了，我们现在可以边说边做口型热替换。",
    "If this sounds natural enough, we can switch the main runtime to GPT-SoVITS.",
]


def main():
    print(f"reference: {REF}")
    print(f"api url : {os.environ.get('GPT_SOVITS_API_URL', 'http://127.0.0.1:9880/tts')}")

    tts = GPTSoVITSClient(speaker_wav=REF, language="auto")

    for i, text in enumerate(SENTENCES, 1):
        print(f"\n[{i}/{len(SENTENCES)}] {text}")
        t0 = time.time()
        wav, dur = tts.generate_audio(text)
        dt = time.time() - t0
        print(f"  synth {dt:.2f}s, audio {dur:.2f}s -> {wav}")
        if not wav:
            print("  generation failed; stop test")
            return
        tts.play_audio_async(wav)
        tts.wait_playback()

    print("\nDone.")


if __name__ == "__main__":
    main()
