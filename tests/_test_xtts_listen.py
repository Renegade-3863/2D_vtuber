"""Quick listen test: synthesize a few sentences with the cloned voice and play them."""
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_runtime.xtts_client import XTTSClient

REF = os.environ.get(
    "XTTS_REFERENCE_WAV",
    str(ROOT / "data" / "voice" / "keqing.mp3"),
)

SENTENCES = [
    # 平静陈述
    "今天的天气还不错。",
    # 开心兴奋（多感叹号 + 语气词）
    "哇！你真的来啦！我等你好久了诶！",
    # 撒娇 / 不满（语气词 + 拖音）
    "哼……不理你了啦，明明说好要早点来的嘛~",
    # 惊讶
    "什么？！这种事居然真的发生了？",
    # 严肃 / 命令
    "退后，让我来处理这件事。",
    # 温柔安慰
    "没关系的，慢慢来，我会一直陪着你。",
    # 英文混合
    "Hmm... well, I guess this is fine.",
]

def main():
    print(f"reference: {REF}")
    tts = XTTSClient(speaker_wav=REF)
    tts._get_model()  # warm up
    for i, text in enumerate(SENTENCES, 1):
        print(f"\n[{i}/{len(SENTENCES)}] {text}")
        t0 = time.time()
        wav, dur = tts.generate_audio(text)
        print(f"  synth {time.time()-t0:.1f}s, audio {dur:.2f}s -> {wav}")
        tts.play_audio_async(wav)
        tts.wait_playback()
    print("\nDone.")

if __name__ == "__main__":
    main()
