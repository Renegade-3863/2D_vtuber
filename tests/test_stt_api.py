import tempfile
import sounddevice as sd
from scipy.io.wavfile import write
from openai import AzureOpenAI
from ai_runtime.config_api import cfg

STT_MODEL = "whisper-1"

def record_wav(seconds=4, sr=16000):
    print(f"Recording {seconds}s...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    write(f.name, sr, audio)
    return f.name

def main():
    client = AzureOpenAI(
        azure_endpoint=cfg.endpoint,
        api_key=cfg.api_key,
        api_version=cfg.api_version
    )

    wav_path = record_wav()
    with open(wav_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f
        )
    print("Transcription:", resp.text)

if __name__ == "__main__":
    main()