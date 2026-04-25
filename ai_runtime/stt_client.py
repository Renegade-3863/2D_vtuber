import os
import re
import tempfile
import time
from pathlib import Path

import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

import numpy as np

class STTClient:
    def __init__(
        self, 
        sample_rate=16000,
        model_dir_name: str = "faster-whisper-small",
        device: str = "cpu",
        compute_type: str = "int8",
        silence_threshold: float = 0.008,
        chunk_seconds: float = 0.1,
    ):    
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.chunk_seconds = chunk_seconds

        project_root = Path(__file__).resolve().parents[1]

        self.model_path = project_root / "models" / model_dir_name

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Whisper model directory not found at {self.model_path}\n"
                f"Please put model files under: {project_root / 'models' / model_dir_name}"
            )
        
        self.model = WhisperModel(
            str(self.model_path),
            device=device,
            compute_type=compute_type,
        )

    def _rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(audio))))

    def _is_silent(self, audio: np.ndarray):
        return self._rms(audio) < self.silence_threshold
        
    def record_once(self, seconds: float = 5.0):
        chunks = []
        total_steps = max(1, int(seconds / self.chunk_seconds))
        chunk_frames = int(self.sample_rate * self.chunk_seconds)

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32"
        ) as stream:
            for _ in range(total_steps):
                data, _ = stream.read(chunk_frames)
                chunks.append(data)
        
        return np.concatenate(chunks, axis=0)

    def record_vad(
        self,
        max_seconds: float = 15.0,
        silence_duration: float = 1.2,
        min_speech_seconds: float = 0.3,
    ) -> np.ndarray | None:
        """VAD 录音：检测到说话开始后，等待静音自动停止。
        
        返回音频 numpy 数组，如果没检测到有效语音返回 None。
        """
        chunk_frames = int(self.sample_rate * self.chunk_seconds)
        max_chunks = int(max_seconds / self.chunk_seconds)
        silence_chunks_needed = int(silence_duration / self.chunk_seconds)

        chunks: list[np.ndarray] = []
        speech_started = False
        silence_count = 0
        speech_chunk_count = 0

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        ) as stream:
            for i in range(max_chunks):
                data, _ = stream.read(chunk_frames)
                chunks.append(data)
                
                is_loud = self._rms(data) >= self.silence_threshold

                if not speech_started:
                    if is_loud:
                        speech_started = True
                        speech_chunk_count = 1
                        silence_count = 0
                else:
                    if is_loud:
                        speech_chunk_count += 1
                        silence_count = 0
                    else:
                        silence_count += 1
                        if silence_count >= silence_chunks_needed:
                            break

        if not speech_started:
            return None

        if speech_chunk_count < int(min_speech_seconds / self.chunk_seconds):
            return None

        audio = np.concatenate(chunks, axis=0)
        # 裁掉尾部纯静音
        trim_samples = int(max(0, silence_count - 2) * chunk_frames)
        if trim_samples > 0 and trim_samples < len(audio):
            audio = audio[: len(audio) - trim_samples]
        return audio

    def transcribe_array(self, audio_array, language: str | None = "zh") -> str:
        """Transcribe a numpy audio array by saving to a temp wav first."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            write(wav_path, self.sample_rate, audio_array)
            kwargs = {}
            if language:
                kwargs["language"] = language
            segments, _ = self.model.transcribe(wav_path, **kwargs)
            text = "".join(seg.text for seg in segments).strip()
            return text
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    def listen_and_transcribe(self, language: str | None = None, max_seconds: float = 15.0) -> str:
        """VAD 录音 + 转写。language=None 时自动检测。返回转写文字，无有效语音则返回空串。"""
        audio = self.record_vad(max_seconds=max_seconds)
        if audio is None:
            return ""
        # 自动检测或使用指定语言
        lang = language or None  # None → faster_whisper 自动检测
        text = self.transcribe_array(audio, language=lang)
        if self._is_hallucination(text):
            return ""
        return text

    # ---- Whisper 幻觉过滤 ----
    _HALLUCINATION_EXACT = {
        # 英文
        "thank you.", "thank you", "thanks.", "thanks",
        "bye.", "bye", "goodbye.", "you", "yeah.",
        # 中文
        "谢谢观看", "谢谢", "字幕由", "请不吝点赞", "订阅",
        "感谢观看", "感谢收看", "感谢聆听",
        "再见", "拜拜",
    }
    _HALLUCINATION_PATTERNS = re.compile(
        r"字幕|subtitle|subscri|请不吝|感谢.*观看|謝謝.*觀看"
        r"|copyright|music|♪|♫|🎵"
        r"|^\.+$|^\s*$",
        re.IGNORECASE,
    )

    def _is_hallucination(self, text: str) -> bool:
        t = text.strip()
        if not t or len(t) <= 1:
            return True
        if t.lower() in self._HALLUCINATION_EXACT:
            return True
        if self._HALLUCINATION_PATTERNS.search(t):
            return True
        return False
    
if __name__ == "__main__":
    stt = STTClient()
    print("说话测试（静音 1.2s 自动结束）...")
    text = stt.listen_and_transcribe(language="zh")
    print("识别结果:", text)