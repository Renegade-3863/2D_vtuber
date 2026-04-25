"""
XTTS v2 voice-cloning TTS client.

Provides the same public interface as ``TTSClient`` (see :mod:`tts_client`):

    - ``generate_audio(text) -> (audio_path, duration)``
    - ``play_audio_async(audio_path)``
    - ``wait_playback()``
    - ``speak_async(text)``
    - ``get_playback_position()``
    - ``estimate_duration(text)``
    - ``self.word_boundaries``  (filled by an optional forced-alignment pass)

Differences from Edge TTS based ``TTSClient``:

    - Voice is cloned from a reference WAV file (``speaker_wav``)
    - Output is WAV (24 kHz) instead of MP3
    - ``word_boundaries`` are NOT produced by the engine itself; they are
      attached after the fact by ``ai_runtime.forced_aligner.ForcedAligner``
      via :meth:`set_word_boundaries`. The lip-sync pipeline is designed to
      gracefully degrade to weighted-estimation alignment when boundaries
      are absent.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import List

import pygame

# Make sure the bundled ffmpeg binary is on PATH so libraries (XTTS, librosa
# audioread fallback, etc.) that shell out to ``ffmpeg`` can find it.
try:
    import imageio_ffmpeg
    _ff = imageio_ffmpeg.get_ffmpeg_exe()
    _ff_dir = str(Path(_ff).parent)
    if _ff_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ff_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

# Re-export ``WordBoundary`` so callers can import it from either module.
from ai_runtime.tts_client import WordBoundary, get_audio_duration  # noqa: E402


# Coqui XTTS v2 model id on Hugging Face / Coqui hub.
DEFAULT_XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


# Map our short language codes to the ones XTTS v2 expects.
# XTTS v2 supports: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko
_LANG_ALIAS = {
    "zh": "zh-cn",
    "zh-cn": "zh-cn",
    "zh-tw": "zh-cn",
    "en": "en",
    "ja": "ja",
}


class XTTSClient:
    """Voice-cloning TTS client backed by Coqui XTTS v2."""

    # Lazy-loaded singleton model (XTTS is ~2 GB on disk, ~2 GB VRAM).
    # We keep it as a class attribute so multiple ``XTTSClient`` instances
    # share the same loaded model.
    _shared_model = None
    _shared_model_lock = threading.Lock()

    def __init__(
        self,
        speaker_wav: str,
        language: str = "auto",
        device: str | None = None,
        model_name: str = DEFAULT_XTTS_MODEL,
    ):
        """
        Args:
            speaker_wav: Path to a clean reference voice clip (WAV/MP3 OK,
                6~15 s of single-speaker speech recommended).
            language: ``'auto' | 'en' | 'zh' | 'ja' | ...``  When ``auto``,
                the language is decided per-call by ``generate_audio``.
            device: ``'cuda'`` | ``'cpu'``. Defaults to CUDA when available.
            model_name: Coqui model id; defaults to XTTS v2 multilingual.
        """
        if not speaker_wav or not os.path.isfile(speaker_wav):
            raise FileNotFoundError(
                f"XTTS speaker reference not found: {speaker_wav}"
            )

        self.speaker_wav = speaker_wav
        self.language = language
        self.model_name = model_name

        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device

        # Mirror the public surface of :class:`TTSClient`.
        self.word_boundaries: List[WordBoundary] = []
        self._playback_thread: threading.Thread | None = None
        self._audio_duration: float = 0.0
        self._playback_start_time: float = 0.0
        self._is_playing: bool = False
        self._audio_position: float = 0.0

    # ------------------------------------------------------------------
    # Model loading (lazy + shared across instances)
    # ------------------------------------------------------------------
    def _get_model(self):
        cls = type(self)
        if cls._shared_model is not None:
            return cls._shared_model
        with cls._shared_model_lock:
            if cls._shared_model is None:
                # Imported lazily to keep startup cheap when XTTS is unused.
                # Coqui XTTS v2 weights are hosted under a non-commercial
                # licence; the library prompts for acceptance on first download
                # unless this env var is set.
                os.environ.setdefault("COQUI_TOS_AGREED", "1")
                from TTS.api import TTS as CoquiTTS  # type: ignore

                print(f"[XTTS] Loading {self.model_name} on {self.device} ...")
                t0 = time.time()
                model = CoquiTTS(self.model_name).to(self.device)
                print(f"[XTTS] Model loaded in {time.time()-t0:.1f}s")
                cls._shared_model = model
        return cls._shared_model

    # ------------------------------------------------------------------
    # Public API (mirrors TTSClient)
    # ------------------------------------------------------------------
    def generate_audio(self, text: str) -> tuple[str | None, float]:
        """Synthesize ``text`` with the cloned voice and return (path, dur)."""
        text = (text or "").strip()
        if not text:
            self.word_boundaries = []
            return None, 0.0

        lang = self.language if self.language != "auto" else self._detect_language(text)
        lang = _LANG_ALIAS.get(lang, lang)

        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"xtts_{int(time.time() * 1000)}.wav")

        try:
            model = self._get_model()
            model.tts_to_file(
                text=text,
                file_path=audio_path,
                speaker_wav=self.speaker_wav,
                language=lang,
                split_sentences=True,
                # --- expressiveness controls (XTTS v2) ---
                # higher temperature -> more variation in prosody/emotion
                temperature=float(os.environ.get("XTTS_TEMPERATURE", 0.85)),
                # discourage flat/monotone repetition
                repetition_penalty=float(os.environ.get("XTTS_REP_PENALTY", 5.0)),
                # encourages slightly longer utterances (more natural prosody)
                length_penalty=float(os.environ.get("XTTS_LEN_PENALTY", 1.0)),
                top_k=int(os.environ.get("XTTS_TOP_K", 50)),
                top_p=float(os.environ.get("XTTS_TOP_P", 0.85)),
            )
        except Exception as e:
            print(f"[XTTS] synthesis failed: {e}")
            self.word_boundaries = []
            return None, 0.0

        if not os.path.exists(audio_path):
            self.word_boundaries = []
            return None, 0.0

        duration = get_audio_duration(audio_path)
        self._audio_duration = duration
        # word_boundaries are filled later (forced alignment); start empty so
        # the lip-sync pipeline falls back to weighted estimation immediately.
        self.word_boundaries = []
        return audio_path, duration

    def set_word_boundaries(self, boundaries: List[WordBoundary]) -> None:
        """Attach forced-alignment results produced by ForcedAligner."""
        self.word_boundaries = boundaries or []

    def get_playback_position(self) -> float:
        if not self._is_playing:
            return 0.0
        return self._audio_position

    def play_audio_async(self, audio_path: str) -> None:
        if not audio_path or not os.path.exists(audio_path):
            return

        if not pygame.mixer.get_init():
            pygame.mixer.init()

        def _play():
            self._is_playing = True
            self._audio_position = 0.0

            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            self._playback_start_time = time.time()

            while pygame.mixer.music.get_busy():
                time.sleep(0.02)
                self._audio_position = min(
                    time.time() - self._playback_start_time,
                    self._audio_duration,
                )

            self._is_playing = False
            self._audio_position = self._audio_duration
            pygame.mixer.music.unload()
            try:
                os.remove(audio_path)
            except Exception:
                pass

        self._playback_thread = threading.Thread(target=_play, daemon=True)
        self._playback_thread.start()

    def wait_playback(self) -> None:
        if self._playback_thread:
            self._playback_thread.join()

    def speak_async(self, text: str) -> tuple[threading.Thread | None, float]:
        audio_path, duration = self.generate_audio(text)
        if audio_path:
            self.play_audio_async(audio_path)
            return self._playback_thread, duration
        return None, duration

    def estimate_duration(self, text: str) -> float:
        text = (text or "").strip()
        if not text:
            return 0.0
        return max(1.2, len(text) / 12.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_language(text: str) -> str:
        """Cheap ASCII vs. CJK heuristic, identical to AudioPhonemeAligner."""
        if not text:
            return "en"
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        return "zh" if cjk >= max(1, len(text) // 6) else "en"
