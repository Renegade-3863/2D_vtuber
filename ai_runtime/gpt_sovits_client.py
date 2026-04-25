"""
GPT-SoVITS local API client (voice cloning).

Public interface mirrors TTSClient/XTTSClient:
    - generate_audio(text) -> (audio_path, duration)
    - play_audio_async(audio_path)
    - wait_playback()
    - speak_async(text)
    - get_playback_position()
    - estimate_duration(text)
    - self.word_boundaries (empty; fill via ForcedAligner)

Expected API style (common GPT-SoVITS WebUI endpoints):
    POST /tts
    JSON payload with keys like:
        text, text_lang, ref_audio_path, prompt_text, prompt_lang,
        top_k, top_p, temperature, speed_factor, media_type, streaming_mode

Response can be either:
    1) raw audio bytes (wav), or
    2) JSON with one of: audio_path / path / output_path / audio(base64)
"""
from __future__ import annotations

import base64
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import List

import pygame
import requests

from ai_runtime.tts_client import WordBoundary, get_audio_duration


DEFAULT_GPT_SOVITS_API_URL = os.environ.get(
    "GPT_SOVITS_API_URL", "http://127.0.0.1:9880/tts"
)


def _lang_for_gpt_sovits(language: str, text: str) -> str:
    """Map our language to common GPT-SoVITS text_lang values."""
    if language and language != "auto":
        low = language.strip().lower()
    else:
        cjk = sum(1 for ch in (text or "") if "\u4e00" <= ch <= "\u9fff")
        low = "zh" if cjk >= max(1, len(text) // 6) else "en"

    if low in ("zh", "zh-cn", "zh-tw"):
        return "zh"
    if low.startswith("ja"):
        return "ja"
    return "en"


class GPTSoVITSClient:
    """Voice-cloning TTS client backed by a local GPT-SoVITS HTTP API."""

    def __init__(
        self,
        speaker_wav: str,
        language: str = "auto",
        api_url: str = DEFAULT_GPT_SOVITS_API_URL,
    ):
        if not speaker_wav or not os.path.isfile(speaker_wav):
            raise FileNotFoundError(f"GPT-SoVITS reference audio not found: {speaker_wav}")

        self.speaker_wav = speaker_wav
        self.language = language
        self.api_url = api_url

        # Optional prompt text/lang used by many GPT-SoVITS deployments.
        self.prompt_text = os.environ.get("GPT_SOVITS_PROMPT_TEXT", "")
        self.prompt_lang = os.environ.get("GPT_SOVITS_PROMPT_LANG", "zh")

        self.word_boundaries: List[WordBoundary] = []
        self._playback_thread: threading.Thread | None = None
        self._audio_duration: float = 0.0
        self._playback_start_time: float = 0.0
        self._is_playing: bool = False
        self._audio_position: float = 0.0

    def _build_payload(self, text: str, lang: str) -> dict:
        payload = {
            "text": text,
            "text_lang": lang,
            "ref_audio_path": self.speaker_wav,
            "prompt_text": self.prompt_text,
            "prompt_lang": self.prompt_lang,
            "media_type": os.environ.get("GPT_SOVITS_MEDIA_TYPE", "wav"),
            "streaming_mode": False,
            "top_k": int(os.environ.get("GPT_SOVITS_TOP_K", "15")),
            "top_p": float(os.environ.get("GPT_SOVITS_TOP_P", "1.0")),
            "temperature": float(os.environ.get("GPT_SOVITS_TEMPERATURE", "1.0")),
            "speed_factor": float(os.environ.get("GPT_SOVITS_SPEED", "1.0")),
        }
        # Some forks require these exact field names.
        payload.setdefault("text_split_method", os.environ.get("GPT_SOVITS_SPLIT", "cut5"))
        return payload

    def _save_temp_wav(self, data: bytes) -> str:
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"gpt_sovits_{int(time.time() * 1000)}.wav")
        with open(audio_path, "wb") as f:
            f.write(data)
        return audio_path

    def generate_audio(self, text: str) -> tuple[str | None, float]:
        text = (text or "").strip()
        if not text:
            self.word_boundaries = []
            return None, 0.0

        # Retry on suspiciously short audio. GPT-SoVITS T2S can EOS too early
        # with random sampling — produces 0.7s of audio for 11 chars instead
        # of the expected ~3s. New seed on retry usually works.
        max_attempts = int(os.environ.get("GPT_SOVITS_MAX_ATTEMPTS", "3"))
        # Minimum acceptable seconds-per-Chinese-char (English roughly /3).
        zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        non_space_chars = len([c for c in text if not c.isspace()])
        # rough lower bound: 0.10s per cjk char, 0.04s per latin char
        min_dur = max(0.6, zh_chars * 0.10 + max(0, non_space_chars - zh_chars) * 0.04)

        result_path: str | None = None
        result_dur: float = 0.0
        for attempt in range(1, max_attempts + 1):
            ap, dur = self._generate_audio_once(text)
            if ap is None:
                # network / decode failure — no point retrying same way
                return None, 0.0
            result_path, result_dur = ap, dur
            if dur >= min_dur:
                if attempt > 1:
                    print(f"[GPT-SoVITS] OK after {attempt} attempts (dur={dur:.2f}s)")
                break
            if attempt < max_attempts:
                print(f"[GPT-SoVITS] short audio dur={dur:.2f}s "
                      f"(< min {min_dur:.2f}s for {zh_chars}\u4e2d/{non_space_chars}all chars), "
                      f"retry {attempt+1}/{max_attempts}")
                try:
                    os.remove(ap)
                except Exception:
                    pass
            else:
                print(f"[GPT-SoVITS] WARN gave up after {max_attempts} attempts, "
                      f"using last dur={dur:.2f}s")

        return result_path, result_dur

    def _generate_audio_once(self, text: str) -> tuple[str | None, float]:
        lang = _lang_for_gpt_sovits(self.language, text)
        payload = self._build_payload(text, lang)
        timeout_s = float(os.environ.get("GPT_SOVITS_TIMEOUT", "90"))

        try:
            resp = requests.post(self.api_url, json=payload, timeout=timeout_s)
            resp.raise_for_status()
        except Exception as e:
            print(f"[GPT-SoVITS] request failed: {e}")
            self.word_boundaries = []
            return None, 0.0

        audio_path: str | None = None
        ctype = (resp.headers.get("Content-Type") or "").lower()

        # Case 1: raw wav/mp3 bytes.
        if "audio" in ctype or "octet-stream" in ctype:
            try:
                audio_path = self._save_temp_wav(resp.content)
            except Exception as e:
                print(f"[GPT-SoVITS] save audio bytes failed: {e}")
                return None, 0.0
        else:
            # Case 2: JSON body with path or base64.
            try:
                data = resp.json()
            except Exception as e:
                print(f"[GPT-SoVITS] unexpected response (not JSON/audio): {e}")
                return None, 0.0

            for key in ("audio_path", "output_path", "path"):
                p = data.get(key)
                if isinstance(p, str) and os.path.exists(p):
                    # Copy to temp so lifecycle is controlled by this client.
                    try:
                        with open(p, "rb") as f:
                            audio_path = self._save_temp_wav(f.read())
                    except Exception as e:
                        print(f"[GPT-SoVITS] copy audio path failed: {e}")
                    break

            if audio_path is None:
                b64 = data.get("audio")
                if isinstance(b64, str) and b64:
                    try:
                        audio_path = self._save_temp_wav(base64.b64decode(b64))
                    except Exception as e:
                        print(f"[GPT-SoVITS] decode base64 audio failed: {e}")
                        return None, 0.0

        if not audio_path or not os.path.exists(audio_path):
            print("[GPT-SoVITS] no usable audio in response")
            self.word_boundaries = []
            return None, 0.0

        duration = get_audio_duration(audio_path)
        self._audio_duration = duration
        self.word_boundaries = []
        return audio_path, duration

    def set_word_boundaries(self, boundaries: List[WordBoundary]) -> None:
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
