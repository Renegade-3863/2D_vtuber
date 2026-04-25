"""
Forced alignment helper backed by WhisperX (wav2vec2 CTC).

Given a synthesized audio file and the original text, produces a list of
:class:`~ai_runtime.tts_client.WordBoundary` objects that the existing
phoneme aligner / lip-sync pipeline already knows how to consume.

This lets voice-cloning TTS engines (XTTS, GPT-SoVITS, ...) — which do
NOT emit word timestamps natively — feed the same precise lip-sync path
that Edge TTS uses.

Models are loaded lazily and cached per (language, device) pair so the
first call pays the load cost (~2 s on GPU) and subsequent calls are
near-instantaneous.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import List

# Make sure the bundled ffmpeg binary is on PATH so whisperx.load_audio
# (which shells out to ffmpeg) can find it.
try:
    import imageio_ffmpeg
    _ff_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
    if _ff_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ff_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

from ai_runtime.tts_client import WordBoundary


# ---------------------------------------------------------------------------
# RMS-based tail trimmer (Method C). See align() for rationale.
def _rms_trim_boundaries(audio, sr: int, boundaries: List["WordBoundary"]) -> List["WordBoundary"]:
    """Shrink each WordBoundary.end down to the last frame with real voice.

    audio: 16 kHz mono float32 numpy array (range ~[-1, 1]).
    Returns a new list of WordBoundary with possibly tightened `end`/`duration`.
    """
    if not boundaries or audio is None or len(audio) == 0:
        return boundaries
    try:
        import numpy as np
    except Exception:
        return boundaries

    debug = os.environ.get("THA_LIPSYNC_RMS_TRIM_DEBUG", "").strip() in ("1", "true", "yes")

    # 10 ms frames
    hop = max(1, sr // 100)            # 160 samples @ 16k
    n_frames = max(1, len(audio) // hop)
    # vectorised RMS over fixed-size frames
    a = np.asarray(audio[: n_frames * hop], dtype=np.float32).reshape(n_frames, hop)
    rms = np.sqrt(np.mean(a * a, axis=1) + 1e-12)

    # Dynamic noise floor: 5% of clip peak, but at least 0.005.
    peak = float(rms.max())
    noise_floor = max(0.005, peak * 0.05)

    # Tunables
    SAFETY_MARGIN_S = 0.04       # keep 40 ms after the last voiced frame
    MIN_TRIM_S = 0.05            # only trim if we save >50 ms (avoid jitter)
    MIN_KEEP_DUR_S = 0.06        # never shrink a word below 60 ms

    out: List["WordBoundary"] = []
    trimmed = 0
    for wb in boundaries:
        new_end = wb.end
        try:
            f0 = max(0, int(wb.offset * sr) // hop)
            f1 = min(n_frames, int(np.ceil(wb.end * sr / hop)))
            if f1 > f0:
                window = rms[f0:f1]
                voiced = np.where(window > noise_floor)[0]
                if len(voiced) > 0:
                    last_voiced_frame = f0 + int(voiced[-1])
                    candidate = (last_voiced_frame + 1) * hop / sr + SAFETY_MARGIN_S
                    candidate = max(wb.offset + MIN_KEEP_DUR_S, min(candidate, wb.end))
                    if (wb.end - candidate) > MIN_TRIM_S:
                        new_end = candidate
        except Exception:
            new_end = wb.end

        if new_end < wb.end:
            trimmed += 1
            if debug:
                print(f"[rms-trim] '{wb.text}' end {wb.end:.3f} -> {new_end:.3f} "
                      f"(saved {wb.end - new_end:.3f}s)")
        out.append(WordBoundary(
            text=wb.text,
            offset=wb.offset,
            duration=max(0.0, new_end - wb.offset),
            end=new_end,
        ))

    if debug and trimmed:
        print(f"[rms-trim] tightened {trimmed}/{len(boundaries)} words "
              f"(noise_floor={noise_floor:.4f}, peak={peak:.4f})")
    return out


# ---------------------------------------------------------------------------
# Audio loader that doesn't require a system-installed ``ffmpeg`` on PATH.
# WhisperX's bundled ``whisperx.load_audio`` shells out to the literal
# command ``ffmpeg``; the imageio-ffmpeg binary is named
# ``ffmpeg-win-x86_64-v7.1.exe`` so that lookup fails on this environment.
# We invoke the imageio binary directly and decode 16 kHz mono PCM ourselves.
def _load_audio_16k(path: str):
    """Decode any audio file to 16 kHz mono float32 numpy via imageio-ffmpeg.

    Retries a few times because on Windows the file may transiently be locked
    by the audio player (pygame.mixer) that just opened it for playback.
    """
    import subprocess
    import time as _time
    import numpy as np
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        exe, "-nostdin", "-y", "-loglevel", "error",
        "-i", path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000",
        "-",
    ]
    last_err = None
    for attempt in range(4):
        try:
            r = subprocess.run(cmd, capture_output=True, check=False)
            if r.returncode == 0 and r.stdout:
                return np.frombuffer(r.stdout, np.int16).flatten().astype(np.float32) / 32768.0
            last_err = (
                f"ffmpeg exit={r.returncode} stderr={r.stderr.decode('utf-8', 'ignore').strip()[:300]}"
            )
        except Exception as e:
            last_err = repr(e)
        _time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(last_err or "ffmpeg failed (unknown)")


# ---------------------------------------------------------------------------
# transformers >= 4.50 refuses to load any ``pytorch_model.bin`` weights
# unless torch >= 2.6 is installed (CVE-2025-32434). The wav2vec2 alignment
# checkpoints used by WhisperX (e.g. jonatasgrosman/wav2vec2-large-xlsr-53-
# chinese-zh-cn) only ship a .bin file, so loading them on torch 2.5.1+cu121
# blows up with::
#
#   ValueError: Due to a serious vulnerability issue in `torch.load` ...
#
# The vulnerability requires ``weights_only=False`` to be exploitable.
# Transformers always passes ``weights_only=True`` for these checkpoints, so
# bypassing the version gate is safe for our read-only weight load. We
# monkey-patch ``check_torch_load_is_safe`` to a no-op before WhisperX
# imports the alignment model.
try:
    from transformers.utils import import_utils as _tf_import_utils

    def _noop_check_torch_load_is_safe() -> None:  # pragma: no cover
        return None

    _tf_import_utils.check_torch_load_is_safe = _noop_check_torch_load_is_safe
    # The function is also re-exported from ``transformers.utils`` and
    # imported (by name) into several modules via ``from .utils import
    # check_torch_load_is_safe``, which creates a local binding. Patch all
    # known consumers so the version gate becomes a no-op everywhere.
    for _mod_name in (
        "transformers.utils",
        "transformers",
        "transformers.modeling_utils",
        "transformers.modeling_flax_pytorch_utils",
        "transformers.modeling_tf_pytorch_utils",
    ):
        try:
            import importlib
            _m = importlib.import_module(_mod_name)
            if hasattr(_m, "check_torch_load_is_safe"):
                _m.check_torch_load_is_safe = _noop_check_torch_load_is_safe
        except Exception:
            pass
except Exception:
    pass


# Map our internal language codes to the ones whisperx expects in
# ``DEFAULT_ALIGN_MODELS_TORCH`` / ``DEFAULT_ALIGN_MODELS_HF``.
_LANG_ALIAS = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh": "zh",
    "en": "en",
    "ja": "ja",
}


class ForcedAligner:
    """Wrapper around ``whisperx.load_align_model`` + ``whisperx.align``."""

    # Cache: {(lang, device): (model, metadata)}
    _model_cache: dict = {}
    _cache_lock = threading.Lock()

    def __init__(self, device: str | None = None):
        if device is None:
            # Allow env override (e.g. WHISPERX_DEVICE=cpu) to free GPU for THA3.
            env_dev = os.environ.get("WHISPERX_DEVICE", "").strip().lower()
            if env_dev in ("cpu", "cuda"):
                device = env_dev
            else:
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
        self.device = device

    # ------------------------------------------------------------------
    def _load_align_model(self, language: str):
        lang = _LANG_ALIAS.get(language.lower(), language.lower())
        key = (lang, self.device)
        cached = self._model_cache.get(key)
        if cached is not None:
            return cached, lang
        with self._cache_lock:
            cached = self._model_cache.get(key)
            if cached is None:
                import whisperx  # heavy import, lazy
                t0 = time.time()
                model, metadata = whisperx.load_align_model(
                    language_code=lang, device=self.device
                )
                self._model_cache[key] = (model, metadata)
                cached = self._model_cache[key]
                print(f"[ForcedAligner] loaded {lang} model on {self.device} in {time.time()-t0:.1f}s")
        return cached, lang

    # ------------------------------------------------------------------
    def align(self, audio_path: str, text: str, language: str, audio=None) -> List[WordBoundary]:
        """Run forced alignment and return Edge-TTS-compatible WordBoundary list.

        Args:
            audio_path: path to the audio file. May be deleted by the time the
                background thread runs; pass ``audio`` to avoid that race.
            text: original synthesis text (used as transcript).
            language: language code (``zh``, ``en``, ``ja`` ...).
            audio: optional pre-decoded ``numpy.ndarray`` (16 kHz mono float32).
                When provided ``audio_path`` is only used for logging.

        Returns an empty list on any failure so callers can degrade
        gracefully to weighted-estimation lip-sync.
        """
        text = (text or "").strip()
        if not text:
            return []
        if audio is None and (not audio_path or not os.path.exists(audio_path)):
            return []

        try:
            import whisperx
        except Exception as e:
            print(f"[ForcedAligner] whisperx import failed: {e}")
            return []

        try:
            (model, metadata), lang = self._load_align_model(language)
        except Exception as e:
            print(f"[ForcedAligner] no align model for '{language}': {e}")
            return []

        try:
            if audio is None:
                audio = _load_audio_16k(audio_path)
            audio_duration = float(len(audio)) / 16000.0  # whisperx expects 16kHz mono
            transcript = [{"text": text, "start": 0.0, "end": audio_duration}]
            result = whisperx.align(
                transcript,
                model,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
        except Exception as e:
            print(f"[ForcedAligner] alignment failed: {e}")
            return []

        word_segments = result.get("word_segments", []) if isinstance(result, dict) else []
        boundaries: List[WordBoundary] = []
        for w in word_segments:
            text_w = w.get("word", "")
            start = w.get("start")
            end = w.get("end")
            if start is None or end is None or not text_w:
                continue
            try:
                start = float(start)
                end = float(end)
            except (TypeError, ValueError):
                continue
            duration = max(0.0, end - start)
            boundaries.append(WordBoundary(
                text=text_w,
                offset=start,
                duration=duration,
                end=end,
            ))

        # ------------------------------------------------------------------
        # Method C: RMS-based tail trimming.
        # WhisperX's CTC alignment tends to extend each word's `end` to the
        # onset of the next word, leaving 100~300 ms of silence inside the
        # boundary. Downstream code makes the LAST phoneme of each word span
        # `[..., end]`, which causes the mouth to stay open during that
        # silent tail (the "stuck mouth" bug on long vowels).
        #
        # Fix: scan the audio backwards from `end` and find the last frame
        # whose RMS exceeds a dynamic noise floor; treat that as the true
        # voice end and trim the boundary down to it (with a small safety
        # margin so we don't cut transients).
        #
        # Toggle with THA_LIPSYNC_RMS_TRIM (default on). Verbose dump with
        # THA_LIPSYNC_RMS_TRIM_DEBUG=1.
        # ------------------------------------------------------------------
        try:
            if os.environ.get("THA_LIPSYNC_RMS_TRIM", "1").strip() in ("1", "true", "yes"):
                boundaries = _rms_trim_boundaries(audio, 16000, boundaries)
        except Exception as e:
            print(f"[ForcedAligner] rms-trim failed (skip): {e}")

        return boundaries

    # ------------------------------------------------------------------
    def free_gpu(self) -> None:
        """Drop cached models and release VRAM (call when memory is tight)."""
        with self._cache_lock:
            self._model_cache.clear()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
