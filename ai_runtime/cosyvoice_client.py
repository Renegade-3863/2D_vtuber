"""
Aliyun DashScope CosyVoice (cloud) TTS client.

Public interface mirrors GPTSoVITSClient / XTTSClient so it can be plugged
into the existing forced-alignment + lip-sync pipeline with no other changes:

    - generate_audio(text) -> (audio_path, duration)
    - play_audio_async(audio_path)
    - wait_playback()
    - speak_async(text)
    - get_playback_position()
    - estimate_duration(text)
    - self.word_boundaries (filled by forced aligner externally)

Environment variables:
    DASHSCOPE_API_KEY     - your aliyun bailian / dashscope API key (sk-...)
    THA_COSYVOICE_VOICE   - voice id, e.g. "longxiaochun" (preset) or your
                            cloned id like "cosyvoice-clone-v2-xxxxxxxx"
    THA_COSYVOICE_MODEL   - model id, default "cosyvoice-v2"
    THA_COSYVOICE_FORMAT  - "wav" (default) or "mp3"
    THA_COSYVOICE_SR      - output sample rate, default 22050

Why CosyVoice 2 (cloud) instead of local GPT-SoVITS:
    - flow-matching + explicit duration predictor: no early-EOS instability
    - native streaming, sub-500ms first packet
    - voice cloning quality on par or better than GPT-SoVITS
    - frees ~5GB VRAM locally (only THA3 + WhisperX stay on GPU)
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from typing import List

import pygame

from ai_runtime.tts_client import WordBoundary, get_audio_duration


def _detect_lang(text: str) -> str:
    cjk = sum(1 for ch in (text or "") if "\u4e00" <= ch <= "\u9fff")
    return "zh" if cjk >= max(1, len(text) // 6) else "en"


def _fix_wav_header(audio_bytes: bytes) -> bytes:
    """Fix invalid RIFF/data size fields in streamed WAV headers.

    In DashScope streaming mode, the first audio chunk may already contain a
    WAV header with placeholder chunk-size / data-size fields, usually 0 or
    0xFFFFFFFF, because the final stream length is not known yet. If those
    bytes are written directly to disk, wave.open / pygame.mixer.Sound may not
    be able to read the correct frame count: wave can report dur=0, pygame may
    raise "Out of memory", and get_audio_duration may fall back to a wrong
    file-size estimate. In that case, audio playback can fail completely.

    Before saving the file, patch the RIFF chunk size and data sub-chunk size
    using the actual byte length.
    """
    if len(audio_bytes) < 44 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        return audio_bytes
    # Standard PCM WAV: the "data" chunk follows the 16-byte fmt chunk at offset 36.
    if audio_bytes[36:40] != b"data":
        return audio_bytes
    ba = bytearray(audio_bytes)
    total = len(ba)
    ba[4:8] = (total - 8).to_bytes(4, "little")
    ba[40:44] = (total - 44).to_bytes(4, "little")
    return bytes(ba)


class CosyVoiceClient:
    """Cloud CosyVoice 2 client via Aliyun DashScope SDK."""

    def __init__(
        self,
        voice: str | None = None,
        model: str | None = None,
        language: str = "auto",
        api_key: str | None = None,
    ):
        try:
            import dashscope  # noqa: F401
            from dashscope.audio.tts_v2 import SpeechSynthesizer  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "dashscope SDK not installed. Run: pip install dashscope"
            ) from e

        import dashscope
        key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY env var not set. Get one at "
                "https://bailian.console.aliyun.com/ -> API-Key."
            )
        dashscope.api_key = key

        self.voice = voice or os.environ.get("THA_COSYVOICE_VOICE", "longxiaochun")
        self.model = model or os.environ.get("THA_COSYVOICE_MODEL", "cosyvoice-v2")
        self.language = language
        self.audio_format = os.environ.get("THA_COSYVOICE_FORMAT", "wav").lower()
        self.sample_rate = int(os.environ.get("THA_COSYVOICE_SR", "22050"))

        self.word_boundaries: List[WordBoundary] = []
        self._playback_thread: threading.Thread | None = None
        self._audio_duration: float = 0.0
        self._playback_start_time: float = 0.0
        self._is_playing: bool = False
        self._audio_position: float = 0.0
        # CosyVoice cloud TTS is stateless: generate_audio creates a new
        # SpeechSynthesizer for every request, so concurrent calls to
        # generate_audio_full() are safe. self.word_boundaries is shared state,
        # though, so concurrent producer paths should use generate_audio_full()
        # and consume the returned boundaries directly.

    # ---- synthesis -------------------------------------------------------

    def _save_temp(self, data: bytes, ext: str) -> str:
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, f"cosyvoice_{int(time.time() * 1000)}.{ext}")
        with open(path, "wb") as f:
            f.write(data)
        return path

    def generate_audio(self, text: str) -> tuple[str | None, float]:
        path, duration, boundaries = self._generate_full(text)
        # Keep self.word_boundaries aligned with this path/duration pair.
        # Multi-threaded producers should call generate_audio_full() directly.
        self.word_boundaries = boundaries
        return path, duration

    def _generate_full(self, text: str) -> tuple[str | None, float, List[WordBoundary]]:
        text = (text or "").strip()
        if not text:
            return None, 0.0, []

        from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat

        # The SDK hard-codes a 5s WebSocket handshake timeout in some paths,
        # and the first connection can often take longer than that. Monkey
        # patch the private connector to ignore the caller-provided timeout and
        # use our configurable value instead.
        try:
            _conn = SpeechSynthesizer.__dict__.get("_SpeechSynthesizer__connect")
            if _conn and not getattr(_conn, "_tha_patched", False):
                import functools
                orig = _conn
                ws_to = float(os.environ.get("THA_COSYVOICE_WS_TIMEOUT", "30"))

                @functools.wraps(orig)
                def _patched(self, timeout_seconds=ws_to):
                    # Ignore too-small timeout values supplied by the SDK call path.
                    return orig(self, timeout_seconds=ws_to)
                _patched._tha_patched = True
                setattr(SpeechSynthesizer, "_SpeechSynthesizer__connect", _patched)
        except Exception:
            pass

        # Map our format string to AudioFormat enum if available; otherwise
        # fall back to passing the raw string (older SDK versions).
        fmt_map = {
            ("wav", 22050): "WAV_22050HZ_MONO_16BIT",
            ("wav", 16000): "WAV_16000HZ_MONO_16BIT",
            ("wav", 24000): "WAV_24000HZ_MONO_16BIT",
            ("mp3", 22050): "MP3_22050HZ_MONO_256KBPS",
            ("mp3", 24000): "MP3_24000HZ_MONO_256KBPS",
        }
        fmt_attr = fmt_map.get((self.audio_format, self.sample_rate),
                               "WAV_22050HZ_MONO_16BIT")
        try:
            audio_format = getattr(AudioFormat, fmt_attr)
        except AttributeError:
            audio_format = fmt_attr  # plain string fallback

        try:
            # Native word timestamp: supported by cosyvoice-v3-flash / v3-plus
            # / v2 (cloned voices, plus a subset of marked system voices).
            # For these models we enable it automatically and skip WhisperX
            # downstream. v3.5-* models don't support it; fall back to
            # WhisperX path. Force-disable with THA_COSYVOICE_WORD_TS=0.
            additional_params: dict = {}
            captured_words: list[dict] = []
            captured_audio = bytearray()
            callback = None
            ts_capable = self.model in (
                "cosyvoice-v3-flash", "cosyvoice-v3-plus", "cosyvoice-v2"
            )
            env_flag = os.environ.get("THA_COSYVOICE_WORD_TS", "").strip().lower()
            if env_flag in ("0", "false", "no"):
                want_ts = False
            elif env_flag in ("1", "true", "yes"):
                want_ts = True
            else:
                want_ts = ts_capable
            if want_ts:
                additional_params["word_timestamp_enabled"] = True
                try:
                    from dashscope.audio.tts_v2 import ResultCallback  # type: ignore

                    done_evt = threading.Event()
                    timing = {"open": None, "started": None, "first_audio": None, "complete": None}
                    t_call_start = [0.0]  # Reserved for external timing updates.

                    class _TSCallback(ResultCallback):
                        def on_open(self):
                            timing["open"] = time.time()
                        def on_complete(self):
                            timing["complete"] = time.time()
                            done_evt.set()
                        def on_close(self):
                            done_evt.set()
                        def on_error(self, message):
                            print(f"[CosyVoice] callback error: {message}")
                            done_evt.set()
                        def on_event(self, message):
                            try:
                                import json
                                msg = json.loads(message) if isinstance(message, str) else message
                                evt = (msg or {}).get("header", {}).get("event")
                                if evt == "task-started" and timing["started"] is None:
                                    timing["started"] = time.time()
                                words = (msg or {}).get("payload", {}).get("output", {}).get("sentence", {}).get("words")
                                if words:
                                    # CosyVoice streaming sentence events are
                                    # inconsistent for the words field:
                                    #  - short text: each event may return the
                                    #    full cumulative list from the start
                                    #  - long multi-sentence text: each event
                                    #    may return only the current sentence
                                    # Appending directly creates duplicates,
                                    # but replacing directly can drop earlier
                                    # sentences. Collect all events, then
                                    # deduplicate by begin_time at the end.
                                    captured_words.extend(words)
                            except Exception:
                                pass
                        def on_data(self, data: bytes):
                            if data:
                                if timing["first_audio"] is None:
                                    timing["first_audio"] = time.time()
                                captured_audio.extend(data)

                    callback = _TSCallback()
                except Exception as e:
                    print(f"[CosyVoice] cannot init word-timestamp callback: {e}")
                    callback = None
                    want_ts = False

            kwargs = dict(model=self.model, voice=self.voice, format=audio_format)
            if additional_params:
                kwargs["additional_params"] = additional_params
            if callback is not None:
                kwargs["callback"] = callback

            synthesizer = SpeechSynthesizer(**kwargs)
            t_call_start_v = time.time()
            # Prefer the streaming_call API over call():
            #   - It uses an explicit half-duplex stream and the SDK can run a
            #     chunk-level pipeline internally, so the first PCM chunk is
            #     usually 100-300ms faster than call().
            #   - It supports multiple future streaming_call() invocations to
            #     append more text. We only use one call here, but keeping this
            #     path makes later LLM token streaming easier.
            #   - streaming_complete() blocks until on_complete/on_error.
            # Set THA_COSYVOICE_USE_LEGACY_CALL=1 to fall back to call().
            use_streaming = os.environ.get(
                "THA_COSYVOICE_USE_LEGACY_CALL", "").strip().lower() not in ("1", "true", "yes")
            ret = None
            if use_streaming and callback is not None and hasattr(synthesizer, "streaming_call"):
                try:
                    synthesizer.streaming_call(text)
                    if hasattr(synthesizer, "streaming_complete"):
                        # This waits for synthesis completion, i.e. on_complete.
                        synthesizer.streaming_complete()
                except Exception as e:
                    # Fallback for older SDKs or transient network errors.
                    print(f"[CosyVoice] streaming_call failed ({e}), fallback to call()")
                    ret = synthesizer.call(text)
            else:
                ret = synthesizer.call(text)
            # With callbacks, the SDK may return immediately; wait for completion.
            if callback is not None:
                wait_to = float(os.environ.get("THA_COSYVOICE_SYNTH_TIMEOUT", "60"))
                if not done_evt.wait(timeout=wait_to):
                    print(f"[CosyVoice] synth wait timeout ({wait_to}s)")
                # Print phase timings for latency debugging.
                def _ms(t): return f"{(t - t_call_start_v) * 1000:.0f}ms" if t else "n/a"
                print(f"[CosyVoice] timing: open={_ms(timing['open'])} "
                      f"task-started={_ms(timing['started'])} "
                      f"first-audio={_ms(timing['first_audio'])} "
                      f"complete={_ms(timing['complete'])}")
            audio_bytes = bytes(captured_audio) if callback is not None else ret
            if not audio_bytes:
                # Print the final server response to aid diagnosis.
                try:
                    resp = synthesizer.get_response()
                    print(f"[CosyVoice] empty audio. server response: {resp}")
                except Exception:
                    pass
        except Exception as e:
            print(f"[CosyVoice] synth failed: {e}")
            return None, 0.0, []

        if not audio_bytes:
            print("[CosyVoice] empty audio response")
            return None, 0.0, []

        ext = "wav" if self.audio_format == "wav" else "mp3"
        if ext == "wav":
            audio_bytes = _fix_wav_header(audio_bytes)
        try:
            path = self._save_temp(audio_bytes, ext)
        except Exception as e:
            print(f"[CosyVoice] save audio failed: {e}")
            return None, 0.0, []

        duration = get_audio_duration(path)
        self._audio_duration = duration

        # Convert DashScope words -> WordBoundary list. DashScope returns
        # begin_time/end_time in milliseconds. Empty list means timestamp
        # unsupported / disabled, downstream will fall back to WhisperX.
        boundaries: List[WordBoundary] = []
        if want_ts and captured_words:
            # Deduplicate by begin_time; sentence events may repeat the same word.
            seen_keys: set = set()
            unique_words: list[dict] = []
            for w in captured_words:
                try:
                    key = (float(w.get("begin_time", -1)), (w.get("text") or "").strip())
                except (TypeError, ValueError):
                    continue
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                unique_words.append(w)
            unique_words.sort(key=lambda x: float(x.get("begin_time", 0)))
            for w in unique_words:
                try:
                    text_w = (w.get("text") or "").strip()
                    if not text_w:
                        continue
                    bt = float(w.get("begin_time", 0)) / 1000.0
                    et = float(w.get("end_time", 0)) / 1000.0
                    if et <= bt:
                        continue
                    boundaries.append(WordBoundary(
                        text=text_w, offset=bt, duration=et - bt, end=et,
                    ))
                except (TypeError, ValueError):
                    continue
            if boundaries:
                print(f"[CosyVoice] word_timestamp: {len(boundaries)} words, "
                      f"first='{boundaries[0].text}'@{boundaries[0].offset:.2f}s "
                      f"last='{boundaries[-1].text}'@{boundaries[-1].end:.2f}s")
            else:
                print(f"[CosyVoice] word_timestamp: requested but got 0 usable words "
                      f"(model={self.model}, voice={self.voice}). "
                      f"Falling back to WhisperX.")
        return path, duration, boundaries

    def generate_audio_full(self, text: str) -> tuple[str | None, float, List[WordBoundary]]:
        """Thread-safe synthesis API returning (path, duration, boundaries).

        Use this when producer threads pre-synthesize multiple segments in
        parallel, so callers do not rely on the shared self.word_boundaries.
        """
        return self._generate_full(text)

    # ---- compatibility shim ---------------------------------------------

    def set_word_boundaries(self, boundaries: List[WordBoundary]) -> None:
        self.word_boundaries = boundaries or []

    def get_playback_position(self) -> float:
        if not self._is_playing:
            return 0.0
        return self._audio_position

    # ---- gapless playback: Sound + Channel.queue() --------------------
    # Avoid pygame.mixer.music because:
    #   1) load() for every segment can add a ~30-50ms startup gap.
    #   2) mp3/wav decode warm-up can eat the first ~20ms of audio.
    #   3) get_busy() can turn False before the hardware buffer is drained,
    #      which may cut the tail.
    # Sound pre-decodes the segment, and Channel.queue() starts the next
    # segment exactly when the current one finishes.
    #
    # play_audio_async is non-blocking and returns (start_evt, end_evt).
    # Submit the next segment before the previous end_evt fires; pygame will
    # queue it automatically, leaving no gap between adjacent segments.
    def _ensure_channel(self):
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init(frequency=self.sample_rate,
                                  size=-16, channels=1, buffer=512)
            except pygame.error:
                pygame.mixer.init()
        ch = getattr(self, "_channel", None)
        if ch is None:
            ch = pygame.mixer.Channel(0)
            self._channel = ch
        return ch

    def _get_gap_bytes(self):
        """Build raw silence bytes for the breath gap between segments.

        The bytes are generated using the active mixer format. The default gap
        is 80ms; set THA_COSYVOICE_GAP_MS=0 to disable it.
        Returns (bytes, gap_sec).
        """
        gap_ms = int(os.environ.get("THA_COSYVOICE_GAP_MS", "80"))
        if gap_ms <= 0:
            return b"", 0.0
        cached = getattr(self, "_silence_bytes", None)
        if cached is not None and getattr(self, "_silence_ms", -1) == gap_ms:
            return cached, gap_ms / 1000.0
        try:
            init = pygame.mixer.get_init()  # (freq, size, channels)
            if not init:
                return b"", 0.0
            freq, size, channels = init
            n_frames = max(1, int(freq * gap_ms / 1000))
            bytes_per_sample = abs(size) // 8
            buf = b"\x00" * (n_frames * bytes_per_sample * channels)
            self._silence_bytes = buf
            self._silence_ms = gap_ms
            return buf, gap_ms / 1000.0
        except Exception as e:
            print(f"[CosyVoice] silence bytes build failed: {e}")
            return b"", 0.0

    def play_audio_async(self, audio_path: str):
        """Non-blocking playback that returns (start_evt, end_evt).

        - start_evt: when this segment's voice audio actually starts; lip-sync
          and state transitions should switch at this point.
        - end_evt: when this segment's voice audio ends, excluding trailing gap.

        For breath gaps between segments, prepend silence to each queued segment
        and merge it into a single Sound object. This uses only one channel queue
        slot and prevents the next segment from overwriting the silence. The
        first segment (was_busy=False) skips leading silence to avoid perceived
        latency.
        """
        start_evt = threading.Event()
        end_evt = threading.Event()
        if not audio_path or not os.path.exists(audio_path):
            start_evt.set()
            end_evt.set()
            return start_evt, end_evt
        # Ensure the mixer is initialized before pygame.mixer.Sound().
        ch = self._ensure_channel()
        try:
            voice_sound = pygame.mixer.Sound(audio_path)
        except Exception as e:
            print(f"[CosyVoice] Sound load failed: {e}")
            start_evt.set()
            end_evt.set()
            return start_evt, end_evt

        # Use the previous segment end time to estimate this segment's start.
        prev_end_evt = getattr(self, "_last_end_evt", None)
        prev_end_etime = float(getattr(self, "_last_end_etime", 0.0))
        was_busy = ch.get_busy()
        voice_dur = voice_sound.get_length()

        # Add leading silence as a breath gap only while another segment is playing.
        leading_gap = 0.0
        play_sound = voice_sound
        if was_busy:
            silence_bytes, gap_sec = self._get_gap_bytes()
            if silence_bytes and gap_sec > 0:
                try:
                    voice_bytes = voice_sound.get_raw()
                    play_sound = pygame.mixer.Sound(
                        buffer=silence_bytes + voice_bytes
                    )
                    leading_gap = gap_sec
                except Exception as e:
                    print(f"[CosyVoice] padded sound build failed: {e}")
                    play_sound = voice_sound

        if was_busy:
            ch.queue(play_sound)
        else:
            ch.play(play_sound)

        # Predict playback timing:
        #   padded sound start = previous segment voice end time
        #   this voice start   = padded sound start + leading_gap
        #   this voice end     = this voice start + voice_dur
        if was_busy and prev_end_etime > 0:
            sound_start_etime = prev_end_etime
        else:
            sound_start_etime = time.time()
        seg_start_etime = sound_start_etime + leading_gap
        seg_end_etime = seg_start_etime + voice_dur

        # Store timing for the next segment; its padded sound starts exactly
        # when this segment's voice audio ends.
        self._last_end_evt = end_evt
        self._last_end_etime = seg_end_etime
        # Keep Sound objects alive; Channel appears to hold weak references.
        keep_alive = getattr(self, "_keep_alive_sounds", None)
        if keep_alive is None:
            keep_alive = []
            self._keep_alive_sounds = keep_alive
        keep_alive.append(play_sound)
        if len(keep_alive) > 8:
            del keep_alive[:-8]

        def _watch():
            # Wait until this segment's voice audio starts, after leading silence.
            wait_t = max(0.0, seg_start_etime - time.time())
            if wait_t > 0:
                time.sleep(wait_t)
            self._is_playing = True
            self._audio_position = 0.0
            self._playback_start_time = time.time()
            self._audio_duration = voice_dur
            start_evt.set()
            while True:
                elapsed = time.time() - self._playback_start_time
                self._audio_position = min(elapsed, voice_dur)
                if elapsed >= voice_dur:
                    break
                time.sleep(0.02)
            self._is_playing = False
            self._audio_position = voice_dur
            end_evt.set()

        th = threading.Thread(target=_watch, daemon=True)
        self._playback_thread = th
        th.start()
        return start_evt, end_evt

    def wait_playback(self) -> None:
        """Block until the most recently submitted segment has finished.

        The channel may still be playing later queued segments, but the last
        segment submitted by this caller has at least ended.
        """
        evt = getattr(self, "_last_end_evt", None)
        if evt is not None:
            evt.wait(timeout=30.0)
        ch = getattr(self, "_channel", None)
        if ch is not None:
            t0 = time.time()
            while ch.get_busy() and time.time() - t0 < 1.0:
                time.sleep(0.01)

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
