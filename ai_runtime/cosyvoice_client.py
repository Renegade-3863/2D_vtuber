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
    """修复流式 WAV 头中错误的 RIFF/data 长度字段。

    DashScope 在 streaming 模式下，第一个 chunk 内的 WAV header 已经包含了
    占位的 chunk-size / data-size 字段（通常为 0 或 0xFFFFFFFF），因为流开始
    时还不知道总字节数。直接落盘后 wave.open / pygame.mixer.Sound 都会读不到
    正确的帧数：wave 返回 dur=0 被 sanity check 静默丢弃，pygame 抛
    "Out of memory"，最终 get_audio_duration 走文件大小估算返回错误时长，
    并且音频根本播不出来。

    这里在落盘前根据真实总长回填 RIFF chunk size 和 data sub-chunk size。
    """
    if len(audio_bytes) < 44 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        return audio_bytes
    # 标准 PCM WAV: "data" 子块紧跟在 16 字节 fmt 之后, 即偏移 36
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
        # CosyVoice äº‘ç«¯ TTS æ˜¯æ— çŠ¶æ€çš„ï¼šæ¯æ¬¡ generate_audio éƒ½æ–°å»º
        # SpeechSynthesizerï¼Œæ‰€ä»¥å¤šçº¿ç¨‹å¹¶å‘è°ƒç”¨ generate_audio_full() å®‰å…¨ï¼›
        # ä½† self.word_boundaries è¿™ä¸ªå…±äº«çŠ¶æ€æ˜¯æœ‰ç«žæ€çš„ï¼Œå¹¶å‘è·¯å¾„è¯·ç”¨
        # generate_audio_full() æ‹¿è¿”å›žå€¼ã€‚

    # ---- synthesis -------------------------------------------------------

    def _save_temp(self, data: bytes, ext: str) -> str:
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, f"cosyvoice_{int(time.time() * 1000)}.{ext}")
        with open(path, "wb") as f:
            f.write(data)
        return path

    def generate_audio(self, text: str) -> tuple[str | None, float]:
        path, duration, boundaries = self._generate_full(text)
        # ç”¨é”ä¿è¯ self.word_boundaries ä¸Žæœ¬æ¬¡è¿”å›žçš„ path/duration ä¸€è‡´
        # ï¼ˆå¤šçº¿ç¨‹ä¸‹ç”Ÿäº§è€…åº”ç›´æŽ¥è°ƒç”¨ generate_audio_fullï¼‰ã€‚
        self.word_boundaries = boundaries
        return path, duration

    def _generate_full(self, text: str) -> tuple[str | None, float, List[WordBoundary]]:
        text = (text or "").strip()
        if not text:
            return None, 0.0, []

        from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat

        # SDK æŠŠ ws æ¡æ‰‹è¶…æ—¶ hard-code æˆ 5 ç§’ï¼ˆä¸”éƒ¨åˆ†è°ƒç”¨ç‚¹æ˜¾å¼ä¼  5ï¼‰ï¼Œ
        # é¦–æ¬¡è¿žæŽ¥ç»å¸¸ >5s å¤±è´¥ã€‚Monkey patch å¼ºåˆ¶å¿½ç•¥å…¥å‚ï¼Œä½¿ç”¨æˆ‘ä»¬çš„å€¼ã€‚
        try:
            _conn = SpeechSynthesizer.__dict__.get("_SpeechSynthesizer__connect")
            if _conn and not getattr(_conn, "_tha_patched", False):
                import functools
                orig = _conn
                ws_to = float(os.environ.get("THA_COSYVOICE_WS_TIMEOUT", "30"))

                @functools.wraps(orig)
                def _patched(self, timeout_seconds=ws_to):
                    # å¿½ç•¥è°ƒç”¨æ–¹ä¼ å…¥çš„å°è¶…æ—¶å€¼
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
            # downstream. v3.5-* models don't support it â€” fall back to
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
                    t_call_start = [0.0]  # ç”±å¤–éƒ¨è®¾ç½®

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
                                    # CosyVoice 流式 sentence 事件的 words 字段行为不统一:
                                    #  - 短句: 每次回「从头到当前」全量累积
                                    #  - 长句多 sentence: 每次只回当前 sentence 的 words
                                    # 直接 extend 会重复, 直接覆盖会丢早期 sentence 的字。
                                    # 这里全量收下, 在最后阶段按 begin_time 去重。
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
            # ä¼˜å…ˆç”¨ streaming_call APIã€‚å’Œ call() æ¯”ï¼š
            #   - æ˜¾å¼åŠåŒå·¥æµï¼ŒSDK å†…éƒ¨èµ° chunk-level pipelineï¼Œ
            #     é¦–å— PCM é€šå¸¸æ¯” call() å¿« 100-300ms
            #   - æ”¯æŒåŽç»­å¤šæ¬¡ streaming_call(ç»§ç»­è¿½åŠ æ–‡æœ¬)ï¼Œ
            #     è™½ç„¶è¿™é‡Œåªç”¨å•æ¬¡ï¼Œä½†ç•™å¥½æŽ¥å£ä¾¿äºŽåŽç»­ LLM token æµå¼
            #   - streaming_complete() é˜»å¡žè‡³ on_complete/on_error
            # THA_COSYVOICE_USE_LEGACY_CALL=1 å¯å›žé€€åˆ°æ—§ call()ã€‚
            use_streaming = os.environ.get(
                "THA_COSYVOICE_USE_LEGACY_CALL", "").strip().lower() not in ("1", "true", "yes")
            ret = None
            if use_streaming and callback is not None and hasattr(synthesizer, "streaming_call"):
                try:
                    synthesizer.streaming_call(text)
                    if hasattr(synthesizer, "streaming_complete"):
                        # è¯¥æ–¹æ³•æœ¬èº«ä¼šç­‰åˆ°åˆæˆç»“æŸï¼ˆä¹Ÿæ˜¯ on_completeï¼‰
                        synthesizer.streaming_complete()
                except Exception as e:
                    # å…¼å®¹æ—§ SDK / ç½‘ç»œå¼‚å¸¸æ—¶é™çº§
                    print(f"[CosyVoice] streaming_call failed ({e}), fallback to call()")
                    ret = synthesizer.call(text)
            else:
                ret = synthesizer.call(text)
            # æœ‰ callback æ—¶ SDK ç«‹å³è¿”å›žï¼Œéœ€ç­‰ on_complete/on_error è§¦å‘
            if callback is not None:
                wait_to = float(os.environ.get("THA_COSYVOICE_SYNTH_TIMEOUT", "60"))
                if not done_evt.wait(timeout=wait_to):
                    print(f"[CosyVoice] synth wait timeout ({wait_to}s)")
                # æ‰“å°åˆ†é˜¶æ®µè€—æ—¶
                def _ms(t): return f"{(t - t_call_start_v) * 1000:.0f}ms" if t else "n/a"
                print(f"[CosyVoice] timing: open={_ms(timing['open'])} "
                      f"task-started={_ms(timing['started'])} "
                      f"first-audio={_ms(timing['first_audio'])} "
                      f"complete={_ms(timing['complete'])}")
            audio_bytes = bytes(captured_audio) if callback is not None else ret
            if not audio_bytes:
                # æŠŠæœåŠ¡ç«¯æœ€åŽä¸€æ¡æŠ¥æ–‡æ‰“å‡ºæ¥ä¾¿äºŽè¯Šæ–­
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
            # 先按 begin_time 去重 (sentence 事件可能重复推送同一个字)
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
        """çº¿ç¨‹å®‰å…¨ç‰ˆæœ¬ï¼šç›´æŽ¥è¿”å›ž (path, duration, boundaries)ã€‚
        ç”Ÿäº§è€…å¹¶å‘é¢„åˆæˆå¤šæ®µæ—¶ä½¿ç”¨ï¼Œä¸ä¾èµ–å…±äº«çš„ self.word_boundariesã€‚"""
        return self._generate_full(text)

    # ---- compatibility shim ---------------------------------------------

    def set_word_boundaries(self, boundaries: List[WordBoundary]) -> None:
        self.word_boundaries = boundaries or []

    def get_playback_position(self) -> float:
        if not self._is_playing:
            return 0.0
        return self._audio_position

    # ---- gapless æ’­æ”¾ï¼šSound + Channel.queue() ----------------------
    # ä¸ç”¨ pygame.mixer.musicï¼ŒåŽŸå› ï¼š
    #   1) æ¯æ®µéƒ½ load() ä¼šæœ‰ ~30-50ms å¯åŠ¨é—´éš™ï¼ˆå¬æ„Ÿé¡¿æŒ«ï¼‰
    #   2) mp3/wav è§£ç  warm-up åƒæŽ‰å‰ ~20ms å¤´éŸ³
    #   3) get_busy() è½¬ False æ—¶ç¡¬ä»¶ buffer è¿˜æ²¡æŽ’ç©º â†’ å°¾è¢«åˆ‡
    # Sound é¢„è§£ç ï¼ˆæ—  warm-upï¼‰ + Channel.queue() åœ¨æœ¬æ®µçœŸæ­£
    # ç»“æŸé‚£ä¸€åˆ»æ— ç¼æŽ¥ä¸Šä¸‹ä¸€æ®µã€‚
    #
    # play_audio_async æ˜¯éžé˜»å¡žçš„ï¼šè¿”å›ž (start_evt, end_evt)ã€‚
    # è°ƒç”¨æ–¹åº”åœ¨ä¸Šæ®µ end_evt è§¦å‘ä¹‹å‰æäº¤ä¸‹æ®µï¼Œpygame ä¼šè‡ªåŠ¨ queue()ï¼Œ
    # è¿™æ ·ä¸Šä¸‹ä¸¤æ®µä¹‹é—´æ— ä»»ä½•é—´éš™ã€‚
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
        """段间换气静音的原始字节（按 mixer 实际格式）。
        默认 80ms，THA_COSYVOICE_GAP_MS=0 禁用。
        返回 (bytes, gap_sec)。
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
        """éžé˜»å¡žï¼Œè¿”å›ž (start_evt, end_evt)ã€‚
        - start_evt: æœ¬æ®µè¯­éŸ³çœŸæ­£å¼€å§‹æ’­æ”¾ï¼ˆå£åž‹/çŠ¶æ€åº”åœ¨æ­¤åˆ‡æ¢ï¼‰
        - end_evt:   æœ¬æ®µè¯­éŸ³ç»“æŸï¼ˆä¸å« trailing gapï¼‰
        æ®µé—´æ¢æ°”ï¼šåœ¨æ¯æ®µè¯­éŸ³å‰ prepend ä¸€æ®µé™éŸ³å¹¶åˆå¹¶ä¸ºå•ä¸ª Sound
        æ’­æ”¾ï¼Œè¿™æ · channel é˜Ÿåˆ—åªå ä¸€æ§½ï¼Œä¸ä¼šè¢«ä¸‹ä¸€æ®µ queue è¦†ç›–ã€‚
        é¦–æ®µ (was_busy=False) ä¸åŠ  leading silenceï¼Œé¿å…ç”¨æˆ·æ„ŸçŸ¥å»¶è¿Ÿã€‚
        """
        start_evt = threading.Event()
        end_evt = threading.Event()
        if not audio_path or not os.path.exists(audio_path):
            start_evt.set()
            end_evt.set()
            return start_evt, end_evt
        # 必须先确保 mixer 已初始化, 否则 pygame.mixer.Sound() 会抛 "mixer not initialized"
        ch = self._ensure_channel()
        try:
            voice_sound = pygame.mixer.Sound(audio_path)
        except Exception as e:
            print(f"[CosyVoice] Sound load failed: {e}")
            start_evt.set()
            end_evt.set()
            return start_evt, end_evt

        # 取上一段的 end_evt 用于估算本段开始时间
        prev_end_evt = getattr(self, "_last_end_evt", None)
        prev_end_etime = float(getattr(self, "_last_end_etime", 0.0))
        was_busy = ch.get_busy()
        voice_dur = voice_sound.get_length()

        # ä»…åœ¨ã€Œä¸Šæ®µè¿˜åœ¨æ’­ã€æ—¶ç»™æœ¬æ®µå‰é¢åŠ  silence ä½œä¸ºæ¢æ°”
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

        # æ—¶é—´é¢„ç®—ï¼š
        #   padded sound çš„å¼€å§‹ = prev_end_etimeï¼ˆä¸Šæ®µè¯­éŸ³ç»“æŸçž¬é—´ï¼‰
        #   æœ¬æ®µè¯­éŸ³çœŸæ­£å¼€å§‹    = padded å¼€å§‹ + leading_gap
        #   æœ¬æ®µè¯­éŸ³ç»“æŸ        = è¯­éŸ³å¼€å§‹ + voice_dur
        if was_busy and prev_end_etime > 0:
            sound_start_etime = prev_end_etime
        else:
            sound_start_etime = time.time()
        seg_start_etime = sound_start_etime + leading_gap
        seg_end_etime = seg_start_etime + voice_dur

        # ç•™ç»™ä¸‹ä¸€æ®µï¼šä¸‹ä¸€æ®µ padded sound åœ¨æœ¬æ®µè¯­éŸ³ç»“æŸçž¬é—´å¼€å§‹æ’­
        self._last_end_evt = end_evt
        self._last_end_etime = seg_end_etime
        # é˜² GCï¼ˆChannel å†…éƒ¨å¯¹ Sound ä¼¼ä¹ŽåªæŒå¼±å¼•ç”¨ï¼‰
        keep_alive = getattr(self, "_keep_alive_sounds", None)
        if keep_alive is None:
            keep_alive = []
            self._keep_alive_sounds = keep_alive
        keep_alive.append(play_sound)
        if len(keep_alive) > 8:
            del keep_alive[:-8]

        def _watch():
            # ç­‰åˆ°æœ¬æ®µè¯­éŸ³çœŸæ­£å¼€å§‹ï¼ˆleading silence æ’­å®Œï¼‰
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
        """é˜»å¡žè‡³æœ€è¿‘ä¸€æ®µæ’­å®Œã€‚Channel ä»å¯èƒ½åœ¨æ’­ queue åŽç»­æ®µï¼Œ
        ä½†è‡³å°‘ wait è°ƒç”¨æ–¹æäº¤çš„æœ€åŽä¸€æ®µå·²ç»“æŸã€‚"""
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
