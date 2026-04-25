import asyncio
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import edge_tts
import pygame

from ai_runtime.config_api import cfg

# ── 可选音色目录 ──────────────────────────────────────────────
# 键名用于 /voice 命令选择，值为 Edge TTS voice ID
VOICE_CATALOG: dict[str, dict] = {
    # ── 中文女声 ──
    "xiaoxiao":   {"id": "zh-CN-XiaoxiaoNeural",            "lang": "zh", "desc": "晓晓 · 温柔女声（默认）"},
    "xiaoyi":     {"id": "zh-CN-XiaoyiNeural",              "lang": "zh", "desc": "晓伊 · 活泼女声"},
    "xiaobei":    {"id": "zh-CN-liaoning-XiaobeiNeural",    "lang": "zh", "desc": "小北 · 东北女声"},
    "xiaoni":     {"id": "zh-CN-shaanxi-XiaoniNeural",      "lang": "zh", "desc": "小妮 · 陕西女声"},
    # ── 中文男声 ──
    "yunxi":      {"id": "zh-CN-YunxiNeural",      "lang": "zh", "desc": "云希 · 阳光男声"},
    "yunjian":    {"id": "zh-CN-YunjianNeural",    "lang": "zh", "desc": "云健 · 沉稳男声"},
    "yunxia":     {"id": "zh-CN-YunxiaNeural",     "lang": "zh", "desc": "云夏 · 少年男声"},
    "yunyang":    {"id": "zh-CN-YunyangNeural",    "lang": "zh", "desc": "云扬 · 播音男声"},
    # ── 英文女声 ──
    "jenny":      {"id": "en-US-JennyNeural",      "lang": "en", "desc": "Jenny · Friendly female (default)"},
    "aria":       {"id": "en-US-AriaNeural",       "lang": "en", "desc": "Aria · Expressive female"},
    "ava":        {"id": "en-US-AvaNeural",        "lang": "en", "desc": "Ava · Natural female"},
    "emma":       {"id": "en-US-EmmaNeural",       "lang": "en", "desc": "Emma · Warm female"},
    "michelle":   {"id": "en-US-MichelleNeural",   "lang": "en", "desc": "Michelle · Clear female"},
    "ana":        {"id": "en-US-AnaNeural",        "lang": "en", "desc": "Ana · Young female"},
    # ── 英文男声 ──
    "guy":        {"id": "en-US-GuyNeural",        "lang": "en", "desc": "Guy · Mature male"},
    "andrew":     {"id": "en-US-AndrewNeural",     "lang": "en", "desc": "Andrew · Warm male"},
    "brian":      {"id": "en-US-BrianNeural",      "lang": "en", "desc": "Brian · Natural male"},
    "christopher":{"id": "en-US-ChristopherNeural","lang": "en", "desc": "Christopher · Clear male"},
    "eric":       {"id": "en-US-EricNeural",       "lang": "en", "desc": "Eric · Calm male"},
    "roger":      {"id": "en-US-RogerNeural",      "lang": "en", "desc": "Roger · Deep male"},
    "steffan":    {"id": "en-US-SteffanNeural",    "lang": "en", "desc": "Steffan · Friendly male"},
    # ── 日文 ──
    "nanami":     {"id": "ja-JP-NanamiNeural",     "lang": "ja", "desc": "七海 · Japanese female"},
    "keita":      {"id": "ja-JP-KeitaNeural",      "lang": "ja", "desc": "圭太 · Japanese male"},
}

# 每种语言的默认音色键
DEFAULT_VOICE_KEY: dict[str, str] = {
    "zh": "xiaoxiao",
    "en": "jenny",
    "ja": "nanami",
}


def list_voices(lang: str | None = None) -> str:
    """返回格式化的音色列表字符串，可按语言筛选"""
    lines = []
    for key, info in VOICE_CATALOG.items():
        if lang and info["lang"] != lang:
            continue
        lines.append(f"  {key:<12s} {info['desc']}")
    return "\n".join(lines) if lines else "  (no voices found)"


def resolve_voice(key_or_id: str, fallback_lang: str = "zh") -> tuple[str, str]:
    """
    将音色键名或完整 ID 解析为 (voice_id, lang)。
    如果是已知键名，从目录查找；否则当作原始 voice ID 直接使用。
    """
    low = key_or_id.strip().lower()
    if low in VOICE_CATALOG:
        entry = VOICE_CATALOG[low]
        return entry["id"], entry["lang"]
    # 当作完整 ID（例如 "zh-TW-HsiaoChenNeural"）
    if "-" in key_or_id and "Neural" in key_or_id:
        lang = "zh" if key_or_id.startswith("zh") else "en"
        return key_or_id, lang
    # 无法识别，返回该语言的默认
    default_key = DEFAULT_VOICE_KEY.get(fallback_lang, "xiaoxiao")
    entry = VOICE_CATALOG[default_key]
    return entry["id"], entry["lang"]


@dataclass
class WordBoundary:
    """TTS 引擎返回的词边界信息（精确时间戳）"""
    text: str          # 对应的文字
    offset: float      # 开始时间（秒）
    duration: float    # 持续时间（秒）
    end: float         # 结束时间（秒）


def get_audio_duration(audio_path: str) -> float:
    """
    获取音频文件实际时长（秒）

    .wav -> 用标准库 wave 精确读取
    .mp3 -> 用 mutagen
    其它 -> 文件大小估算（仅 MP3 假设）
    """
    ext = os.path.splitext(audio_path)[1].lower()

    # WAV: 标准库精确读取
    if ext == ".wav":
        try:
            import wave
            with wave.open(audio_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    dur = frames / float(rate)
                    # Sanity check: a single TTS reply shouldn't exceed 5min.
                    # Some TTS providers return non-standard wav containers
                    # (e.g. with embedded mp3) where wave parses garbage.
                    if 0.05 <= dur <= 300.0:
                        return dur
        except Exception as e:
            print(f"[TTS] wave read failed: {e}")

    # MP3: mutagen
    if ext == ".mp3":
        try:
            from mutagen.mp3 import MP3
            audio = MP3(audio_path)
            dur = float(audio.info.length)
            if 0.05 <= dur <= 300.0:
                return dur
        except ImportError:
            pass
        except Exception as e:
            print(f"[TTS] mutagen failed: {e}")


    # 通用降级：用 pygame 加载（支持 wav/mp3/ogg 任意格式，不依赖文件头）
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        snd = pygame.mixer.Sound(audio_path)
        dur = float(snd.get_length())
        if 0.05 <= dur <= 300.0:
            return dur
    except Exception as e:
        print(f"[TTS] pygame Sound length failed: {e}")

    # 最后兜底：文件大小估算
    try:
        size_bytes = os.path.getsize(audio_path)
        return size_bytes / (16 * 1024)
    except:
        return 1.0


class TTSClient:
    """TTS 客户端（支持音素对齐）"""

    def __init__(self, voice: str = None, language: str = "auto"):
        """
        Args:
            voice: Azure TTS 语音名称（可选，如果为 None 则根据语言自动选择）
            language: 'auto' | 'en' | 'zh'
        """
        # 根据语言自动选择语音
        if voice is None:
            if language == 'zh':
                self.voice = "zh-CN-XiaoxiaoNeural"  # 中文女声
            else:
                self.voice = "en-US-JennyNeural"  # 英文女声
        else:
            self.voice = voice
        
        self.language = language
        self.word_boundaries: List[WordBoundary] = []  # 词边界时间戳
        self._playback_thread: threading.Thread | None = None
        self._stop_playback = False
        self._audio_duration: float = 0.0  # 实际音频时长
        self._playback_start_time: float = 0.0  # 实际播放开始时间
        self._is_playing: bool = False  # 是否正在播放
        self._audio_position: float = 0.0  # 当前播放位置
        
    def get_playback_position(self) -> float:
        """
        获取当前音频播放位置（秒）
        这是实时位置，考虑了所有缓冲延迟
        """
        if not self._is_playing:
            return 0.0
        
        # 返回实时更新的播放位置
        return self._audio_position

    def generate_audio(self, text: str) -> tuple[str | None, float]:
        """
        生成 TTS 音频并保存到临时文件，同时捕获词边界时间戳

        Args:
            text: 要合成的文本

        Returns:
            (音频文件路径，时长秒数)
        调用后可通过 self.word_boundaries 获取词边界列表
        """
        if not text:
            self.word_boundaries = []
            return None, 0.0

        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"tts_{int(time.time() * 1000)}.mp3")

        try:
            # 使用 stream() + boundary="WordBoundary" 获取精确词边界
            communicate = edge_tts.Communicate(
                text, self.voice, boundary="WordBoundary"
            )

            word_boundaries: List[WordBoundary] = []

            async def _stream_and_save():
                with open(audio_path, "wb") as f:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            f.write(chunk["data"])
                        elif chunk["type"] == "WordBoundary":
                            offset_s = chunk["offset"] / 1e7
                            dur_s = chunk["duration"] / 1e7
                            word_boundaries.append(WordBoundary(
                                text=chunk["text"],
                                offset=offset_s,
                                duration=dur_s,
                                end=offset_s + dur_s,
                            ))

            asyncio.run(_stream_and_save())

            self.word_boundaries = word_boundaries

            duration = get_audio_duration(audio_path)
            self._audio_duration = duration  # 保存时长供 play_audio_async 使用

            return audio_path, duration
        except Exception as e:
            print(f"[TTS] Error generating audio: {e}")
            # 如果不是默认音色，回退到默认音色重试
            default_id = "zh-CN-XiaoxiaoNeural"
            if self.voice != default_id:
                print(f"[TTS] 回退到默认音色 {default_id} 重试...")
                old_voice = self.voice
                self.voice = default_id
                result = self.generate_audio(text)
                self.voice = old_voice  # 恢复，让用户可以再次尝试
                return result
            self.word_boundaries = []
            return None, 0.0

    def _get_audio_duration(self, audio_path: str) -> float:
        """
        获取音频文件实际时长（秒）
        
        使用 mutagen 获取 MP3 时长
        """
        return get_audio_duration(audio_path)

    def _estimate_duration_from_file(self, audio_path: str) -> float:
        """通过文件大小估算 MP3 时长（约 16KB/s）"""
        try:
            size_bytes = os.path.getsize(audio_path)
            # MP3 约 16KB/s (128kbps)
            return size_bytes / (16 * 1024)
        except:
            return 1.0
    
    def play_audio_async(self, audio_path: str):
        """
        异步播放音频

        Args:
            audio_path: 音频文件路径
        """
        if not audio_path or not os.path.exists(audio_path):
            return

        # 确保 pygame mixer 只初始化一次（避免每次播放花几百ms）
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        def _play():
            self._is_playing = True
            self._audio_position = 0.0  # 重置位置

            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            # 在 play() 之后才开始计时，确保和实际音频同步
            self._playback_start_time = time.time()

            # 等待播放完成（用 get_busy 检测）
            while pygame.mixer.music.get_busy():
                time.sleep(0.02)  # 20ms 更新间隔（匹配 20fps 渲染）
                # 更新播放位置
                self._audio_position = min(
                    time.time() - self._playback_start_time,
                    self._audio_duration
                )

            self._is_playing = False
            self._audio_position = self._audio_duration  # 播放完成

            # 清理
            pygame.mixer.music.unload()
            try:
                os.remove(audio_path)
            except:
                pass

        self._playback_thread = threading.Thread(target=_play, daemon=True)
        self._playback_thread.start()

    def wait_playback(self):
        """等待播放完成"""
        if self._playback_thread:
            self._playback_thread.join()
    
    def speak_async(self, text: str) -> tuple[threading.Thread | None, float]:
        """
        生成并播放 TTS 音频（旧接口，保持兼容）
        
        Args:
            text: 要合成的文本
            
        Returns:
            (播放线程，时长秒数)
        """
        audio_path, duration = self.generate_audio(text)
        
        if audio_path:
            self.play_audio_async(audio_path)
            return self._playback_thread, duration
        
        return None, duration
    
    def _estimate_duration(self, text: str) -> float:
        """估算语音时长 (秒)"""
        text = (text or "").strip()
        if not text:
            return 0.0
        return max(1.2, len(text) / 12.0)
    
    def estimate_duration(self, text: str) -> float:
        """公开时长估算方法"""
        return self._estimate_duration(text)
