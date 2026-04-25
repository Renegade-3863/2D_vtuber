import gc
import logging
import os
import sys
import time
import threading
import queue
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tha3.poser.modes.load_poser import load_poser
from tha3.util import extract_pytorch_image_from_PIL_image, rgba_to_numpy_image

from ai_runtime.tha_pose_mapper import THAPoseMapper
from ai_runtime.llm_api_client import LLMApiClient
from ai_runtime.tts_client import TTSClient, VOICE_CATALOG, DEFAULT_VOICE_KEY, list_voices, resolve_voice
from ai_runtime.phoneme_mouth_mapper import PhonemeMouthMapper, PhonemeFrame
from ai_runtime.audio_phoneme_aligner import AudioPhonemeAligner
from ai_runtime.face_tracker import FaceTracker
from ai_runtime.stt_client import STTClient

# ---- TTS engine selection (env-controlled) -------------------------------
# TTS_ENGINE=edge       -> Edge TTS (default, preset voices, native word boundaries)
# TTS_ENGINE=xtts       -> Coqui XTTS v2 voice cloning + WhisperX forced alignment
# TTS_ENGINE=gpt_sovits -> local GPT-SoVITS API voice cloning + WhisperX forced alignment
# TTS_ENGINE=cosyvoice  -> Aliyun DashScope CosyVoice cloud TTS + WhisperX forced alignment
# XTTS_REFERENCE_WAV: path to a clean ~6-15s reference voice clip
TTS_ENGINE = os.environ.get("TTS_ENGINE", "edge").strip().lower()
XTTS_REFERENCE_WAV = os.environ.get(
    "XTTS_REFERENCE_WAV",
    str(Path(__file__).resolve().parents[1] / "data" / "voice" / "keqing.mp3"),
)
GPT_SOVITS_REFERENCE_WAV = os.environ.get(
    "GPT_SOVITS_REFERENCE_WAV",
    XTTS_REFERENCE_WAV,
)

# For debugging purpose, set to True for debugging the mouth shape
DEBUG_MOUTH = False

# Display post-process upscale (render stays 512, display can be larger)
DISPLAY_UPSCALE = 2.0  # 512 -> 1024
DISPLAY_INTERPOLATION = cv2.INTER_CUBIC

# Input mode debug option:
# - Default can be changed to "text" when developing in quiet environments (e.g., library)
# - Can be overridden by env THA_INPUT_MODE=text|mic
# - Can be overridden by CLI args --text / --mic
DEFAULT_INPUT_MODE = "text"
INPUT_MODE_ENV_KEY = "THA_INPUT_MODE"

# threshold for closing the mouth
SILENCE_THRESHOLD = 0.10  # 100ms 以上的真空隙才认为是停顿，避免连读时误判

# Module for character states machine
class InteractionState(Enum):
    """角色交互状态机"""
    IDLE = "idle"            # 空闲：放松、随机注视、正常眨眼
    LISTENING = "listening"  # 倾听：注视用户、微微前倾、偶尔点头
    THINKING = "thinking"    # 思考：眼神飘移、歪头、皱眉
    SPEAKING = "speaking"    # 说话：口型同步、手势、情绪表达
    POST_SPEAK = "post_speak"  # 说完过渡：微微放松、短暂停顿后回到 idle

# The shared board between the input loop and the render loop, holding the current state of the character
# Handle cross thread data synchronization with locks
@dataclass
class RenderState:
    # The emotion now, guiding the poser to generate the corresponding expression.
    emotion: str = "neutral"
    # The intensity of the emotion
    # TODO: might not be necessary, some expressions are tested to be weird if not performed at full intensity.
    intensity: float = 0.5
    # The motion hint for the poser to generate corresponding motion 
    motion_hint: str = "none"
    # Whether the character is currently speaking or not
    speaking: bool = False
    # Mark the starting and ending time of the speaking segment
    speak_start: float = 0.0
    speak_end: float = 0.0
    # The list of phoneme frames for the current speaking segment, used for lip sync and mouth shape generation
    phoneme_frames: list = None 
    # The total time duration of the current speaking segment, used for lip sync and mouth shape generation
    audio_duration: float = 0.0  # 音频时长
    # Two purposes: 
    # 1) Asynchronously play the audio
    # 2) Align audio with phonemes to get the phoneme timeline for lip sync
    audio_path: str = None  
    # tts object, used to get the real-time playback position for better lip sync (especially for long sentences)
    tts: object = None 
    # Emotion timeline is used by the poser to switch between different emotions during one single response. aiming to make the character more lively and less static.
    # TODO: Optiomize this 
    emotion_timeline: list = None
    # The current interaction state of the character, used to guide the motion animation during idle time like in games or vtuber live streaming.
    interaction_state: str = "idle"  
    # The time when the current interaction state enters
    state_enter_time: float = 0.0   

# Build the emotion timeline based on LLM segments and TTS word boundaries
def _build_emotion_timeline(                  
    segments: list,
    word_boundaries: list,
    total_duration: float,
) -> list:
    """
    根据 LLM 返回的 segments 和 TTS 的 word_boundaries，
    构建 [(offset_sec, emotion, intensity), ...] 时间线。

    原理：将各 segment 的文本长度按比例映射到 word_boundaries 的时间轴上。
    """
    if not segments:
        return [(0.0, "neutral", 0.5)]
    if len(segments) == 1:
        seg = segments[0]
        return [(0.0, seg.get("emotion", "neutral"), float(seg.get("intensity", 0.5)))]

    # 计算每个 segment 的字符数和累计字符边界
    seg_char_ends = []  # 每段结束的字符位置
    char_count = 0
    for seg in segments:
        char_count += len(seg.get("text", ""))
        seg_char_ends.append(char_count)
    total_chars = char_count

    if total_chars == 0:
        return [(0.0, "neutral", 0.5)]

    # 用 word_boundaries 建立"已朗读字符数 → 时间"的映射
    # word_boundaries: [{text, offset, duration, end}, ...]
    char_time_map = []  # [(累计字符数, 该词的起始时间)]
    spoken_chars = 0
    if word_boundaries:
        for wb in word_boundaries:
            char_time_map.append((spoken_chars, wb.offset))
            spoken_chars += len(wb.text)
        # 结尾哨兵
        char_time_map.append((spoken_chars, total_duration))

    def chars_to_time(char_pos: int) -> float:
        """将字符位置转换为时间偏移"""
        if not char_time_map:
            # 没有 word_boundaries → 按字符比例估算
            return (char_pos / total_chars) * total_duration
        # 在 char_time_map 中找到该字符位置对应的时间
        for i in range(len(char_time_map) - 1):
            c0, t0 = char_time_map[i]
            c1, t1 = char_time_map[i + 1]
            if c0 <= char_pos <= c1:
                if c1 == c0:
                    return t0
                frac = (char_pos - c0) / (c1 - c0)
                return t0 + frac * (t1 - t0)
        return total_duration

    timeline = []
    prev_end = 0
    for i, seg in enumerate(segments):
        offset = chars_to_time(prev_end)
        emo = seg.get("emotion", "neutral")
        inten = float(seg.get("intensity", 0.5))
        timeline.append((offset, emo, inten))
        prev_end = seg_char_ends[i]

    return timeline


def smooth_pose(current, target, alpha=0.35):
    """平滑插值，alpha 越大变化越快"""
    return [c + (t - c) * alpha for c, t in zip(current, target)]


def add_pose(base: list, add: dict, name_to_idx: dict, name_to_range: dict):
    for k, v in add.items():
        idx = name_to_idx.get(k)
        if idx is None:
            continue
        lo, hi = name_to_range.get(k, (0.0, 1.0))
        base[idx] = max(lo, min(hi, base[idx] + float(v)))


def _intensity_curve(raw: float) -> float:
    """非线性 intensity 映射：让中等值（0.4~0.6）也能产生充分表情。
    
    原始 intensity 由 LLM 给出 (0~1)，但线性乘法导致中等值表情太弱。
    用更激进的曲线：0.3→0.75, 0.5→0.87, 0.7→0.94, 1.0→1.0
    下限 0.55，确保表情始终明显可见。
    """
    if raw <= 0.0:
        return 0.0
    boosted = raw ** 0.35  # 比 sqrt 更激进的曲线
    return max(0.55, min(1.0, boosted))


class EmotionSmoother:
    """平滑过渡情绪参数（避免表情跳变）"""

    def __init__(self, attack: float = 0.30, release: float = 0.08):
        self._current_emotion: str = "neutral"
        self._current_intensity: float = 0.0
        self._target_emotion: str = "neutral"
        self._target_intensity: float = 0.0
        self._attack = attack   # 进入新情绪的速度（每帧增量）
        self._release = release  # 旧情绪淡出速度
        self._blend: float = 0.0  # 0=旧情绪，1=新情绪

    def set_target(self, emotion: str, intensity: float):
        if emotion != self._target_emotion:
            # 新情绪来了，开始过渡
            self._current_emotion = self._target_emotion
            self._current_intensity = self._target_intensity * self._blend
            self._target_emotion = emotion
            self._target_intensity = intensity
            self._blend = 0.0
        else:
            self._target_intensity = intensity

    def sample(self) -> tuple:
        """返回 (旧情绪, 旧强度, 新情绪, 新强度, blend因子)"""
        # blend 逐渐从 0→1
        if self._blend < 1.0:
            self._blend = min(1.0, self._blend + self._attack)
        # 同时旧情绪淡出
        if self._current_intensity > 0:
            self._current_intensity = max(0, self._current_intensity - self._release)
        return (
            self._current_emotion,
            self._current_intensity * (1.0 - self._blend),
            self._target_emotion,
            self._target_intensity * self._blend,
        )


def is_silence_gap(phoneme_frames: list, current_time: float, threshold: float = 0.10) -> bool:
    """
    检测当前时间是否处于停顿间隙

    两类静音：
    1. current_time 落在两个音素帧之间（comma/period 的硬停顿）→ 直接 True
    2. current_time 在某音素帧末尾 60%+，且到下一帧间隔 > threshold → True
    """
    if not phoneme_frames:
        return False

    # ---- Case 1: current_time 完全在所有 phoneme 之间的空隙里 ----
    # 这是逗号/句号停顿的核心场景，之前漏掉了。
    prev_end = 0.0
    for frame in phoneme_frames:
        if current_time < frame.start_time:
            # 落在 (prev_end, frame.start_time) 这个空白区间
            gap_len = frame.start_time - prev_end
            if gap_len > threshold:
                return True
            return False
        prev_end = frame.end_time

    # ---- Case 2: 在某个 phoneme 内，且接近尾部并且后面有空隙 ----
    for i, frame in enumerate(phoneme_frames):
        if frame.start_time <= current_time < frame.end_time:
            progress = (current_time - frame.start_time) / frame.duration if frame.duration > 0 else 0

            if i == len(phoneme_frames) - 1:
                return False

            if progress > 0.60:
                next_frame = phoneme_frames[i + 1]
                gap = next_frame.start_time - current_time
                if gap > threshold:
                    return True

            return False

    # 超出所有 phoneme 末尾——这种情况下让 render 走闭嘴逻辑。
    # 常见于 WhisperX 只对齐了前半句，后半句丢失。
    return True


class IdleAnimator:
    """空闲动画生成器（眨眼、视线、呼吸等）"""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        now = time.time()

        self._blink_start: float | None = None
        self._blink_duration_s: float = 0.14
        self._next_blink_t: float = now + float(self.rng.uniform(2.0, 5.0))
        self._blink_interval_range: tuple = (2.5, 5.5)

        self._gaze_current = np.array([0.0, 0.0], dtype=np.float32)
        self._gaze_from = self._gaze_current.copy()
        self._gaze_to = self._gaze_current.copy()
        self._gaze_move_start: float = now
        self._gaze_move_duration_s: float = 0.12
        self._gaze_hold_until: float = now + float(self.rng.uniform(0.4, 1.2))

        # 倾听状态的小点头
        self._listen_nod_next: float = now + 2.0
        self._listen_nod_start: float = 0.0
        self._listen_nodding: bool = False

        # 倾听微反应（多样化）
        self._listen_reaction_next: float = now + float(self.rng.uniform(3.0, 6.0))
        self._listen_reaction_type: str = "none"
        self._listen_reaction_start: float = 0.0

        # 相位漂移（让动作不那么机械重复）
        self._phase_drift = float(self.rng.uniform(0, 2 * np.pi))

        # 微表情
        self._micro_expr_next: float = now + float(self.rng.uniform(4.0, 8.0))
        self._micro_expr_type: str = "none"
        self._micro_expr_start: float = 0.0
        self._micro_expr_duration: float = 0.0

        # 面部追踪
        self._face_tracker = None
        self._face_active = False
        self._face_blend = 0.0          # 0=idle, 1=追踪 (平滑过渡)
        self._face_blend_speed = 3.0    # 每秒变化量 (0→1 约 0.33s)
        self._face_head_y = 0.0
        self._face_head_x = 0.0
        self._face_iris_x = 0.0
        self._face_iris_y = 0.0
        self._last_sample_time = now
        # 窗口位置修正
        self._win_offset_y = 0.0        # 角色窗口偏离屏幕中心的水平修正
        self._win_offset_x = 0.0        # 角色窗口偏离屏幕中心的垂直修正

        # 待机表演
        self._idle_perf = IdlePerformance(self.rng)

    def set_face_tracker(self, tracker):
        """设置外部面部追踪器"""
        self._face_tracker = tracker

    def update_window_offset(self, win_rect: tuple, screen_size: tuple):
        """
        根据角色窗口在屏幕上的位置，计算视线修正偏移。

        win_rect: (x, y, w, h) 窗口矩形
        screen_size: (screen_w, screen_h) 屏幕分辨率
        """
        wx, wy, ww, wh = win_rect
        sw, sh = screen_size
        # 窗口中心相对屏幕中心的归一化偏移 [-1, 1]
        win_cx = (wx + ww / 2) / sw * 2.0 - 1.0
        win_cy = (wy + wh / 2) / sh * 2.0 - 1.0
        # 窗口偏右 → 角色应该往左看向用户 → 负修正
        self._win_offset_y = -win_cx * 0.20
        # 窗口偏下 → 角色应该往上看 → 负修正
        self._win_offset_x = -win_cy * 0.15

    def _ease(self, x: float) -> float:
        x = max(0.0, min(1.0, float(x)))
        return x * x * (3.0 - 2.0 * x)

    def _blink_value(self, now: float) -> float:
        if self._blink_start is None:
            if now < self._next_blink_t:
                return 0.0
            self._blink_start = now
            self._blink_duration_s = float(self.rng.uniform(0.10, 0.18))
            lo, hi = self._blink_interval_range
            self._next_blink_t = now + float(self.rng.uniform(lo, hi))

        t = (now - self._blink_start) / max(1e-6, self._blink_duration_s)
        if t >= 1.0:
            self._blink_start = None
            return 0.0

        return float(np.sin(np.pi * t) ** 2)

    def gaze_at(self, now: float) -> np.ndarray:
        t = (now - self._gaze_move_start) / max(1e-6, self._gaze_move_duration_s)
        if t <= 0.0:
            return self._gaze_from
        if t >= 1.0:
            return self._gaze_to
        w = self._ease(t)
        return (1.0 - w) * self._gaze_from + w * self._gaze_to

    def sample(self, now: float, speaking_active: bool,
               interaction_state: str = "idle") -> dict:
        """根据交互状态生成 idle 动画参数"""

        # ---- 根据状态调整行为参数 ----
        if interaction_state == "listening":
            idle_scale = 0.45
            self._blink_interval_range = (3.0, 6.0)
            gaze_mode = "focus"
        elif interaction_state == "thinking":
            idle_scale = 0.55
            self._blink_interval_range = (3.5, 7.0)
            gaze_mode = "think"
        elif interaction_state == "speaking":
            idle_scale = 0.80
            self._blink_interval_range = (2.5, 5.5)
            gaze_mode = "speak"
        elif interaction_state == "post_speak":
            idle_scale = 0.85
            self._blink_interval_range = (2.0, 4.0)
            gaze_mode = "focus"
        else:  # idle
            idle_scale = 1.0
            self._blink_interval_range = (2.5, 5.5)
            gaze_mode = "random"

        # ---- 注视行为 ----
        self._update_gaze(now, gaze_mode)
        gaze = self.gaze_at(now)
        blink = self._blink_value(now)

        # ---- 身体晃动（基础正弦叠加 + 相位漂移让动作不重复）----
        drift = 0.3 * np.sin(now * 0.023 + self._phase_drift)
        head_y = idle_scale * (0.18 * np.sin(now * 0.85 + drift) + 0.08 * np.sin(now * 1.33 + 1.4 + drift * 1.3))
        head_x = idle_scale * (0.08 * np.sin(now * 0.70 + 2.2 + drift * 0.7) + 0.03 * np.sin(now * 1.1 + 0.8))
        neck_z = idle_scale * (0.07 * np.sin(now * 0.55 + 0.6 + drift * 0.9) + 0.03 * np.sin(now * 0.9 + 1.7))
        body_y = idle_scale * (0.055 * np.sin(now * 0.42 + 1.0 + drift * 0.5) + 0.02 * np.sin(now * 0.78 + 2.0))
        body_z = idle_scale * (0.045 * np.sin(now * 0.48 + 2.6 + drift * 0.6))

        # ---- 呼吸（节奏随状态变化）----
        breath_rates = {"idle": 1.57, "listening": 1.80, "thinking": 1.26,
                        "speaking": 1.57, "post_speak": 1.40}
        breath_rate = breath_rates.get(interaction_state, 1.57)
        breathing = 0.40 * np.sin(now * breath_rate)

        result = {
            "head_y": float(head_y),
            "head_x": float(head_x),
            "neck_z": float(neck_z),
            "body_y": float(body_y),
            "body_z": float(body_z),
            "breathing": float(breathing),
            "eye_wink_left": float(1.00 * blink),
            "eye_wink_right": float(1.00 * blink),
            "iris_rotation_x": float(gaze[0]),
            "iris_rotation_y": float(gaze[1]),
        }

        # ---- 微表情（让面部在空闲时更生动）----
        micro = self._sample_micro_expression(now, interaction_state)
        for k, v in micro.items():
            result[k] = result.get(k, 0.0) + v

        # ---- 面部追踪：角色看向用户 ----
        # face_x/y = 用户的脸在摄像头画面中的位置
        # 角色的瞳孔和头部都追向用户所在方向（同方向，不同增益）
        dt = now - self._last_sample_time
        self._last_sample_time = now
        face_detected_now = False
        if self._face_tracker is not None:
            face_x, face_y, face_detected_now = self._face_tracker.get_face_position()
            if face_detected_now:
                # 头部跟随用户位置
                face_head_y = face_x * -0.30
                face_head_x = face_y * -0.25
                # 瞳孔跟随头部方向（同步，情绪效果由 PupilAnimator 叠加）
                face_iris_y = face_head_y
                face_iris_x = face_head_x
                # 摄像头在屏幕上方 → 向下偏移补偿
                _CAM_SCREEN_OFFSET = 0.12
                face_iris_x += _CAM_SCREEN_OFFSET
                face_head_x += _CAM_SCREEN_OFFSET * 0.5
                # 窗口位置修正（角色窗口不在屏幕中央时补偿视线）
                face_iris_y += self._win_offset_y
                face_iris_x += self._win_offset_x
                face_head_y += self._win_offset_y * 0.6
                face_head_x += self._win_offset_x * 0.6
                blend_map = {"listening": 0.95, "idle": 0.95, "speaking": 0.5,
                             "thinking": 0.3, "post_speak": 0.7}
                blend = blend_map.get(interaction_state, 0.7)
                self._face_head_y = float(face_head_y)
                self._face_head_x = float(
                    result.get("head_x", 0) * (1 - blend) + face_head_x * blend)
                self._face_iris_x = float(face_iris_x)
                self._face_iris_y = float(face_iris_y)
                self._face_active = True
            else:
                self._face_active = False
        else:
            self._face_active = False

        # 平滑过渡 face_blend: 检测到脸时渐入，丢失时渐出
        if self._face_active:
            self._face_blend = min(1.0, self._face_blend + self._face_blend_speed * dt)
        else:
            self._face_blend = max(0.0, self._face_blend - self._face_blend_speed * dt)

        # 将追踪值按 face_blend 混入 result（blend=0 时完全用 idle 值）
        if self._face_blend > 0.001:
            fb = self._face_blend
            result["head_y"] = float(result["head_y"] * (1 - fb) + self._face_head_y * fb)
            result["head_x"] = float(result["head_x"] * (1 - fb) + self._face_head_x * fb)
            result["iris_rotation_x"] = float(result["iris_rotation_x"] * (1 - fb) + self._face_iris_x * fb)
            result["iris_rotation_y"] = float(result["iris_rotation_y"] * (1 - fb) + self._face_iris_y * fb)

        # ---- 状态特有叠加 ----
        if interaction_state == "listening":
            result["body_z"] = result.get("body_z", 0) + 0.06
            result["head_x"] = result.get("head_x", 0) + 0.05
            nod = self._listen_nod_value(now)
            if nod != 0:
                result["head_x"] = result.get("head_x", 0) + nod
            # 多样化倾听反应
            reaction = self._sample_listening_reaction(now)
            for k, v in reaction.items():
                result[k] = result.get(k, 0.0) + v

        elif interaction_state == "thinking":
            result["neck_z"] = result.get("neck_z", 0) + 0.12
            result["head_x"] = result.get("head_x", 0) - 0.06
            result["body_y"] = result.get("body_y", 0) + 0.03

        elif interaction_state == "post_speak":
            result["head_x"] = result.get("head_x", 0) - 0.04
            result["body_z"] = result.get("body_z", 0) - 0.03

        # ---- 待机表演（仅 idle 时触发，其他状态自动中断）----
        perf = self._idle_perf.sample(now, interaction_state)
        if perf:
            for k, v in perf.items():
                result[k] = result.get(k, 0.0) + v

        return result

    def _listen_nod_value(self, now: float) -> float:
        """倾听时的小点头（自然的回应动作）"""
        if not self._listen_nodding:
            if now >= self._listen_nod_next:
                self._listen_nodding = True
                self._listen_nod_start = now
            return 0.0
        elapsed = now - self._listen_nod_start
        dur = 0.4  # 一次点头 0.4s
        if elapsed >= dur:
            self._listen_nodding = False
            self._listen_nod_next = now + float(self.rng.uniform(2.0, 4.0))
            return 0.0
        # 单次正弦脉冲
        return float(0.08 * np.sin(np.pi * elapsed / dur))

    def _sample_micro_expression(self, now: float, interaction_state: str) -> dict:
        """生成微表情参数（空闲/倾听时让面部更生动）"""
        if interaction_state in ("speaking", "thinking", "post_speak"):
            self._micro_expr_type = "none"
            return {}

        if self._micro_expr_type == "none" and now >= self._micro_expr_next:
            choices = ["brow_raise", "brow_furrow", "eye_soften", "half_smile"]
            self._micro_expr_type = self.rng.choice(choices)
            self._micro_expr_start = now
            self._micro_expr_duration = float(self.rng.uniform(0.6, 1.2))

        if self._micro_expr_type == "none":
            return {}

        elapsed = now - self._micro_expr_start
        if elapsed >= self._micro_expr_duration:
            self._micro_expr_type = "none"
            self._micro_expr_next = now + float(self.rng.uniform(4.0, 8.0))
            return {}

        strength = float(np.sin(np.pi * elapsed / self._micro_expr_duration))

        if self._micro_expr_type == "brow_raise":
            return {"eyebrow_raised_left": 0.15 * strength,
                    "eyebrow_raised_right": 0.15 * strength}
        elif self._micro_expr_type == "brow_furrow":
            return {"eyebrow_lowered_left": 0.10 * strength,
                    "eyebrow_lowered_right": 0.10 * strength}
        elif self._micro_expr_type == "eye_soften":
            return {"eye_relaxed_left": 0.06 * strength,
                    "eye_relaxed_right": 0.06 * strength}
        elif self._micro_expr_type == "half_smile":
            return {"mouth_raised_corner_left": 0.10 * strength,
                    "mouth_raised_corner_right": 0.10 * strength}
        return {}

    def _sample_listening_reaction(self, now: float) -> dict:
        """倾听时的多样化微反应"""
        if self._listen_reaction_type == "none" and now >= self._listen_reaction_next:
            choices = ["eyebrow_raise", "slight_tilt", "eye_widen"]
            self._listen_reaction_type = self.rng.choice(choices)
            self._listen_reaction_start = now

        if self._listen_reaction_type == "none":
            return {}

        elapsed = now - self._listen_reaction_start
        dur = 0.5
        if elapsed >= dur:
            self._listen_reaction_type = "none"
            self._listen_reaction_next = now + float(self.rng.uniform(2.5, 5.0))
            return {}

        strength = float(np.sin(np.pi * elapsed / dur))

        if self._listen_reaction_type == "eyebrow_raise":
            return {"eyebrow_raised_left": 0.20 * strength,
                    "eyebrow_raised_right": 0.20 * strength}
        elif self._listen_reaction_type == "slight_tilt":
            return {"neck_z": 0.06 * strength}
        elif self._listen_reaction_type == "eye_widen":
            return {"eye_surprised_left": 0.08 * strength,
                    "eye_surprised_right": 0.08 * strength}
        return {}

    def _update_gaze(self, now: float, gaze_mode: str = "random"):
        if now < self._gaze_hold_until:
            return

        self._gaze_current = self.gaze_at(now)
        self._gaze_from = self._gaze_current.copy()

        if gaze_mode == "focus":
            # 注视用户（镜头正前方），非常小的随机偏移
            self._gaze_to = np.array([
                float(self.rng.uniform(-0.06, 0.06)),
                float(self.rng.uniform(-0.04, 0.04)),
            ], dtype=np.float32)
            hold_s = float(self.rng.uniform(1.0, 2.5))  # 久一些

        elif gaze_mode == "think":
            # 思考时目光飘向左上方
            self._gaze_to = np.array([
                float(self.rng.uniform(-0.25, -0.05)),  # 偏左
                float(self.rng.uniform(-0.20, -0.05)),  # 偏上
            ], dtype=np.float32)
            hold_s = float(self.rng.uniform(0.8, 2.0))

        elif gaze_mode == "speak":
            # 说话时：70% 看用户，30% 短暂移开
            if self.rng.random() < 0.70:
                self._gaze_to = np.array([
                    float(self.rng.uniform(-0.08, 0.08)),
                    float(self.rng.uniform(-0.06, 0.06)),
                ], dtype=np.float32)
                hold_s = float(self.rng.uniform(0.8, 2.0))
            else:
                self._gaze_to = np.array([
                    float(self.rng.uniform(-0.20, 0.20)),
                    float(self.rng.uniform(-0.15, 0.15)),
                ], dtype=np.float32)
                hold_s = float(self.rng.uniform(0.3, 0.6))

        else:  # random
            max_x, max_y = 0.28, 0.20
            self._gaze_to = np.array([
                float(self.rng.uniform(-max_x, max_x)),
                float(self.rng.uniform(-max_y, max_y)),
            ], dtype=np.float32)
            hold_s = float(self.rng.uniform(0.35, 1.0))

        self._gaze_move_start = now
        self._gaze_move_duration_s = float(self.rng.uniform(0.07, 0.14))
        self._gaze_hold_until = now + self._gaze_move_duration_s + hold_s


# ========== 看板娘待机动画系统 ==========
class IdlePerformance:
    """
    待机表演系统 — 适配半身像角色（无手部动作）

    在角色无交互（idle）时，随机触发有存在感的待机动画片段，
    让角色像游戏主界面的看板娘一样有"活人感"。

    所有表演仅使用头部/眼睛/眉毛/嘴巴/身体微动，
    不暗示任何手臂或手部动作。

    动画类型：
    - head_tilt:   好奇歪头（歪头 + 眨眼 + 微笑，像在观察用户）
    - look_around: 张望四周（眼球左右快扫 + 头部跟随 + 好奇表情）
    - doze:        打瞌睡（缓慢低头 + 眼睛渐闭 → 猛抬头惊醒 + 眨眼）
    - hum:         哼歌摇摆（轻微左右摇晃 + 微笑 + 微张嘴）
    - deep_breath: 深呼吸（闭眼 + 身体缓缓起伏 + 放松回正）
    - glance_away: 害羞移开视线（侧头 + 视线偏移 + 微笑）
    - alert:       忽然警觉（眼睛睁大 + 身体微挺 + 快速扫视 → 放松）
    - bored:       无聊（微低头 + 无聊眼 + 轻叹）
    """

    # (动画名, 时长秒, 权重)
    PERFORMANCES = [
        ("head_tilt",   2.5, 2.0),
        ("look_around", 3.0, 1.5),
        ("doze",        4.5, 0.8),
        ("hum",         4.0, 1.5),
        ("deep_breath", 3.5, 1.5),
        ("glance_away", 2.5, 1.8),
        ("alert",       2.0, 1.0),
        ("bored",       3.0, 1.5),
        ("glance_away", 2.5, 1.8),
        ("alert",       2.0, 1.0),
        ("bored",       3.0, 1.2),
    ]

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        self._current: str = "none"
        self._start: float = 0.0
        self._duration: float = 0.0
        self._next_trigger: float = time.time() + float(self.rng.uniform(3.0, 6.0))
        self._cooldown_range = (5.0, 12.0)  # 两次表演之间的间隔
        self._interrupted: bool = False

    def interrupt(self):
        """用户开始交互时，立即中断当前表演"""
        if self._current != "none":
            self._interrupted = True

    @property
    def is_playing(self) -> bool:
        return self._current != "none"

    @property
    def current_name(self) -> str:
        return self._current

    def sample(self, now: float, interaction_state: str) -> dict:
        """
        每帧调用。只在 idle 状态下触发和播放表演。
        返回参数增量 dict。
        """
        # 非 idle 状态：中断并跳过
        if interaction_state != "idle":
            if self._current != "none":
                self._current = "none"
                self._interrupted = True
            return {}

        # 被中断后或从非 idle 刚回来：短暂冷却
        if self._interrupted:
            self._interrupted = False
            self._next_trigger = now + float(self.rng.uniform(3.0, 6.0))
            return {}

        # 触发新表演
        if self._current == "none":
            if now < self._next_trigger:
                return {}
            self._pick_performance(now)

        # 播放中
        elapsed = now - self._start
        if elapsed >= self._duration:
            self._current = "none"
            lo, hi = self._cooldown_range
            self._next_trigger = now + float(self.rng.uniform(lo, hi))
            return {}

        t = elapsed / self._duration  # 归一化进度 0→1
        return self._animate(self._current, t, elapsed)

    def _pick_performance(self, now: float):
        """按权重随机选择一个表演"""
        names = [p[0] for p in self.PERFORMANCES]
        weights = np.array([p[2] for p in self.PERFORMANCES])
        weights /= weights.sum()
        choice = self.rng.choice(len(names), p=weights)
        self._current = names[choice]
        self._duration = self.PERFORMANCES[choice][1]
        self._start = now
        print(f"[IdlePerformance] ▶ {self._current} ({self._duration:.1f}s)")
    def _ease_in_out(self, t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _bell(self, t: float, center: float, width: float) -> float:
        """钟形脉冲：在 center±width 区间内平滑上升再下降"""
        x = (t - center) / max(0.001, width)
        if abs(x) > 1.0:
            return 0.0
        return float((1.0 - x * x) ** 2)

    def _animate(self, name: str, t: float, elapsed: float) -> dict:
        PI = np.pi

        if name == "head_tilt":
            # 好奇歪头：歪过去 → 停留观察 → 回正
            tilt_dir = 1.0  # 向右歪
            if t < 0.3:
                p = self._ease_in_out(t / 0.3)
                return {
                    "neck_z": float(0.18 * tilt_dir * p),
                    "head_y": float(0.06 * tilt_dir * p),
                    "eyebrow_raised_left": float(0.25 * p),
                    "eyebrow_raised_right": float(0.15 * p),
                    "eye_surprised_left": float(0.10 * p),
                    "eye_surprised_right": float(0.10 * p),
                }
            elif t < 0.7:
                # 停留观察，微微晃动
                sway = 0.02 * np.sin(elapsed * 3.0)
                blink_pulse = self._bell(t, 0.5, 0.06)
                return {
                    "neck_z": float(0.18 * tilt_dir + sway),
                    "head_y": float(0.06 * tilt_dir),
                    "eyebrow_raised_left": float(0.25),
                    "eyebrow_raised_right": float(0.15),
                    "eye_surprised_left": float(0.10),
                    "eye_surprised_right": float(0.10),
                    "eye_wink_left": float(0.8 * blink_pulse),
                    "eye_wink_right": float(0.8 * blink_pulse),
                    "mouth_raised_corner_left": float(0.12),
                    "mouth_raised_corner_right": float(0.12),
                }
            else:
                p = self._ease_in_out((t - 0.7) / 0.3)
                return {
                    "neck_z": float(0.18 * tilt_dir * (1 - p)),
                    "head_y": float(0.06 * tilt_dir * (1 - p)),
                    "mouth_raised_corner_left": float(0.12 * (1 - p)),
                    "mouth_raised_corner_right": float(0.12 * (1 - p)),
                }

        elif name == "look_around":
            # 张望四周：眼球左看 → 头跟随 → 右看 → 回正
            if t < 0.2:
                p = self._ease_in_out(t / 0.2)
                return {
                    "iris_rotation_y": float(-0.25 * p),
                    "head_y": float(-0.08 * p),
                    "eyebrow_raised_left": float(0.15 * p),
                    "eyebrow_raised_right": float(0.15 * p),
                }
            elif t < 0.4:
                p = self._ease_in_out((t - 0.2) / 0.2)
                return {
                    "iris_rotation_y": float(-0.25 + 0.50 * p),
                    "head_y": float(-0.08 + 0.16 * p),
                    "eyebrow_raised_left": float(0.15),
                    "eyebrow_raised_right": float(0.15),
                    "eye_surprised_left": float(0.08 * p),
                    "eye_surprised_right": float(0.08 * p),
                }
            elif t < 0.65:
                # 右侧停留
                sway = 0.02 * np.sin(elapsed * 4.0)
                return {
                    "iris_rotation_y": float(0.25 + sway),
                    "head_y": float(0.08),
                    "eyebrow_raised_left": float(0.15),
                    "eyebrow_raised_right": float(0.15),
                    "eye_surprised_left": float(0.08),
                    "eye_surprised_right": float(0.08),
                }
            else:
                p = self._ease_in_out((t - 0.65) / 0.35)
                return {
                    "iris_rotation_y": float(0.25 * (1 - p)),
                    "head_y": float(0.08 * (1 - p)),
                    "eyebrow_raised_left": float(0.15 * (1 - p)),
                    "eyebrow_raised_right": float(0.15 * (1 - p)),
                    "eye_surprised_left": float(0.08 * (1 - p)),
                    "eye_surprised_right": float(0.08 * (1 - p)),
                }

        elif name == "doze":
            # 打瞌睡：慢慢低头闭眼 → 猛抬头惊醒 → 眨眼恢复
            if t < 0.45:
                # 缓慢下垂
                p = self._ease_in_out(t / 0.45)
                return {
                    "head_x": float(0.15 * p),
                    "body_z": float(0.06 * p),
                    "eye_wink_left": float(0.70 * p),
                    "eye_wink_right": float(0.70 * p),
                    "eyebrow_lowered_left": float(0.15 * p),
                    "eyebrow_lowered_right": float(0.15 * p),
                    "iris_rotation_x": float(0.10 * p),
                    "mouth_aaa": float(0.04 * p),
                }
            elif t < 0.55:
                # 猛抬头惊醒！
                wake_t = (t - 0.45) / 0.10
                p_up = self._ease_in_out(wake_t)
                return {
                    "head_x": float(0.15 * (1 - p_up) - 0.10 * p_up),
                    "body_z": float(0.06 * (1 - p_up) - 0.04 * p_up),
                    "eye_wink_left": float(0.70 * (1 - p_up)),
                    "eye_wink_right": float(0.70 * (1 - p_up)),
                    "eye_surprised_left": float(0.30 * p_up),
                    "eye_surprised_right": float(0.30 * p_up),
                    "eyebrow_raised_left": float(0.35 * p_up),
                    "eyebrow_raised_right": float(0.35 * p_up),
                    "iris_rotation_x": float(0.10 * (1 - p_up)),
                    "mouth_aaa": float(0.04 + 0.15 * p_up),
                }
            elif t < 0.70:
                # 惊醒后愣住
                blink_pulse = self._bell((t - 0.55) / 0.15, 0.5, 0.3)
                return {
                    "head_x": float(-0.10),
                    "body_z": float(-0.04),
                    "eye_surprised_left": float(0.30),
                    "eye_surprised_right": float(0.30),
                    "eyebrow_raised_left": float(0.35),
                    "eyebrow_raised_right": float(0.35),
                    "eye_wink_left": float(0.9 * blink_pulse),
                    "eye_wink_right": float(0.9 * blink_pulse),
                    "mouth_aaa": float(0.15),
                }
            else:
                # 缓慢恢复 + 不好意思的微笑
                p = self._ease_in_out((t - 0.70) / 0.30)
                return {
                    "head_x": float(-0.10 * (1 - p)),
                    "body_z": float(-0.04 * (1 - p)),
                    "eye_surprised_left": float(0.30 * (1 - p)),
                    "eye_surprised_right": float(0.30 * (1 - p)),
                    "eyebrow_raised_left": float(0.35 * (1 - p) + 0.10 * p),
                    "eyebrow_raised_right": float(0.35 * (1 - p) + 0.10 * p),
                    "mouth_raised_corner_left": float(0.20 * p),
                    "mouth_raised_corner_right": float(0.20 * p),
                    "mouth_aaa": float(0.15 * (1 - p)),
                }

        elif name == "hum":
            # 哼歌摇摆：轻微左右晃 + 微笑 + 嘴微动
            sway_y = 0.10 * np.sin(elapsed * 2.2)
            sway_z = 0.06 * np.sin(elapsed * 2.2 + 0.5)
            head_bob = 0.04 * np.sin(elapsed * 4.4)
            mouth_hum = 0.06 + 0.04 * np.sin(elapsed * 3.3)
            # 渐入渐出
            envelope = 1.0
            if t < 0.15:
                envelope = self._ease_in_out(t / 0.15)
            elif t > 0.85:
                envelope = self._ease_in_out((1.0 - t) / 0.15)
            return {
                "head_y": float(sway_y * envelope),
                "neck_z": float(sway_z * envelope),
                "head_x": float(head_bob * envelope - 0.03 * envelope),
                "body_y": float(sway_y * 0.4 * envelope),
                "eye_happy_wink_left": float(0.10 * envelope),
                "eye_happy_wink_right": float(0.10 * envelope),
                "mouth_raised_corner_left": float(0.15 * envelope),
                "mouth_raised_corner_right": float(0.15 * envelope),
                "mouth_aaa": float(mouth_hum * envelope),
            }

        elif name == "deep_breath":
            # 深呼吸：闭眼 + 身体缓缓起伏 + 放松表情
            if t < 0.3:
                # 吸气：身体微挺 + 闭眼
                p = self._ease_in_out(t / 0.3)
                return {
                    "head_x": float(-0.06 * p),       # 微微仰头
                    "body_z": float(-0.05 * p),        # 身体微挺
                    "breathing": float(0.25 * p),      # 吸气
                    "eye_relaxed_left": float(0.40 * p),
                    "eye_relaxed_right": float(0.40 * p),
                    "eye_wink_left": float(0.30 * p),  # 缓缓闭眼
                    "eye_wink_right": float(0.30 * p),
                    "eyebrow_raised_left": float(0.10 * p),
                    "eyebrow_raised_right": float(0.10 * p),
                }
            elif t < 0.55:
                # 屏息：闭眼保持
                hold_sway = 0.01 * np.sin(elapsed * 1.5)
                return {
                    "head_x": float(-0.06),
                    "body_z": float(-0.05 + hold_sway),
                    "breathing": float(0.25),
                    "eye_relaxed_left": float(0.40),
                    "eye_relaxed_right": float(0.40),
                    "eye_wink_left": float(0.30),
                    "eye_wink_right": float(0.30),
                    "eyebrow_raised_left": float(0.10),
                    "eyebrow_raised_right": float(0.10),
                }
            elif t < 0.80:
                # 呼气：身体放松回落 + 睁眼
                p = self._ease_in_out((t - 0.55) / 0.25)
                return {
                    "head_x": float(-0.06 * (1 - p) + 0.03 * p),
                    "body_z": float(-0.05 * (1 - p) + 0.03 * p),
                    "breathing": float(0.25 * (1 - p) - 0.15 * p),
                    "eye_relaxed_left": float(0.40 * (1 - p)),
                    "eye_relaxed_right": float(0.40 * (1 - p)),
                    "eye_wink_left": float(0.30 * (1 - p)),
                    "eye_wink_right": float(0.30 * (1 - p)),
                    "mouth_raised_corner_left": float(0.12 * p),
                    "mouth_raised_corner_right": float(0.12 * p),
                }
            else:
                # 回正 + 舒适微笑
                p = self._ease_in_out((t - 0.80) / 0.20)
                return {
                    "head_x": float(0.03 * (1 - p)),
                    "body_z": float(0.03 * (1 - p)),
                    "mouth_raised_corner_left": float(0.12 * (1 - p)),
                    "mouth_raised_corner_right": float(0.12 * (1 - p)),
                }

        elif name == "glance_away":
            # 害羞/不好意思地移开视线
            if t < 0.25:
                p = self._ease_in_out(t / 0.25)
                return {
                    "head_y": float(-0.10 * p),        # 头微转
                    "neck_z": float(-0.08 * p),        # 微侧
                    "iris_rotation_y": float(-0.20 * p),  # 视线偏移
                    "iris_rotation_x": float(0.06 * p),   # 微微看下
                    "eye_relaxed_left": float(0.12 * p),
                    "eye_relaxed_right": float(0.12 * p),
                    "mouth_raised_corner_left": float(0.10 * p),
                    "mouth_raised_corner_right": float(0.08 * p),
                }
            elif t < 0.60:
                # 停留：偏着头的自然晃动
                sway = 0.01 * np.sin(elapsed * 2.5)
                return {
                    "head_y": float(-0.10 + sway),
                    "neck_z": float(-0.08),
                    "iris_rotation_y": float(-0.20 + sway * 2),
                    "iris_rotation_x": float(0.06),
                    "eye_relaxed_left": float(0.12),
                    "eye_relaxed_right": float(0.12),
                    "mouth_raised_corner_left": float(0.10),
                    "mouth_raised_corner_right": float(0.08),
                }
            elif t < 0.80:
                # 回头看用户
                p = self._ease_in_out((t - 0.60) / 0.20)
                return {
                    "head_y": float(-0.10 * (1 - p)),
                    "neck_z": float(-0.08 * (1 - p)),
                    "iris_rotation_y": float(-0.20 * (1 - p)),
                    "iris_rotation_x": float(0.06 * (1 - p)),
                    "eye_relaxed_left": float(0.12 * (1 - p)),
                    "eye_relaxed_right": float(0.12 * (1 - p)),
                    "mouth_raised_corner_left": float(0.10),
                    "mouth_raised_corner_right": float(0.08),
                }
            else:
                # 微笑淡出
                p = self._ease_in_out((t - 0.80) / 0.20)
                return {
                    "mouth_raised_corner_left": float(0.10 * (1 - p)),
                    "mouth_raised_corner_right": float(0.08 * (1 - p)),
                }

        elif name == "alert":
            # 忽然警觉：睁大眼 + 身体微挺 + 快速扫视 → 放松
            if t < 0.15:
                # 突然警觉
                p = self._ease_in_out(t / 0.15)
                return {
                    "eye_surprised_left": float(0.30 * p),
                    "eye_surprised_right": float(0.30 * p),
                    "eyebrow_raised_left": float(0.30 * p),
                    "eyebrow_raised_right": float(0.30 * p),
                    "head_x": float(-0.06 * p),   # 微抬头
                    "body_z": float(-0.04 * p),    # 身体挺直
                    "iris_rotation_y": float(0.15 * p),  # 看向一侧
                }
            elif t < 0.40:
                # 快速扫视另一侧
                p = self._ease_in_out((t - 0.15) / 0.25)
                return {
                    "eye_surprised_left": float(0.30),
                    "eye_surprised_right": float(0.30),
                    "eyebrow_raised_left": float(0.30),
                    "eyebrow_raised_right": float(0.30),
                    "head_x": float(-0.06),
                    "body_z": float(-0.04),
                    "iris_rotation_y": float(0.15 - 0.30 * p),  # 从右扫到左
                    "head_y": float(0.05 - 0.10 * p),
                }
            elif t < 0.60:
                # 确认没事，微松一口气
                p = self._ease_in_out((t - 0.40) / 0.20)
                blink = self._bell(t, 0.50, 0.04)
                return {
                    "eye_surprised_left": float(0.30 * (1 - p)),
                    "eye_surprised_right": float(0.30 * (1 - p)),
                    "eyebrow_raised_left": float(0.30 * (1 - p * 0.6)),
                    "eyebrow_raised_right": float(0.30 * (1 - p * 0.6)),
                    "head_x": float(-0.06 * (1 - p)),
                    "body_z": float(-0.04 * (1 - p)),
                    "iris_rotation_y": float(-0.15 * (1 - p)),
                    "head_y": float(-0.05 * (1 - p)),
                    "eye_wink_left": float(0.85 * blink),
                    "eye_wink_right": float(0.85 * blink),
                }
            else:
                # 放松回正 + 微笑（虚惊一场）
                p = self._ease_in_out((t - 0.60) / 0.40)
                return {
                    "eyebrow_raised_left": float(0.12 * (1 - p)),
                    "eyebrow_raised_right": float(0.12 * (1 - p)),
                    "mouth_raised_corner_left": float(0.12 * (1 - p * 0.5)),
                    "mouth_raised_corner_right": float(0.12 * (1 - p * 0.5)),
                }

        elif name == "bored":
            # 无聊：微低头 + 无聊眼 + 轻叹
            if t < 0.25:
                p = self._ease_in_out(t / 0.25)
                return {
                    "head_x": float(0.08 * p),        # 低头
                    "neck_z": float(0.06 * p),         # 歪一点
                    "body_z": float(0.04 * p),         # 前倾
                    "eye_unimpressed_left": float(0.35 * p),
                    "eye_unimpressed_right": float(0.35 * p),
                    "eyebrow_lowered_left": float(0.12 * p),
                    "eyebrow_lowered_right": float(0.12 * p),
                    "iris_rotation_x": float(0.05 * p),  # 目光下垂
                }
            elif t < 0.50:
                # 无聊保持 + 轻叹（嘴微张呼气）
                sigh_t = (t - 0.25) / 0.25
                mouth_open = 0.08 * self._bell(sigh_t, 0.4, 0.35)
                sway = 0.01 * np.sin(elapsed * 1.8)
                return {
                    "head_x": float(0.08 + sway),
                    "neck_z": float(0.06),
                    "body_z": float(0.04),
                    "eye_unimpressed_left": float(0.35),
                    "eye_unimpressed_right": float(0.35),
                    "eyebrow_lowered_left": float(0.12),
                    "eyebrow_lowered_right": float(0.12),
                    "iris_rotation_x": float(0.05),
                    "mouth_aaa": float(mouth_open),
                    "breathing": float(-0.12),  # 叹气呼出
                }
            elif t < 0.75:
                # 视线飘到一边
                p = self._ease_in_out((t - 0.50) / 0.25)
                return {
                    "head_x": float(0.08),
                    "neck_z": float(0.06),
                    "body_z": float(0.04),
                    "eye_unimpressed_left": float(0.35),
                    "eye_unimpressed_right": float(0.35),
                    "iris_rotation_y": float(0.15 * p),   # 视线飘向一侧
                    "iris_rotation_x": float(0.05),
                }
            else:
                # 缓慢回正
                p = self._ease_in_out((t - 0.75) / 0.25)
                return {
                    "head_x": float(0.08 * (1 - p)),
                    "neck_z": float(0.06 * (1 - p)),
                    "body_z": float(0.04 * (1 - p)),
                    "eye_unimpressed_left": float(0.35 * (1 - p)),
                    "eye_unimpressed_right": float(0.35 * (1 - p)),
                    "iris_rotation_y": float(0.15 * (1 - p)),
                    "iris_rotation_x": float(0.05 * (1 - p)),
                }

        return {}


class GestureAnimator:
    """
    肢体动作动画生成器
    
    根据 motion_hint（nod/shake/tilt）和说话状态生成身体动画：
    - 点头/摇头/歪头：由 LLM 返回的 motion_hint 触发，有限次振荡后衰减
    - 说话手势：说话时自动添加细微的身体节奏感（强调动作）
    - 情绪肢体：不同情绪对应不同的身体姿态倾向
    """

    def __init__(self):
        self._active_gesture: str = "none"
        self._gesture_start: float = 0.0
        self._gesture_duration: float = 0.0
        self._last_motion_hint: str = "none"

        # 说话手势：基于语音节奏的强调动作
        self._speak_start_time: float = 0.0
        self._speak_phrase_idx: int = 0  # 第几句话，用于变换动作方向

    def trigger(self, motion_hint: str, now: float):
        """当 LLM 返回新的 motion_hint 时触发动作"""
        if motion_hint == self._last_motion_hint:
            return
        self._last_motion_hint = motion_hint
        if motion_hint == "none":
            return
        self._active_gesture = motion_hint
        self._gesture_start = now
        if motion_hint == "nod":
            self._gesture_duration = 0.8   # 点头 0.8s
        elif motion_hint == "shake":
            self._gesture_duration = 1.0   # 摇头 1.0s
        elif motion_hint in ("tilt_left", "tilt_right"):
            self._gesture_duration = 1.2   # 歪头 1.2s（含回正）
        else:
            self._gesture_duration = 0.6

    def on_speak_start(self, now: float):
        """开始说新一句话时调用"""
        self._speak_start_time = now
        self._speak_phrase_idx += 1

    def _ease_out(self, t: float) -> float:
        """衰减曲线：1 → 0"""
        t = max(0.0, min(1.0, t))
        return 1.0 - t * t

    def sample(self, now: float, speaking_active: bool,
               emotion: str, intensity: float) -> dict:
        """
        返回当前帧的肢体动作参数增量（与 idle 叠加）
        """
        result = {}

        # ---- 1. motion_hint 触发的动作 ----
        if self._active_gesture != "none":
            elapsed = now - self._gesture_start
            if elapsed < self._gesture_duration:
                t = elapsed / self._gesture_duration  # 0→1
                decay = self._ease_out(t)             # 逐渐衰减
                gesture_params = self._sample_gesture(self._active_gesture, elapsed, decay)
                result.update(gesture_params)
            else:
                self._active_gesture = "none"

        # ---- 2. 说话时的身体节奏动作 ----
        if speaking_active:
            speak_t = now - self._speak_start_time
            speak_params = self._sample_speaking_body(speak_t, self._speak_phrase_idx, emotion, intensity)
            # 与手势动作叠加
            for k, v in speak_params.items():
                result[k] = result.get(k, 0.0) + v

        # ---- 3. 情绪肢体偏移 ----
        emotion_body = self._sample_emotion_body(emotion, intensity, now)
        for k, v in emotion_body.items():
            result[k] = result.get(k, 0.0) + v

        return result

    def _sample_gesture(self, gesture: str, elapsed: float, decay: float) -> dict:
        """具体动作的关键帧动画"""
        PI = np.pi

        if gesture == "nod":
            freq = 2.5 * 2 * PI
            amp = 0.22 * decay
            head_x = amp * np.sin(elapsed * freq)
            body_z_nod = 0.06 * decay * np.sin(elapsed * freq * 0.5)
            return {"head_x": float(head_x), "body_z": float(body_z_nod)}

        elif gesture == "shake":
            freq = 2.0 * 2 * PI
            amp = 0.25 * decay
            head_y = amp * np.sin(elapsed * freq)
            body_y = 0.07 * decay * np.sin(elapsed * freq * 0.5)
            return {"head_y": float(head_y), "body_y": float(body_y)}

        elif gesture == "tilt_left":
            t = elapsed / max(0.01, self._gesture_duration)
            if t < 0.4:
                progress = t / 0.4
                tilt = -0.22 * (progress * progress * (3 - 2 * progress))
            else:
                progress = (t - 0.4) / 0.6
                tilt = -0.22 * (1.0 - progress * progress * (3 - 2 * progress))
            return {"neck_z": float(tilt), "body_y": float(tilt * 0.2), "body_z": float(tilt * 0.3)}

        elif gesture == "tilt_right":
            t = elapsed / max(0.01, self._gesture_duration)
            if t < 0.4:
                progress = t / 0.4
                tilt = 0.22 * (progress * progress * (3 - 2 * progress))
            else:
                progress = (t - 0.4) / 0.6
                tilt = 0.22 * (1.0 - progress * progress * (3 - 2 * progress))
            return {"neck_z": float(tilt), "body_y": float(tilt * 0.2), "body_z": float(tilt * 0.3)}

        return {}

    def _sample_speaking_body(self, speak_t: float, phrase_idx: int,
                              emotion: str = "neutral", intensity: float = 0.5) -> dict:
        """
        说话时的细微身体节奏（根据情绪变化风格）
        """
        sign = 1.0 if phrase_idx % 2 == 0 else -1.0

        # 情绪对节奏的影响
        if emotion == "happy":
            speed_mult, amp_mult, lean = 1.3, 1.2, 0.04
            bounce = 0.03 * intensity * np.sin(speak_t * 3.5)
        elif emotion == "sad":
            speed_mult, amp_mult, lean = 0.7, 0.6, 0.07
            bounce = 0.0
        elif emotion == "angry":
            speed_mult, amp_mult, lean = 1.1, 0.8, 0.06
            bounce = 0.0
        elif emotion == "surprised":
            speed_mult, amp_mult, lean = 1.0, 1.0, -0.03
            bounce = 0.0
        else:
            speed_mult, amp_mult, lean = 1.0, 1.0, 0.05
            bounce = 0.0

        # 头部节奏晃动（三频叠加，更自然）
        head_y_sway = sign * 0.07 * amp_mult * np.sin(speak_t * 1.8 * speed_mult + phrase_idx * 0.7)
        head_y_sway += 0.04 * amp_mult * np.sin(speak_t * 2.7 * speed_mult + 1.3)
        head_y_sway += 0.02 * amp_mult * np.sin(speak_t * 0.6 * speed_mult + 0.3)

        head_x_bob = 0.04 * amp_mult * np.sin(speak_t * 2.2 * speed_mult + 0.5) + bounce

        neck_z_sway = sign * 0.03 * amp_mult * np.sin(speak_t * 1.4 * speed_mult + 0.9)

        body_y_sway = sign * 0.04 * amp_mult * np.sin(speak_t * 1.2 * speed_mult)
        body_z_lean = lean * 0.6

        fade_in = min(1.0, speak_t / 0.5)

        return {
            "head_y": float(head_y_sway * fade_in),
            "head_x": float((head_x_bob + lean) * fade_in),
            "neck_z": float(neck_z_sway * fade_in),
            "body_y": float(body_y_sway * fade_in),
            "body_z": float(body_z_lean * fade_in),
        }

    def _sample_emotion_body(self, emotion: str, intensity: float,
                             now: float) -> dict:
        """情绪对应的身体姿态倾向"""
        scale = intensity * 0.8

        if emotion == "happy":
            bounce = 0.03 * scale * np.sin(now * 3.0)
            return {"head_x": float(-0.05 * scale + bounce),
                    "body_z": float(-0.03 * scale),
                    "body_y": float(0.02 * scale * np.sin(now * 2.5))}

        elif emotion == "sad":
            return {"head_x": float(0.08 * scale),
                    "body_z": float(0.05 * scale),
                    "neck_z": float(-0.04 * scale)}

        elif emotion == "angry":
            return {"head_x": float(0.05 * scale),
                    "body_z": float(0.06 * scale),
                    "body_y": float(0.02 * scale)}

        elif emotion == "surprised":
            return {"head_x": float(-0.07 * scale),
                    "body_z": float(-0.05 * scale)}

        elif emotion == "thinking":
            return {"head_x": float(-0.04 * scale),
                    "neck_z": float(0.08 * scale),
                    "body_y": float(0.03 * scale)}

        return {}


class PupilAnimator:
    """
    情绪驱动的瞳孔动态效果

    不同情绪触发独特的瞳孔行为，增强角色的情感表现力：
    - 惊讶：瞳孔急缩 + 微颤（震惊反射）
    - 开心：瞳孔微闪（灵动活泼感）
    - 生气：瞳孔微缩 + 低频颤抖（压抑怒意）
    - 悲伤：视线缓缓下垂 + 飘移
    - 思考：目光缓慢扫视（沉思）
    - 严肃：锁定正前方，极少移动
    """

    def __init__(self):
        self._current_emotion: str = "neutral"
        self._emotion_enter_time: float = 0.0

    def sample(self, now: float, emotion: str, intensity: float) -> dict:
        # 情绪切换时记录时间（用于触发式效果）
        if emotion != self._current_emotion:
            self._current_emotion = emotion
            self._emotion_enter_time = now

        elapsed = now - self._emotion_enter_time
        result = {}

        if emotion == "surprised":
            # ---- 瞳孔急缩 → 缓慢恢复 ----
            if elapsed < 0.10:
                shrink = (elapsed / 0.10) * 0.70  # 快速冲到峰值
            elif elapsed < 0.5:
                shrink = 0.70  # 保持半秒让用户看清
            else:
                decay = max(0.0, 1.0 - (elapsed - 0.5) / 2.0)
                shrink = 0.70 * decay
            # 震惊微颤（高频振荡 0.5s 内衰减）
            jitter = 0.0
            if elapsed < 0.5:
                jitter = 0.05 * (1.0 - elapsed / 0.5) * np.sin(elapsed * 40)
            result["iris_small_left"] = float(shrink)
            result["iris_small_right"] = float(shrink)
            result["iris_rotation_y"] = float(jitter)

        elif emotion == "happy":
            # ---- 灵动微闪：轻微瞳孔摆动（目光活泼）----
            sparkle_x = 0.025 * intensity * np.sin(now * 2.5 + 0.7)
            sparkle_y = 0.020 * intensity * np.sin(now * 3.2 + 1.3)
            result["iris_rotation_x"] = float(sparkle_x)
            result["iris_rotation_y"] = float(sparkle_y)

        elif emotion == "angry":
            # ---- 瞳孔微缩 + 低频颤抖（压抑怒意）----
            shrink = 0.15 * intensity * min(1.0, elapsed / 0.3)
            tremble = 0.015 * intensity * np.sin(now * 14)
            result["iris_small_left"] = float(shrink)
            result["iris_small_right"] = float(shrink)
            result["iris_rotation_y"] = float(tremble)

        elif emotion == "sad":
            # ---- 视线缓缓下垂 + 偶尔飘移 ----
            droop = 0.06 * intensity * min(1.0, elapsed / 1.0)
            drift = 0.03 * intensity * np.sin(now * 0.35 + 0.5)
            result["iris_rotation_x"] = float(droop)
            result["iris_rotation_y"] = float(drift)

        elif emotion == "thinking":
            # ---- 目光缓慢扫视（沉思飘移）----
            sweep_x = -0.06 * intensity * np.sin(now * 0.45)
            sweep_y = -0.08 * intensity * np.sin(now * 0.28 + 0.8)
            result["iris_rotation_x"] = float(sweep_x)
            result["iris_rotation_y"] = float(sweep_y)

        elif emotion == "serious":
            # ---- 极少移动，专注锁定 ----
            micro = 0.008 * np.sin(now * 1.8)
            result["iris_rotation_y"] = float(micro)

        elif emotion == "excited":
            # ---- 灵动跳跃：比 happy 更活泼的睤孔摆动 ----
            bounce_x = 0.04 * intensity * np.sin(now * 4.0 + 0.5)
            bounce_y = 0.035 * intensity * np.sin(now * 5.5 + 1.0)
            result["iris_rotation_x"] = float(bounce_x)
            result["iris_rotation_y"] = float(bounce_y)

        return result


def render_loop(
    state: RenderState,
    state_lock: threading.Lock,
    stop: threading.Event,
    ready: threading.Event,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real-time THA3 rendering.")
    
    print("[render] loading THA3 model... (please wait)")
    
    device = torch.device("cuda")
    model_name = "separable_half"
    
    poser = load_poser(model_name, device)
    dtype = poser.get_dtype()
    
    repo_root = Path(__file__).resolve().parents[1]
    # Default character portrait. Override with env THA_PORTRAIT to point at any
    # 512x512 RGBA PNG that is THA3-ready (see helpers/prepare_tha3_input.py).
    img_path = Path(os.environ.get(
        "THA_PORTRAIT",
        str(repo_root / "data" / "images" / "testPic_tha3_ready.png"),
    ))
    pil = Image.open(img_path).convert("RGBA")
    
    # 降低分辨率到 256x256 提高性能（可选）
    # pil = pil.resize((256, 256), Image.LANCZOS)
    
    torch_image = extract_pytorch_image_from_PIL_image(pil).to(device).to(dtype)
    
    mapper = THAPoseMapper(model_name=model_name, device=device)
    mouth_mapper = PhonemeMouthMapper(pose_mapper=mapper)
    
    # 口型增强系数：不同角色图的嘴部大小/画法差异较大，
    # 调大此值可让口型更明显（默认 1.0，建议范围 1.0 ~ 1.8）
    MOUTH_BOOST = 1.10
    # 短音素（1~2 帧）容易还没张到位就切下一个音素，做轻量补偿
    SHORT_PHONEME_SEC = 0.09
    SHORT_PHONEME_BOOST = 1.20
    # 发音部位分层约束（coarticulation）：
    # strong: 双唇闭合（m/b/p）
    # medium: 塞音/鼻音/边音（d/t/n/l/g/k）
    # light: 擦音/塞擦音（f/h/z/c/s/zh/ch/sh/r/j/q/x）
    STRONG_CONSTRICTION_PHONEMES = {"m", "b", "p"}
    MEDIUM_CONSTRICTION_PHONEMES = {"d", "t", "n", "l", "g", "k"}
    LIGHT_CONSTRICTION_PHONEMES = {
        "f", "h", "z", "c", "s", "zh", "ch", "sh", "r", "j", "q", "x"
    }
    CONSTRICTION_RULES = {
        "strong": {"mul": 0.10, "close_speed": 0.82},
        "medium": {"mul": 0.26, "close_speed": 0.70},
        "light": {"mul": 0.52, "close_speed": 0.56},
        "none": {"mul": 1.00, "close_speed": 0.0},
    }
    
    # 提取所有嘴型参数的索引，方便后续强制更新
    mouth_indices = [idx for name, idx in mapper.name_to_index.items() if 'mouth' in name]
    mouth_indices_set = set(mouth_indices)

    # 提取眼部/眉毛参数索引 — 说话时需要额外平滑以防抽搐
    _EYE_BROW_PREFIXES = ('eye_', 'eyebrow_', 'iris_')
    eye_indices_set = set(
        idx for name, idx in mapper.name_to_index.items()
        if any(name.startswith(p) for p in _EYE_BROW_PREFIXES)
    )
    # 说话时眼部参数每帧最大变化量
    EYE_MAX_DELTA_PER_FRAME = 0.06   # 小抖动级别的限幅
    EYE_SMOOTH_SPEAK = 0.18          # 说话时眼部平滑因子
    EYE_SMOOTH_IDLE  = 0.30          # 非说话时眼部平滑因子

    idle_anim = IdleAnimator()
    gesture_anim = GestureAnimator()
    emotion_smoother = EmotionSmoother()
    pupil_anim = PupilAnimator()

    # 面部追踪（摄像头驱动瞳孔跟随）
    face_tracker = None
    try:
        face_tracker = FaceTracker(camera_index=0, mirror=True)
        idle_anim.set_face_tracker(face_tracker)
    except Exception as e:
        print(f"[FaceTracker] 初始化失败: {e}，瞳孔追踪已禁用")
    
    fps = 60  # 目标 FPS
    target_frame_interval = 1.0 / fps
    current_pose = [0.0] * poser.get_num_parameters()
    prev_speaking_active = False  # 用于检测说话状态切换
    
    # ==========================================
    # 表情参数映射表（更精细的面部表情）
    # ==========================================
    EMOTION_MAP = {
        "neutral": {},
        "happy": {
            "eyebrow_raised_left": 0.25, "eyebrow_raised_right": 0.25,
            "eyebrow_happy_left": 1.0, "eyebrow_happy_right": 1.0,
            "eye_happy_wink_left": 1.0, "eye_happy_wink_right": 1.0,
            "mouth_raised_corner_left": 0.45, "mouth_raised_corner_right": 0.45,
            "mouth_aaa": 0.08,
        },
        "sad": {
            "eyebrow_troubled_left": 0.80, "eyebrow_troubled_right": 0.80,
            "eye_relaxed_left": 0.25, "eye_relaxed_right": 0.25,
            "eye_raised_lower_eyelid_left": 0.30, "eye_raised_lower_eyelid_right": 0.30,
            "mouth_lowered_corner_left": 0.70, "mouth_lowered_corner_right": 0.70,
        },
        "angry": {
            "eyebrow_angry_left": 0.90, "eyebrow_angry_right": 0.90,
            "eyebrow_lowered_left": 0.60, "eyebrow_lowered_right": 0.60,
            "eyebrow_serious_left": 0.40, "eyebrow_serious_right": 0.40,
            "eye_unimpressed_left": 0.45, "eye_unimpressed_right": 0.45,
            "eye_wink_left": 0.18, "eye_wink_right": 0.18,
            "mouth_lowered_corner_left": 0.45, "mouth_lowered_corner_right": 0.45,
            "mouth_delta": 0.35,
        },
        "surprised": {
            "eye_surprised_left": 0.90, "eye_surprised_right": 0.90,
            "eyebrow_raised_left": 0.90, "eyebrow_raised_right": 0.90,
            "mouth_aaa": 0.55,
        },
        "thinking": {
            "eye_relaxed_left": 0.18, "eye_relaxed_right": 0.18,
            "eyebrow_raised_left": 0.45, "eyebrow_raised_right": 0.45,
            "eyebrow_troubled_left": 0.20, "eyebrow_troubled_right": 0.20,
            "mouth_smirk": 0.25,
        },
        "serious": {
            "eyebrow_serious_left": 0.70, "eyebrow_serious_right": 0.70,
            "eyebrow_lowered_left": 0.35, "eyebrow_lowered_right": 0.35,
            "eye_raised_lower_eyelid_left": 0.25, "eye_raised_lower_eyelid_right": 0.25,
        },
        "excited": {
            "eyebrow_raised_left": 0.60, "eyebrow_raised_right": 0.60,
            "eyebrow_happy_left": 0.70, "eyebrow_happy_right": 0.70,
            "eye_surprised_left": 0.25, "eye_surprised_right": 0.25,
            "mouth_raised_corner_left": 0.65, "mouth_raised_corner_right": 0.65,
            "mouth_aaa": 0.25,
        },
    }
    # ==========================================

    cv2.namedWindow("THA_LIP_SYNC", cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty("THA_LIP_SYNC", cv2.WND_PROP_TOPMOST, 1)  # 窗口置顶

    # 屏幕分辨率（用于窗口位置视线修正）
    try:
        import ctypes
        user32 = ctypes.windll.user32
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        print(f"[render] 屏幕分辨率: {screen_w}x{screen_h}")
    except Exception:
        screen_w, screen_h = 1920, 1080

    ready.set()
    print("[render] ready.")

    last_debug_t = 0.0
    
    # 自适应延迟补偿
    render_latency_avg = 0.15  # 初始估计 150ms
    audio_buffer_latency = 0.08  # pygame 音频缓冲约 80ms

    _gc_frame_counter = 0  # 用于定期回收内存

    while not stop.is_set():
        frame_start = time.time()  # 记录帧开始时间
        now = frame_start

        # 更新窗口位置视线修正（每帧，支持拖动窗口后实时响应）
        try:
            win_rect = cv2.getWindowImageRect("THA_LIP_SYNC")
            if win_rect and win_rect[2] > 0:
                idle_anim.update_window_offset(win_rect, (screen_w, screen_h))
        except Exception:
            pass

        with state_lock:
            emotion = state.emotion
            intensity = state.intensity
            motion_hint = state.motion_hint
            speaking_flag = state.speaking
            speak_start = state.speak_start
            speak_end = state.speak_end
            phoneme_frames = state.phoneme_frames
            audio_duration = state.audio_duration
            interaction_state = state.interaction_state
            state_enter_time = state.state_enter_time
            emotion_timeline = state.emotion_timeline

        speaking_active = bool(speaking_flag and (audio_duration <= 0.0 or now - speak_start < audio_duration))

        # 情绪时间线：说话时根据播放进度切换情绪
        if speaking_active and emotion_timeline and len(emotion_timeline) > 1:
            elapsed = now - speak_start
            # 找到当前应该使用的 segment
            cur_emo, cur_int = emotion_timeline[0][1], emotion_timeline[0][2]
            for t_offset, emo, inten in emotion_timeline:
                if elapsed >= t_offset:
                    cur_emo, cur_int = emo, inten
                else:
                    break
            emotion = cur_emo
            intensity = cur_int

        # 自动状态转换：说话结束 → POST_SPEAK → IDLE
        if interaction_state == "speaking" and not speaking_active:
            with state_lock:
                state.interaction_state = "post_speak"
                state.state_enter_time = now
                interaction_state = "post_speak"
                state_enter_time = now
        elif interaction_state == "post_speak" and now - state_enter_time > 0.5:
            with state_lock:
                state.interaction_state = "idle"
                state.state_enter_time = now
                # 回归默认表情
                state.emotion = "neutral"
                state.intensity = 0.0
                interaction_state = "idle"
                emotion = "neutral"
                intensity = 0.0

        # 检测说话状态切换 → 触发手势
        if speaking_active and not prev_speaking_active:
            gesture_anim.on_speak_start(now)
        prev_speaking_active = speaking_active

        # 平滑情绪过渡
        emotion_smoother.set_target(emotion, intensity)

        # 触发 motion_hint 手势动画（替代静态偏移）
        gesture_anim.trigger(motion_hint, now)

        # ===== 核心：音素驱动口型 =====
        is_gap = True  # 默认非说话状态视为"静音"
        active_phoneme_duration = None
        active_phoneme = None
        if speaking_active and phoneme_frames:
            # 使用实时播放位置，而不是估算的时间
            # 这样可以精确同步，包括所有停顿
            with state_lock:
                tts_instance = state.tts
            
            if tts_instance and hasattr(tts_instance, 'get_playback_position'):
                # 获取实际音频播放位置（考虑了所有缓冲和停顿）
                current_audio_time = tts_instance.get_playback_position()
            else:
                # 回退到估算时间
                current_audio_time = now - speak_start

            # 检测是否处于停顿间隙
            is_gap = is_silence_gap(phoneme_frames, current_audio_time, threshold=SILENCE_THRESHOLD)

            if is_gap:
                # 停顿期间，嘴型归零
                mouth_pose = [0.0] * poser.get_num_parameters()
            else:
                # 获取带插值的嘴型 pose
                # 使用实际音频播放位置（而不是估算的时间）
                # 这样可以自动补偿所有延迟（渲染 + 音频缓冲）
                with state_lock:
                    tts_instance = state.tts
                    audio_duration = state.audio_duration  # 获取实际音频时长
                
                if tts_instance and hasattr(tts_instance, 'get_playback_position'):
                    # 获取实际音频播放位置
                    current_audio_time = tts_instance.get_playback_position()
                    
                    # 修复口型延迟：前瞻补偿 (Lookahead Compensation)
                    # 渲染 (~30ms) + 显示缓冲 (~20ms) ≈ 50ms 视觉延迟
                    # 补偿 80ms（略多于实际延迟，留少量余量）
                    current_audio_time += 0.08 
                    
                    # 限制不超过音频总时长
                    current_audio_time = min(current_audio_time, audio_duration)
                else:
                    # 回退到估算时间
                    current_audio_time = now - speak_start
                
                mouth_pose = mouth_mapper.interpolate_mouth_pose(
                    phoneme_frames=phoneme_frames,
                    current_time=current_audio_time,
                )

                # 段末 fade-out：避免最后一个元音（"啊/吗/呢"）口型卡在张大状态。
                # 在最后 120ms 线性收回到闭合。
                _tail_fade = 0.12
                if audio_duration > 0 and current_audio_time > audio_duration - _tail_fade:
                    fade_k = max(0.0, (audio_duration - current_audio_time) / _tail_fade)
                    mouth_pose = [v * fade_k for v in mouth_pose]

                # 长 phoneme safety fade（仅对 duration>0.45s 的"疑似对齐错"
                # 的 phoneme 生效）：CosyVoice 正常 phoneme 是 80~250ms，
                # 超过 450ms 通常是漏字/错位导致一个 phoneme 拖太久，按
                # 绝对时间 fade 避免卡张嘴。正常 phoneme 完全不触发。
                _LONG_THRESH = 0.45
                _LONG_HOLD = 0.30
                _LONG_FADE = 0.10
                for frame in phoneme_frames:
                    if frame.start_time <= current_audio_time < frame.end_time:
                        if frame.duration > _LONG_THRESH:
                            elapsed = current_audio_time - frame.start_time
                            if elapsed > _LONG_HOLD:
                                long_fade_k = max(0.0, 1.0 - (elapsed - _LONG_HOLD) / _LONG_FADE)
                                long_fade_k = min(1.0, long_fade_k)
                                mouth_pose = [v * long_fade_k for v in mouth_pose]
                                if os.environ.get("THA_LIPSYNC_DEBUG"):
                                    _max_v = max(mouth_pose) if mouth_pose else 0.0
                                    print(f"[lipsync] long-fade ph={frame.phoneme} "
                                          f"t={current_audio_time:.2f}s "
                                          f"elapsed={elapsed:.2f}s k={long_fade_k:.2f} "
                                          f"max_pose={_max_v:.2f}")
                        break

                # 记录当前音素时长，用于短音素口型补偿
                for frame in phoneme_frames:
                    if frame.start_time <= current_audio_time < frame.end_time:
                        active_phoneme_duration = frame.duration
                        active_phoneme = str(frame.phoneme).lower().strip()
                        break
        else:
            # 非说话状态：中性嘴型
            mouth_pose = [0.0] * poser.get_num_parameters()

        # 构建目标 pose（motion_hint 由 GestureAnimator 处理，不再传给 mapper）
        llm_state = {
            "emotion": "neutral",
            "intensity": 0.0,
            "motion_hint": "none",
        }
        target_pose = mapper.build_pose_vector(llm_state, mouth_open=0.0)

        # 合并嘴型 pose (修复标点符号不闭嘴的问题)
        # 强制更新嘴型参数，即使是 0.0（闭嘴）也要应用，防止嘴巴卡住
        mouth_boost = MOUTH_BOOST
        if active_phoneme_duration is not None and active_phoneme_duration < SHORT_PHONEME_SEC:
            mouth_boost *= SHORT_PHONEME_BOOST
        for i in mouth_indices:
            target_pose[i] = min(mouth_pose[i] * mouth_boost, 1.0)

        # 发音收口约束：按发音部位给不同级别的收口强度
        constriction_level = "none"
        if speaking_active and not is_gap and active_phoneme:
            if active_phoneme in STRONG_CONSTRICTION_PHONEMES:
                constriction_level = "strong"
            elif active_phoneme in MEDIUM_CONSTRICTION_PHONEMES:
                constriction_level = "medium"
            elif active_phoneme in LIGHT_CONSTRICTION_PHONEMES:
                constriction_level = "light"
        if constriction_level != "none":
            close_mul = CONSTRICTION_RULES[constriction_level]["mul"]
            for pname in ("mouth_aaa", "mouth_iii", "mouth_uuu", "mouth_eee", "mouth_ooo"):
                idx = mapper.name_to_index.get(pname)
                if idx is not None:
                    target_pose[idx] *= close_mul

        # ============================================================
        # 应用表情参数（带平滑过渡）
        # ============================================================
        old_emo, old_int, new_emo, new_int = emotion_smoother.sample()

        # 对 intensity 做非线性提升，避免中等值表情太弱
        old_int_curved = _intensity_curve(old_int)
        new_int_curved = _intensity_curve(new_int)

        # 表情不应该控制开闭口（mouth_aaa/iii/uuu/eee/ooo）——开闭口完全
        # 由 lip-sync 决定。否则在静音/停顿瞬间，发音口型归零后，表情会
        # 把 mouth_aaa 顶回去，看起来角色"默认张嘴"。
        _MOUTH_OPEN_PARAMS = {"mouth_aaa", "mouth_iii", "mouth_uuu",
                              "mouth_eee", "mouth_ooo", "mouth_delta"}

        # 淡出旧表情
        old_params = EMOTION_MAP.get(old_emo, {})
        for param_name, value in old_params.items():
            if param_name in _MOUTH_OPEN_PARAMS:
                continue
            idx = mapper.name_to_index.get(param_name)
            if idx is not None:
                target_pose[idx] += value * old_int_curved

        # 淡入新表情
        new_params = EMOTION_MAP.get(new_emo, {})
        for param_name, value in new_params.items():
            if param_name in _MOUTH_OPEN_PARAMS:
                continue
            idx = mapper.name_to_index.get(param_name)
            if idx is not None:
                target_pose[idx] += value * new_int_curved

        # 思考状态额外微表情（单独处理，不依赖 LLM emotion）
        if interaction_state == "thinking":
            think_params = EMOTION_MAP.get("thinking", {})
            # 渐入（状态进入后 0.5s 内渐入）
            think_fade = min(1.0, (now - state_enter_time) / 0.5)
            for param_name, value in think_params.items():
                idx = mapper.name_to_index.get(param_name)
                if idx is not None:
                    target_pose[idx] = max(target_pose[idx], value * think_fade * 0.5)

        # POST_SPEAK 微笑（说完话后的微微放松表情）
        if interaction_state == "post_speak":
            ps_elapsed = now - state_enter_time
            ps_fade = max(0.0, 1.0 - ps_elapsed / 0.5)  # 0.5s 内淡出
            for pname, val in [("mouth_raised_corner_left", 0.25),
                               ("mouth_raised_corner_right", 0.25),
                               ("eye_happy_wink_left", 0.08),
                               ("eye_happy_wink_right", 0.08)]:
                idx = mapper.name_to_index.get(pname)
                if idx is not None:
                    target_pose[idx] += val * ps_fade

        # 添加 idle 动画（状态感知）
        idle = idle_anim.sample(now, speaking_active=speaking_active,
                                interaction_state=interaction_state)

        # 表情已经闭眼时，抑制眨眼动画（避免闭眼还在眨）
        _eye_close_params = ("eye_happy_wink_left", "eye_happy_wink_right",
                             "eye_relaxed_left", "eye_relaxed_right",
                             "eye_wink_left", "eye_wink_right")
        _eyes_already_closed = any(
            target_pose[mapper.name_to_index[p]] > 0.35
            for p in _eye_close_params
            if p in mapper.name_to_index
        )
        if _eyes_already_closed:
            idle.pop("eye_wink_left", None)
            idle.pop("eye_wink_right", None)

        add_pose(target_pose, idle, mapper.name_to_index, mapper.name_to_range)

        # 添加肢体手势动画
        gesture = gesture_anim.sample(now, speaking_active, emotion, intensity)
        add_pose(target_pose, gesture, mapper.name_to_index, mapper.name_to_range)

        # 情绪驱动的瞳孔效果（惊讶缩瞳、开心灵动、生气紧盯等）
        pupil_fx = pupil_anim.sample(now, emotion, intensity)
        if not getattr(idle_anim, '_face_active', False):
            add_pose(target_pose, pupil_fx, mapper.name_to_index, mapper.name_to_range)

        # 平滑插值（说话中：开口快、闭口慢；静音时：快速闭合）
        if speaking_active and not is_gap:
            # 释放相位：从 strong/medium 收口切到非收口时，给一次更快开口
            prev_constriction = getattr(render_loop, '_prev_constriction', "none")
            release_boost = prev_constriction in ("strong", "medium") and constriction_level == "none"

            if constriction_level != "none":
                # 收口相位：按级别快速收口，确保过渡可见
                close_speed = CONSTRICTION_RULES[constriction_level]["close_speed"]
                for i in mouth_indices:
                    current_pose[i] = current_pose[i] + (target_pose[i] - current_pose[i]) * close_speed
            else:
                open_speed = 0.75 if release_boost else 0.65
                for i in mouth_indices:
                    delta = target_pose[i] - current_pose[i]
                    if delta >= 0:
                        # 开口——跟踪目标口型
                        current_pose[i] = current_pose[i] + delta * open_speed
                    else:
                        # 说话中口型切换（如 a→x→ian）：需要够快体现不同音素
                        current_pose[i] = current_pose[i] + delta * 0.58
            for i in range(len(current_pose)):
                if i not in mouth_indices_set:
                    if i in eye_indices_set:
                        # 眼部参数：自适应平滑
                        # 大变化（情绪切换）放行，小抖动（THA3 耦合）限幅
                        raw_diff = target_pose[i] - current_pose[i]
                        if abs(raw_diff) < 0.03:
                            # 接近目标：直接 snap，避免永远差一点点
                            current_pose[i] = target_pose[i]
                        else:
                            delta = raw_diff * EYE_SMOOTH_SPEAK
                            if abs(raw_diff) < 0.25:
                                delta = max(-EYE_MAX_DELTA_PER_FRAME, min(EYE_MAX_DELTA_PER_FRAME, delta))
                            current_pose[i] = current_pose[i] + delta
                    else:
                        current_pose[i] = current_pose[i] + (target_pose[i] - current_pose[i]) * 0.35
            render_loop._prev_constriction = constriction_level
        else:
            # 停顿/非说话：快速回到闭嘴
            for i in mouth_indices:
                current_pose[i] = current_pose[i] + (target_pose[i] - current_pose[i]) * 0.72
            for i in range(len(current_pose)):
                if i not in mouth_indices_set:
                    if i in eye_indices_set:
                        # 眼部参数：非说话时平滑（无严格限速）
                        raw_diff = target_pose[i] - current_pose[i]
                        if abs(raw_diff) < 0.03:
                            current_pose[i] = target_pose[i]
                        else:
                            current_pose[i] = current_pose[i] + raw_diff * EYE_SMOOTH_IDLE
                    else:
                        current_pose[i] = current_pose[i] + (target_pose[i] - current_pose[i]) * 0.4
            render_loop._prev_constriction = "none"

        # 面部追踪：按 face_blend 混合写入 head + iris 参数
        # face_blend 平滑过渡，避免检测到/丢失时跳变
        # 待机表演播放时降低追踪权重，让表演动作可见
        fb = getattr(idle_anim, '_face_blend', 0.0)
        perf_playing = getattr(idle_anim, '_idle_perf', None) and idle_anim._idle_perf.is_playing
        if perf_playing:
            fb *= 0.15  # 表演时只保留 15% 追踪（让表演主导头部动作）
        if fb > 0.001:
            for pname, val in [
                ("head_y", idle_anim._face_head_y),
                ("head_x", idle_anim._face_head_x),
                ("iris_rotation_x", idle_anim._face_iris_x + pupil_fx.get("iris_rotation_x", 0)),
                ("iris_rotation_y", idle_anim._face_iris_y + pupil_fx.get("iris_rotation_y", 0)),
            ]:
                idx = mapper.name_to_index.get(pname)
                if idx is not None:
                    current_pose[idx] = current_pose[idx] * (1 - fb) + val * fb
            # 瞳孔缩放（惊讶等情绪效果）
            for k in ("iris_small_left", "iris_small_right"):
                if k in pupil_fx:
                    idx = mapper.name_to_index.get(k)
                    if idx is not None:
                        current_pose[idx] = pupil_fx[k]

        # ===== lipsync 实测峰值打印（每个音素只打印一次）=====
        # 在 current_pose 已经合成完毕（包括 smoothing/boost）但尚未渲染前，
        # 累计当前 active_phoneme 的实际口型峰值；当音素切换或说话结束时，
        # 把上一段累计的最大值打印出来 — 用来比对 [lipsync] timeline 表里的
        # "请求"值，看动画是否真的张到位了。
        if os.environ.get("THA_LIPSYNC_DEBUG"):
            _LS_PARAMS = ("mouth_aaa", "mouth_iii", "mouth_uuu",
                          "mouth_eee", "mouth_ooo")
            track = getattr(render_loop, "_ls_track", None)
            if track is None:
                track = {"ph": None, "max": {}, "dur": 0.0}
                render_loop._ls_track = track

            cur_ph = active_phoneme if (speaking_active and not is_gap) else None

            if cur_ph != track["ph"]:
                # flush 上一段
                if track["ph"] is not None and track["max"]:
                    parts = " ".join(
                        f"{p[6:].upper()}={track['max'].get(p, 0.0):.2f}"
                        for p in _LS_PARAMS
                    )
                    print(f"[ls-peak] ph={track['ph']:<6s} dur={track['dur']:.2f}s "
                          f"actual: {parts}")
                track["ph"] = cur_ph
                track["max"] = {}
                track["dur"] = active_phoneme_duration or 0.0

            if cur_ph is not None:
                for _name in _LS_PARAMS:
                    _idx = mapper.name_to_index.get(_name)
                    if _idx is not None:
                        v = current_pose[_idx]
                        if v > track["max"].get(_name, 0.0):
                            track["max"][_name] = v

        # 推理渲染
        pose_tensor = mapper.to_tensor(current_pose)

        with torch.inference_mode():
            out = poser.pose(torch_image, pose_tensor)[0].detach()
        # NOTE: do NOT call torch.cuda.synchronize() here. The .cpu() below is
        # already a sync point. An explicit sync makes every render frame block
        # on the entire GPU queue, so any background CUDA work (e.g. WhisperX
        # forced alignment) directly stalls the renderer and causes stutter.

        # 转换为 numpy 并显示
        t2 = time.time()
        
        # out 是 [4,512,512] tensor，值范围 -1 到 1
        # 转 float32 避免 float16 numpy 分配碎片化
        img = out.float().clamp(0, 1) if out.min() >= -0.01 else ((out.float().clamp(-1, 1) + 1) / 2)
        img = torch.pow(img, 1/2.2)  # gamma 校正
        
        # 转 numpy [H,W,C] float32
        img_np = img.permute(1, 2, 0).cpu().numpy()  # [512,512,4] float32
        del img, out  # 立即释放 tensor

        # Alpha 合成到白色背景，直接输出 uint8
        alpha = img_np[:, :, 3:4]
        rgb = img_np[:, :, :3]
        composited = rgb * alpha + (1.0 - alpha)   # 白底合成
        img_bgr = (composited * 255).clip(0, 255).astype(np.uint8)
        del img_np, alpha, rgb, composited  # 立即释放
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

        # Post-upscale for display comfort (does not change model render resolution)
        if DISPLAY_UPSCALE != 1.0:
            disp_w = int(img_bgr.shape[1] * DISPLAY_UPSCALE)
            disp_h = int(img_bgr.shape[0] * DISPLAY_UPSCALE)
            display_frame = cv2.resize(
                img_bgr,
                (disp_w, disp_h),
                interpolation=DISPLAY_INTERPOLATION,
            )
        else:
            display_frame = img_bgr

        cv2.imshow("THA_LIP_SYNC", display_frame)
        del display_frame  # 释放上一帧
        time_convert = time.time() - t2

        # 定期回收内存（每 200 帧 ≈ 10 秒）
        _gc_frame_counter += 1
        if _gc_frame_counter % 200 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        # 更新平均渲染延迟（低通滤波，平滑波动）
        frame_time = (time.time() - frame_start) * 1000  # 当前帧耗时 ms
        current_render_latency = frame_time / 1000.0
        # 使用更大的权重（0.2），让补偿值更快收敛
        render_latency_avg = render_latency_avg * 0.8 + current_render_latency * 0.2

        # 每 30 秒显示一次性能
        if now - last_debug_t > 30.0:
            print(f"[perf] 渲染={render_latency_avg*1000:.0f}ms 情绪={emotion}")
            last_debug_t = now

        if cv2.waitKey(1) & 0xFF == ord("q"):  # 不阻塞等待
            stop.set()
            break

        # 帧率上限：避免渲染独占 GPU，让 TTS / WhisperX 有算力窗口
        elapsed = time.time() - frame_start
        sleep_time = target_frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    if face_tracker is not None:
        face_tracker.stop()
    cv2.destroyAllWindows()

# Input loop is used to handle the user's input, then generate TTS and phoneme alignment, and update the shared state for rendering.
def input_loop(
    # State and its lock, the input loop need to know the state for the character
    state: RenderState,
    state_lock: threading.Lock,
    # Control events
    stop: threading.Event,
    ready: threading.Event,
    initial_input_mode: str = "mic",
):
    # Get a LLM client instance (with conversation history support)
    llm = LLMApiClient(max_history=20)

    aligner = AudioPhonemeAligner(language="auto")  # 自动检测语言
    selected_voice_key: str | None = None  # None = 自动按语言选择

    # ---- TTS engine selection ---------------------------------------------
    # ``edge`` (default) keeps the existing preset-voice pipeline.
    # ``xtts`` and ``gpt_sovits`` switch to voice-cloning. Forced alignment
    # runs in a background thread so audio playback is never delayed; the
    # precise phoneme frames are hot-swapped into ``state.phoneme_frames``
    # once alignment finishes (typically within the first 1-2s of speech).
    use_xtts = (TTS_ENGINE == "xtts")
    use_gpt_sovits = (TTS_ENGINE == "gpt_sovits")
    use_cosyvoice = (TTS_ENGINE == "cosyvoice")
    xtts_singleton = None        # type: ignore  # XTTSClient, lazy-loaded
    gpt_sovits_singleton = None  # type: ignore  # GPTSoVITSClient, lazy-loaded
    cosyvoice_singleton = None   # type: ignore  # CosyVoiceClient, lazy-loaded
    forced_aligner = None        # type: ignore  # ForcedAligner, lazy-loaded
    if use_xtts:
        try:
            from ai_runtime.xtts_client import XTTSClient
            from ai_runtime.forced_aligner import ForcedAligner
            print(f"[tts] Engine = XTTS v2 (voice cloning)")
            print(f"[tts] Reference voice: {XTTS_REFERENCE_WAV}")
            if not os.path.isfile(XTTS_REFERENCE_WAV):
                print(f"[tts] WARNING: reference wav not found, falling back to Edge TTS")
                use_xtts = False
            else:
                # Load XTTS now so first user turn is fast.
                xtts_singleton = XTTSClient(speaker_wav=XTTS_REFERENCE_WAV, language="auto")
                forced_aligner = ForcedAligner()

                # Pre-warm both models in a background thread so by the time
                # the user types their first message, model weights are
                # already on the GPU. Without this, the first turn pays a
                # ~17s XTTS load + ~6s WhisperX load before any audio comes
                # out.
                def _prewarm_tts_models(_xtts=xtts_singleton, _aligner=forced_aligner):
                    try:
                        t0 = time.time()
                        _xtts._get_model()
                        print(f"[tts] XTTS prewarmed in {time.time()-t0:.1f}s")
                    except Exception as e:
                        print(f"[tts] XTTS prewarm failed: {e}")
                    try:
                        t0 = time.time()
                        _aligner._load_align_model("zh")
                        print(f"[tts] WhisperX(zh) prewarmed in {time.time()-t0:.1f}s")
                    except Exception as e:
                        print(f"[tts] WhisperX(zh) prewarm failed: {e}")
                    try:
                        t0 = time.time()
                        _aligner._load_align_model("en")
                        print(f"[tts] WhisperX(en) prewarmed in {time.time()-t0:.1f}s")
                    except Exception as e:
                        print(f"[tts] WhisperX(en) prewarm failed: {e}")

                threading.Thread(target=_prewarm_tts_models, daemon=True).start()
        except Exception as e:
            print(f"[tts] XTTS init failed ({e}); falling back to Edge TTS")
            use_xtts = False
    elif use_gpt_sovits:
        try:
            from ai_runtime.gpt_sovits_client import GPTSoVITSClient
            from ai_runtime.forced_aligner import ForcedAligner
            print(f"[tts] Engine = GPT-SoVITS (local API voice cloning)")
            print(f"[tts] Reference voice: {GPT_SOVITS_REFERENCE_WAV}")
            if not os.path.isfile(GPT_SOVITS_REFERENCE_WAV):
                print(f"[tts] WARNING: reference wav not found, falling back to Edge TTS")
                use_gpt_sovits = False
            else:
                gpt_sovits_singleton = GPTSoVITSClient(
                    speaker_wav=GPT_SOVITS_REFERENCE_WAV,
                    language="auto",
                )
                forced_aligner = ForcedAligner()
                # Pre-warm zh + en alignment models so first turn (and a
                # mid-conversation language switch) does not stall the render.
                def _prewarm_aligner(_aligner=forced_aligner):
                    try:
                        t0 = time.time()
                        _aligner._load_align_model("zh")
                        print(f"[tts] WhisperX(zh) prewarmed in {time.time()-t0:.1f}s")
                    except Exception as e:
                        print(f"[tts] WhisperX(zh) prewarm failed: {e}")
                    try:
                        t0 = time.time()
                        _aligner._load_align_model("en")
                        print(f"[tts] WhisperX(en) prewarmed in {time.time()-t0:.1f}s")
                    except Exception as e:
                        print(f"[tts] WhisperX(en) prewarm failed: {e}")

                threading.Thread(target=_prewarm_aligner, daemon=True).start()
        except Exception as e:
            print(f"[tts] GPT-SoVITS init failed ({e}); falling back to Edge TTS")
            use_gpt_sovits = False
    elif use_cosyvoice:
        try:
            from ai_runtime.cosyvoice_client import CosyVoiceClient
            cosyvoice_singleton = CosyVoiceClient(language="auto")
            print(f"[tts] Engine = CosyVoice (Aliyun DashScope cloud)")
            print(f"[tts] Voice = {cosyvoice_singleton.voice}, Model = {cosyvoice_singleton.model}")
            # CosyVoice v2 / v3 复刻音色原生返回 word_timestamp，无需 WhisperX
            # 兜底；偶尔拿不到时会回退到 audio_phoneme_aligner 的权重估算。
            # 不在这里加载 ForcedAligner，省 GPU 和启动时间。
            # 如确需启用：set THA_COSYVOICE_USE_WHISPERX=1
            if os.environ.get("THA_COSYVOICE_USE_WHISPERX", "").lower() in ("1", "true", "yes"):
                from ai_runtime.forced_aligner import ForcedAligner
                forced_aligner = ForcedAligner()

                def _prewarm_aligner_cv(_aligner=forced_aligner):
                    try:
                        t0 = time.time()
                        _aligner._load_align_model("zh")
                        print(f"[tts] WhisperX(zh) prewarmed in {time.time()-t0:.1f}s")
                    except Exception as e:
                        print(f"[tts] WhisperX(zh) prewarm failed: {e}")
                    try:
                        t0 = time.time()
                        _aligner._load_align_model("en")
                        print(f"[tts] WhisperX(en) prewarmed in {time.time()-t0:.1f}s")
                    except Exception as e:
                        print(f"[tts] WhisperX(en) prewarm failed: {e}")

                threading.Thread(target=_prewarm_aligner_cv, daemon=True).start()
        except Exception as e:
            print(f"[tts] CosyVoice init failed ({e}); falling back to Edge TTS")
            use_cosyvoice = False
    if not use_xtts:
        if not use_gpt_sovits:
            if not use_cosyvoice:
                print(f"[tts] Engine = Edge TTS (preset voices)")

    # ---- 输入模式初始化 ----
    mic_mode = (initial_input_mode == "mic")
    stt: STTClient | None = None

    def _init_stt_if_needed() -> bool:
        nonlocal stt
        if stt is not None:
            return True
        try:
            stt = STTClient(device="cpu", compute_type="int8")
            print("[stt] Whisper 模型加载完成，语音输入已就绪")
            return True
        except Exception as e:
            print(f"[stt] 初始化失败: {e}")
            return False

    if mic_mode and not _init_stt_if_needed():
        print("[stt] 回退到文字输入")
        mic_mode = False

    ready.wait()
    print("[info] 命令: /mic 切换语音/文字  /mode text|mic  /voices 查看音色  /voice <name> 切换音色")
    if mic_mode:
        print("[mic] 语音模式 — 请说话（静音 1.2s 自动结束，最长 15s）")
    else:
        print("[mic] 文字模式 — 输入文本开始对话")

    while not stop.is_set():
        user_text = ""

        if mic_mode and stt:
            # ---- 语音输入 ----
            print("\n🎤 listening...", end="", flush=True)
            with state_lock:
                state.interaction_state = "listening"
                state.state_enter_time = time.time()
            try:
                user_text = stt.listen_and_transcribe(language=None, max_seconds=15.0)
            except Exception as e:
                print(f"\r[stt] 录音错误: {e}")
                continue
            if not user_text:
                print("\r                ", end="\r")  # 清掉 listening 提示
                with state_lock:
                    state.interaction_state = "idle"
                    state.state_enter_time = time.time()
                continue
            print(f"\rYou: {user_text}")
        else:
            # ---- 文字输入 ----
            try:
                user_text = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                stop.set()
                break
            except Exception as e:
                print(f"[input] Error: {e}")
                stop.set()
                break

            if not user_text:
                continue

        # ---- 斜杠命令 ----
        if user_text.lower() == "/mic":
            if not mic_mode and not _init_stt_if_needed():
                print("[stt] 模型未加载，保持文字模式")
            else:
                mic_mode = not mic_mode
                if mic_mode:
                    print("[mic] 语音模式 — 请说话")
                else:
                    print("[mic] 文字模式 — 请打字")
            continue

        if user_text.lower().startswith("/mode "):
            mode = user_text[6:].strip().lower()
            if mode not in ("mic", "text"):
                print("[mode] 用法: /mode text 或 /mode mic")
                continue
            if mode == "mic":
                if not _init_stt_if_needed():
                    print("[stt] 模型未加载，无法切换到语音模式")
                    continue
                mic_mode = True
                print("[mic] 语音模式 — 请说话")
            else:
                mic_mode = False
                print("[mic] 文字模式 — 请打字")
            continue

        if user_text.lower() in ("/voices", "/voice list"):
            print("\n=== 可用音色 ===")
            print(list_voices())
            cur = selected_voice_key or "auto"
            print(f"\n当前: {cur}  (输入 /voice <name> 切换, /voice auto 恢复自动)\n")
            continue

        if user_text.lower().startswith("/voice "):
            arg = user_text[7:].strip().lower()
            if arg == "auto":
                selected_voice_key = None
                print("[voice] 已切换为自动选择模式")
            elif arg in VOICE_CATALOG:
                selected_voice_key = arg
                info = VOICE_CATALOG[arg]
                print(f"[voice] 已切换为: {info['desc']} ({info['id']})")
            else:
                print(f"[voice] 未知音色 '{arg}'，输入 /voices 查看可用列表")
            continue

        # ---- 用户输入 → LISTENING 状态（文字模式短暂确认）----
        if not mic_mode:
            with state_lock:
                state.interaction_state = "listening"
                state.state_enter_time = time.time()
            time.sleep(0.4)  # 短暂"倾听确认"

        # ---- 开始处理 → THINKING 状态 ----
        with state_lock:
            state.interaction_state = "thinking"
            state.state_enter_time = time.time()

        # LLM 生成回复（带对话历史）
        llm_json = llm.chat(user_text)
        segments = llm_json.get("segments", [])
        reply = llm_json.get("reply", "") or "".join(s["text"] for s in segments)

        print("AI:", reply)

        # 检查 LLM 是否要求切换音色
        voice_change = llm_json.get("voice_change")
        if voice_change and isinstance(voice_change, str):
            vc = voice_change.strip().lower()
            if vc == "auto":
                selected_voice_key = None
            elif vc in VOICE_CATALOG:
                selected_voice_key = vc

        # 根据回复内容选择语音
        detected_language = aligner._detect_language(reply)

        if use_xtts and xtts_singleton is not None:
            # ---- XTTS voice-cloning path -----------------------------------
            # No preset voice catalog; we always use the cloned reference voice.
            # Word boundaries come from a background WhisperX alignment pass.
            xtts_singleton.language = detected_language
            tts = xtts_singleton
        elif use_gpt_sovits and gpt_sovits_singleton is not None:
            # ---- GPT-SoVITS voice-cloning path -----------------------------
            # Same as XTTS path from lip-sync perspective: synth now, then
            # refine word boundaries via background WhisperX forced alignment.
            gpt_sovits_singleton.language = detected_language
            tts = gpt_sovits_singleton
        elif use_cosyvoice and cosyvoice_singleton is not None:
            # ---- CosyVoice (cloud) path -----------------------------------
            # Aliyun DashScope cloud TTS. Word boundaries via WhisperX forced
            # alignment on the returned wav, identical to gpt_sovits path.
            cosyvoice_singleton.language = detected_language
            tts = cosyvoice_singleton
        else:
            if selected_voice_key:
                voice_info = VOICE_CATALOG[selected_voice_key]
                # 如果选中的音色语言与回复语言不匹配，回退到该语言的默认音色
                if voice_info["lang"] != detected_language:
                    fallback_key = DEFAULT_VOICE_KEY.get(detected_language, "xiaoxiao")
                    voice_id, voice_lang = resolve_voice(fallback_key, detected_language)
                else:
                    voice_id, voice_lang = resolve_voice(selected_voice_key, detected_language)
            else:
                default_key = DEFAULT_VOICE_KEY.get(detected_language, "xiaoxiao")
                voice_id, voice_lang = resolve_voice(default_key, detected_language)
            tts = TTSClient(voice=voice_id, language=detected_language)

        # ===== 流式分段合成 + 顺序播放 =====
        # 把 LLM 返回的 segments 交给后台 producer 线程：逐段做 TTS + 词级
        # 对齐，结果放入队列。主线程从队列取一段立刻播放，下一段在
        # producer 那边并行合成。这样长回复不再需要等整段合成完才能开口，
        # "首音延迟"始终≈一个最短段的合成时间。
        valid_segs = [
            s for s in segments
            if isinstance(s, dict) and (s.get("text") or "").strip()
        ]
        if not valid_segs:
            valid_segs = [{"text": reply, "emotion": "neutral", "intensity": 0.5}]

        motion_hint = llm_json.get("motion_hint", "none")
        seg_queue: "queue.Queue" = queue.Queue()
        _SENTINEL = object()

        # 并发预合成：多段同时跑 TTS 网络请求，按原顺序入队。
        # CosyVoice 云端单段 4-8s 但播放只 2-3s，串行会导致段间空挡。
        # 默认 3 路并发，可用 THA_TTS_PARALLEL 调。Edge TTS 不并行（=1）。
        try:
            _default_par = "3" if (use_cosyvoice or use_xtts or use_gpt_sovits) else "1"
            _par = max(1, int(os.environ.get("THA_TTS_PARALLEL", _default_par)))
        except ValueError:
            _par = 1

        def _synth_one(seg, _tts=tts, _lang0=detected_language):
            """单段合成 + 对齐 + 生成 phoneme_frames，纯函数返回 dict。"""
            seg_text = (seg.get("text") or "").strip()
            if not seg_text:
                return None
            seg_lang = aligner._detect_language(seg_text) or _lang0
            # 注意：language 是共享状态，并发时简单赋值即可（CosyVoice 在
            # _generate_full 内部不读 self.language；GPT-SoVITS 读取，因此
            # GPT-SoVITS 路径建议把 THA_TTS_PARALLEL=1）。
            if hasattr(_tts, "language"):
                _tts.language = seg_lang
            t0 = time.time()
            try:
                if hasattr(_tts, "generate_audio_full"):
                    ap, dur, boundaries = _tts.generate_audio_full(seg_text)
                    boundaries = list(boundaries or [])
                else:
                    ap, dur = _tts.generate_audio(seg_text)
                    boundaries = list(getattr(_tts, "word_boundaries", []) or [])
            except Exception as e:
                print(f"[stream] TTS failed: {e}")
                return None
            tts_ms = (time.time() - t0) * 1000
            if not ap:
                return None
            need_whisperx = (
                (use_xtts or use_gpt_sovits)
                and forced_aligner is not None
                and not boundaries
            )
            if need_whisperx:
                t1 = time.time()
                try:
                    from ai_runtime.forced_aligner import _load_audio_16k
                    audio_np = _load_audio_16k(ap)
                    boundaries = forced_aligner.align(
                        ap, seg_text, seg_lang, audio=audio_np
                    ) or boundaries
                except Exception as e:
                    print(f"[stream] align failed: {e}")
                align_ms = (time.time() - t1) * 1000
            else:
                align_ms = 0
            try:
                alignment = aligner.align_text_to_audio(
                    seg_text, ap, dur, word_boundaries=boundaries
                )
                phoneme_frames = alignment.phoneme_frames
            except Exception as e:
                print(f"[stream] phoneme estimate failed: {e}")
                phoneme_frames = []
            print(f"[stream] seg ready: tts={tts_ms:.0f}ms align={align_ms:.0f}ms "
                  f"dur={dur:.2f}s text={seg_text[:20]}")

            # Lip-sync 调试: 打印每个 phoneme 的口型映射, 用于核对
            # "音素 → 嘴型" 是否符合预期。开关: THA_LIPSYNC_DEBUG=1
            if os.environ.get("THA_LIPSYNC_DEBUG") and phoneme_frames:
                from ai_runtime.phoneme_mouth_mapper import (
                    ENGLISH_PHONEME_TO_MOUTH, CHINESE_PINYIN_TO_MOUTH,
                )
                _MOUTH_KEYS = ('mouth_aaa', 'mouth_iii', 'mouth_uuu',
                               'mouth_eee', 'mouth_ooo')

                def _ph_to_params(ph: str):
                    p = ph.lower().strip()
                    for tbl in (ENGLISH_PHONEME_TO_MOUTH, CHINESE_PINYIN_TO_MOUTH):
                        if p in tbl:
                            return tbl[p]
                    base = p.rstrip('0123456789')
                    for tbl in (ENGLISH_PHONEME_TO_MOUTH, CHINESE_PINYIN_TO_MOUTH):
                        if base in tbl:
                            return tbl[base]
                    return {}

                print(f"[lipsync] === phoneme timeline for: {seg_text} ===")
                print(f"[lipsync] {'time':>11s}  {'ph':<6s}  "
                      f"{'A':>4s} {'I':>4s} {'U':>4s} {'E':>4s} {'O':>4s}  shape")
                for fr in phoneme_frames:
                    params = _ph_to_params(fr.phoneme)
                    vals = [params.get(k, 0.0) for k in _MOUTH_KEYS]
                    if max(vals) < 0.3:
                        shape = '(闭)' if fr.phoneme.startswith('sil') else '(弱)'
                    else:
                        shape = 'AIUEO'[vals.index(max(vals))]
                    print(f"[lipsync] {fr.start_time:5.2f}-{fr.end_time:5.2f}  "
                          f"{fr.phoneme:<6s}  "
                          f"{vals[0]:4.2f} {vals[1]:4.2f} {vals[2]:4.2f} "
                          f"{vals[3]:4.2f} {vals[4]:4.2f}  {shape}")
            return {
                "audio_path": ap,
                "duration": dur,
                "phoneme_frames": phoneme_frames,
                "boundaries": boundaries,
                "emotion": seg.get("emotion", "neutral"),
                "intensity": float(seg.get("intensity", 0.5)),
                "text": seg_text,
            }

        def _produce_segments(_segs=valid_segs):
            if _par <= 1 or len(_segs) <= 1:
                for seg in _segs:
                    item = _synth_one(seg)
                    if item is not None:
                        seg_queue.put(item)
                seg_queue.put(_SENTINEL)
                return
            # 优先级调度 + 流水线深度=1：第一段独占带宽先合成（首音延迟
            # 最优），合成完毕立刻入队让主循环开播；之后剩余段也用单 worker
            # 串行合成，但 producer 不等播放完成就开工 —— 借「上一段正在
            # 播放」的时间掩盖「下一段网络合成」的延迟。
            #   - N 路并发会让多段抢 ws 连接 + 服务端排队，单段延迟翻倍，
            #     反而拖慢首段（甚至中段）出声 → 出现"乱序完成"。
            #   - 单 worker 串行 + producer 抢跑，每段都独享带宽，延迟最小。
            from concurrent.futures import ThreadPoolExecutor
            first_item = _synth_one(_segs[0])
            if first_item is not None:
                seg_queue.put(first_item)
            rest = _segs[1:]
            if not rest:
                seg_queue.put(_SENTINEL)
                return
            # max_workers=1：永远只有一段在合成，下一段等上段完成才起
            with ThreadPoolExecutor(max_workers=1,
                                    thread_name_prefix="tts-synth") as pool:
                futures = [pool.submit(_synth_one, seg) for seg in rest]
                for fut in futures:
                    try:
                        item = fut.result()
                    except Exception as e:
                        print(f"[stream] synth future failed: {e}")
                        continue
                    if item is not None:
                        seg_queue.put(item)
            seg_queue.put(_SENTINEL)

        threading.Thread(target=_produce_segments, daemon=True).start()

        first = True
        while True:
            item = seg_queue.get()
            if item is _SENTINEL:
                break
            # 先提交播放，拿到 (start_evt, end_evt)。CosyVoice 在前一段
            # 还没播完时会 prepend 静音再 queue()，真正的 seg 起点要等
            # start_evt 触发；老的 TTSClient 没有事件，这里 fallback 到
            # 旧的 sleep(duration) 行为。
            play_ret = tts.play_audio_async(item["audio_path"])
            if isinstance(play_ret, tuple) and len(play_ret) == 2:
                start_evt, end_evt = play_ret
            else:
                start_evt, end_evt = None, None

            # 等待本段音频真正出声（gapless 拼接时这里会卡上段剩余时长）
            if start_evt is not None:
                start_evt.wait(timeout=item["duration"] + 10.0)

            with state_lock:
                state.emotion = item["emotion"]
                state.intensity = item["intensity"]
                state.emotion_timeline = [(0.0, item["emotion"], item["intensity"])]
                if first:
                    state.motion_hint = motion_hint
                state.speaking = True
                state.speak_start = time.time()
                state.speak_end = state.speak_start + item["duration"]
                state.phoneme_frames = item["phoneme_frames"]
                if os.environ.get("THA_LIPSYNC_DEBUG"):
                    pfs = item["phoneme_frames"] or []
                    longs = [(getattr(f, "phoneme", "?"),
                              round(getattr(f, "start_time", 0.0), 3),
                              round(getattr(f, "end_time", 0.0), 3),
                              round(getattr(f, "duration", 0.0), 3))
                             for f in pfs if getattr(f, "duration", 0.0) > 0.4]
                    print(f"[lipsync] seg dur={item['duration']:.2f}s "
                          f"phonemes={len(pfs)} longs(>0.4s)={longs} "
                          f"text={item.get('text','')[:24]}")
                state.audio_duration = item["duration"]
                state.audio_path = item["audio_path"]
                state.tts = tts
                tts.set_word_boundaries(item["boundaries"])
                state.interaction_state = "speaking"
                if first:
                    state.state_enter_time = time.time()

            # 等本段播完再解锁下一段（cosyvoice 用 end_evt 精确同步，
            # 其它后端没事件就保留 sleep(duration - 20ms)）
            if end_evt is not None:
                end_evt.wait(timeout=item["duration"] + 10.0)
            else:
                time.sleep(max(0.0, item["duration"] - 0.02))
            # 段间立刻闭嘴：清掉旧的 phoneme_frames，避免下一段还没合成完
            # 时画面停留在上段最后一个元音的张嘴形状。
            with state_lock:
                state.phoneme_frames = []
                state.audio_duration = 0.0
            first = False

        with state_lock:
            state.speaking = False
            # render_loop 会自动把 speaking → post_speak → idle


def main():
    state = RenderState()
    # Create a mutex lock, for safe access to two threads: input_loop and render_loop
    state_lock = threading.Lock()

    # Fixed function calls for thread safety
    stop = threading.Event()
    ready = threading.Event()

    # Switch between mic and text input modes based on environment variable and CLI args
    env_mode = os.getenv(INPUT_MODE_ENV_KEY, DEFAULT_INPUT_MODE).strip().lower()
    input_mode = env_mode if env_mode in ("mic", "text") else DEFAULT_INPUT_MODE
    if "--text" in sys.argv:
        input_mode = "text"
    elif "--mic" in sys.argv:
        input_mode = "mic"
    print(f"[startup] input_mode={input_mode} (CLI 优先, env: {INPUT_MODE_ENV_KEY})")
    
    # Start the input thread to handle user input and LLM communication
    input_thread = threading.Thread(
        target=input_loop,
        args=(state, state_lock, stop, ready, input_mode),
        daemon=True,
    )
    input_thread.start()
    
    # Start the render loop to handle lip sync and animation
    render_loop(state, state_lock, stop, ready)


if __name__ == "__main__":
    main()