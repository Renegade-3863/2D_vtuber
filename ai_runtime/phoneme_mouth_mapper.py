from dataclasses import dataclass
from typing import Dict, List, Tuple
import os

import numpy as np


@dataclass
class PhonemeFrame:
    """一个音素帧"""
    phoneme: str
    start_time: float  # 秒
    end_time: float    # 秒
    duration: float    # 秒


# ========== 英语音素映射 ==========
ENGLISH_PHONEME_TO_MOUTH = {
    # ===== 元音（主要决定嘴型）=====
    
    # AAA 类 (张大嘴)
    'a': {'mouth_aaa': 1.0, 'mouth_eee': 0.1},
    'aa': {'mouth_aaa': 1.0, 'mouth_eee': 0.1},
    'ah': {'mouth_aaa': 0.9, 'mouth_ooo': 0.3},
    'ax': {'mouth_aaa': 0.7, 'mouth_eee': 0.3},  # schwa
    'ae': {'mouth_aaa': 0.8, 'mouth_eee': 0.4},  # cat
    'ao': {'mouth_aaa': 0.7, 'mouth_ooo': 0.5},  # law
    
    # III 类 (咧嘴/扁平)
    'i': {'mouth_iii': 1.0, 'mouth_eee': 0.2},
    'ii': {'mouth_iii': 1.0, 'mouth_eee': 0.2},
    'ih': {'mouth_iii': 0.8, 'mouth_eee': 0.4},  # sit
    'iy': {'mouth_iii': 1.0, 'mouth_eee': 0.2},  # seat
    'e': {'mouth_eee': 0.8, 'mouth_iii': 0.4},
    'eh': {'mouth_eee': 0.8, 'mouth_aaa': 0.4},  # set
    'ey': {'mouth_eee': 0.7, 'mouth_iii': 0.5},  # say
    
    # UUU 类 (圆嘴)
    'u': {'mouth_uuu': 1.0, 'mouth_ooo': 0.2},
    'uu': {'mouth_uuu': 1.0, 'mouth_ooo': 0.2},
    'uh': {'mouth_uuu': 0.8, 'mouth_ooo': 0.4},  # book
    'uw': {'mouth_uuu': 0.9, 'mouth_ooo': 0.3},  # boot
    'oo': {'mouth_ooo': 0.9, 'mouth_uuu': 0.3},  # go
    'oh': {'mouth_ooo': 0.9, 'mouth_aaa': 0.3},  # go
    'o': {'mouth_ooo': 0.8, 'mouth_aaa': 0.4},
    'ow': {'mouth_ooo': 0.7, 'mouth_aaa': 0.5},  # cow
    
    # 双元音
    'ay': {'mouth_aaa': 0.6, 'mouth_iii': 0.6},  # my
    'aw': {'mouth_aaa': 0.7, 'mouth_ooo': 0.5},  # how
    'oy': {'mouth_ooo': 0.6, 'mouth_iii': 0.6},  # boy
    'er': {'mouth_eee': 0.6, 'mouth_ooo': 0.4},  # bird
    'ar': {'mouth_aaa': 0.8, 'mouth_ooo': 0.4},  # car
    
    # ===== 辅音（次要，轻微嘴型）=====
    
    # 鼻音
    'm': {'mouth_aaa': 0.2, 'mouth_ooo': 0.3},
    'n': {'mouth_aaa': 0.2, 'mouth_eee': 0.3},
    'ng': {'mouth_aaa': 0.2, 'mouth_ooo': 0.3},
    
    # 爆破音
    'p': {'mouth_aaa': 0.2},
    'b': {'mouth_aaa': 0.2},
    't': {'mouth_aaa': 0.3, 'mouth_eee': 0.3},
    'd': {'mouth_aaa': 0.3, 'mouth_eee': 0.3},
    'k': {'mouth_aaa': 0.3, 'mouth_ooo': 0.3},
    'g': {'mouth_aaa': 0.3, 'mouth_ooo': 0.3},
    
    # 摩擦音
    'f': {'mouth_eee': 0.5, 'mouth_iii': 0.3},
    # 'v' 是唇齿音 (下唇贴上齿)，视觉上接近收唇 — 用 U/F 系；
    # 同时也避免与中文拼音 'v' (=ü) 在 phoneme_to_mouth_params 里的查表冲突
    'v': {'mouth_uuu': 0.5, 'mouth_eee': 0.3},
    'th': {'mouth_eee': 0.4, 'mouth_aaa': 0.3},
    'dh': {'mouth_eee': 0.4, 'mouth_aaa': 0.3},
    's': {'mouth_eee': 0.6, 'mouth_iii': 0.3},
    'z': {'mouth_eee': 0.6, 'mouth_iii': 0.3},
    'sh': {'mouth_ooo': 0.6, 'mouth_uuu': 0.4},
    'zh': {'mouth_ooo': 0.6, 'mouth_uuu': 0.4},
    'hh': {'mouth_aaa': 0.4},
    
    # 破擦音
    'ch': {'mouth_aaa': 0.5, 'mouth_eee': 0.3},
    'j': {'mouth_aaa': 0.5, 'mouth_eee': 0.3},
    'jh': {'mouth_aaa': 0.5, 'mouth_eee': 0.3},  # CMU 标准 (judge / gym)

    # 流音
    'l': {'mouth_aaa': 0.4, 'mouth_eee': 0.3},
    'r': {'mouth_ooo': 0.5, 'mouth_uuu': 0.3},
    'w': {'mouth_ooo': 0.6, 'mouth_uuu': 0.4},
    'y': {'mouth_iii': 0.6, 'mouth_eee': 0.3},
    
    # 静音
    'sil': {'mouth_aaa': 0.0},
    'sp': {'mouth_aaa': 0.0},
    'pau': {'mouth_aaa': 0.0},
    
    # 常见音素变体（g2p_en 可能生成这些）
    'ss': {'mouth_eee': 0.6, 'mouth_iii': 0.3},  # s 的变体
}


# ========== 中文拼音映射 ==========
CHINESE_PINYIN_TO_MOUTH = {
    # ===== 韵母（主要决定嘴型）=====
    
    # A 类 (张大嘴)
    'a': {'mouth_aaa': 1.0},
    'ai': {'mouth_aaa': 0.9, 'mouth_eee': 0.3},  # ai 从 a 滑向 i
    'ao': {'mouth_aaa': 0.8, 'mouth_ooo': 0.4},  # ao 从 a 滑向 o
    
    # O 类 (圆嘴)
    'o': {'mouth_ooo': 0.9, 'mouth_aaa': 0.3},
    'ou': {'mouth_ooo': 0.8, 'mouth_uuu': 0.4},  # ou 从 o 滑向 u
    
    # E 类 (扁平/微张)
    'e': {'mouth_eee': 0.7, 'mouth_aaa': 0.3},
    'ei': {'mouth_eee': 0.8, 'mouth_iii': 0.3},  # ei 从 e 滑向 i
    'er': {'mouth_eee': 0.6, 'mouth_aaa': 0.3},
    
    # I 类 (咧嘴)
    'i': {'mouth_iii': 1.0, 'mouth_eee': 0.2},
    'ie': {'mouth_iii': 0.8, 'mouth_eee': 0.4},
    'iu': {'mouth_iii': 0.5, 'mouth_ooo': 0.5, 'mouth_uuu': 0.5},  # iu(iou) 从 i 经 o 滑向 u
    
    # U 类 (圆唇)
    'u': {'mouth_uuu': 1.0, 'mouth_ooo': 0.2},
    
    # V/Ü 类 (撮口)
    'v': {'mouth_uuu': 0.9, 'mouth_iii': 0.3},
    've': {'mouth_uuu': 0.8, 'mouth_eee': 0.3},
    'ue': {'mouth_uuu': 0.8, 'mouth_eee': 0.3},  # 月/乐/雪/学 (同 ve)
    
    # 复合韵母 — i 介母
    'ia': {'mouth_aaa': 0.9, 'mouth_iii': 0.3},    # 家/下/夏
    'iao': {'mouth_aaa': 0.7, 'mouth_ooo': 0.4, 'mouth_iii': 0.2},  # 好/小/笑
    'ian': {'mouth_aaa': 0.7, 'mouth_eee': 0.3, 'mouth_iii': 0.2},  # 天/见/面
    'iang': {'mouth_aaa': 0.8, 'mouth_ooo': 0.3, 'mouth_iii': 0.2}, # 想/样/两
    'iong': {'mouth_ooo': 0.8, 'mouth_uuu': 0.3, 'mouth_iii': 0.2}, # 用/穷/兄

    # 复合韵母 — u 介母
    'ua': {'mouth_aaa': 0.9, 'mouth_uuu': 0.3},    # 花/话/画
    'uo': {'mouth_ooo': 0.8, 'mouth_uuu': 0.3},    # 我/过/做
    'ui': {'mouth_uuu': 0.7, 'mouth_eee': 0.4},    # 回/对/水 (uei 的缩写)
    'uai': {'mouth_aaa': 0.7, 'mouth_iii': 0.3, 'mouth_uuu': 0.2}, # 快/外/坏
    'uan': {'mouth_aaa': 0.7, 'mouth_eee': 0.3, 'mouth_uuu': 0.2}, # 关/完/看
    'uang': {'mouth_aaa': 0.8, 'mouth_ooo': 0.3, 'mouth_uuu': 0.2}, # 光/黄/王
    'ueng': {'mouth_eee': 0.5, 'mouth_ooo': 0.3, 'mouth_uuu': 0.3}, # 翁

    # 复合韵母 — ü 介母 (pypinyin 有时输出 v 有时输出 u)
    'van': {'mouth_aaa': 0.6, 'mouth_eee': 0.3, 'mouth_uuu': 0.3}, # 元/远/圆
    'vang': {'mouth_aaa': 0.6, 'mouth_ooo': 0.3, 'mouth_uuu': 0.3}, # (罕见)
    
    # 鼻韵母
    'an': {'mouth_aaa': 0.8, 'mouth_eee': 0.3},  # 以 n 结尾，嘴型稍扁
    'en': {'mouth_eee': 0.6, 'mouth_aaa': 0.3},
    'in': {'mouth_iii': 0.8, 'mouth_eee': 0.3},
    'un': {'mouth_uuu': 0.8, 'mouth_eee': 0.3},
    'vn': {'mouth_uuu': 0.8, 'mouth_iii': 0.3},
    
    'ang': {'mouth_aaa': 0.9, 'mouth_ooo': 0.3},  # 后鼻音
    'eng': {'mouth_eee': 0.6, 'mouth_ooo': 0.3},
    'ing': {'mouth_iii': 0.8, 'mouth_ooo': 0.3},
    'ong': {'mouth_ooo': 0.9, 'mouth_uuu': 0.3},
    
    # ===== 声母（次要，轻微嘴型）=====
    
    # 双唇音
    'b': {'mouth_aaa': 0.2},  # 闭口音
    'p': {'mouth_aaa': 0.2},
    'm': {'mouth_aaa': 0.2, 'mouth_ooo': 0.3},
    
    # 唇齿音
    'f': {'mouth_eee': 0.4, 'mouth_iii': 0.3},
    
    # 舌尖音
    'd': {'mouth_aaa': 0.3, 'mouth_eee': 0.3},
    't': {'mouth_aaa': 0.3, 'mouth_eee': 0.3},
    'n': {'mouth_aaa': 0.2, 'mouth_eee': 0.3},
    'l': {'mouth_aaa': 0.4, 'mouth_eee': 0.3},
    
    # 舌根音
    'g': {'mouth_aaa': 0.3, 'mouth_ooo': 0.3},
    'k': {'mouth_aaa': 0.3, 'mouth_ooo': 0.3},
    'h': {'mouth_aaa': 0.3},
    
    # 舌面音
    'j': {'mouth_iii': 0.5, 'mouth_eee': 0.3},
    'q': {'mouth_iii': 0.5, 'mouth_eee': 0.3},
    'x': {'mouth_iii': 0.4, 'mouth_eee': 0.4},
    
    # 舌尖后音（翘舌）
    'zh': {'mouth_aaa': 0.4, 'mouth_eee': 0.3},
    'ch': {'mouth_aaa': 0.4, 'mouth_eee': 0.3},
    'sh': {'mouth_eee': 0.5, 'mouth_ooo': 0.3},
    'r': {'mouth_ooo': 0.5, 'mouth_uuu': 0.3},
    
    # 舌尖前音（平舌）
    'z': {'mouth_aaa': 0.3, 'mouth_eee': 0.4},
    'c': {'mouth_aaa': 0.3, 'mouth_eee': 0.4},
    's': {'mouth_eee': 0.5, 'mouth_iii': 0.3},
    
    # 零声母/其他
    'y': {'mouth_iii': 0.5, 'mouth_eee': 0.3},
    'w': {'mouth_ooo': 0.6, 'mouth_uuu': 0.4},
    
    # 标点符号/停顿（闭嘴）
    '，': {'mouth_aaa': 0.0},
    '。': {'mouth_aaa': 0.0},
    '？': {'mouth_aaa': 0.0},
    '！': {'mouth_aaa': 0.0},
    '；': {'mouth_aaa': 0.0},
    '、': {'mouth_aaa': 0.0},
    '：': {'mouth_aaa': 0.0},
    ' ': {'mouth_aaa': 0.0},  # 空格
    'sil': {'mouth_aaa': 0.0},  # 静音
    'sil_short': {'mouth_aaa': 0.0},   # 短停顿 0.3s
    'sil_medium': {'mouth_aaa': 0.0},  # 中停顿 0.45s
    'sil_long': {'mouth_aaa': 0.0},    # 长停顿 0.6s
}


# 所有可能的嘴型参数名
MOUTH_PARAM_NAMES = [
    'mouth_aaa',
    'mouth_iii',
    'mouth_uuu',
    'mouth_eee',
    'mouth_ooo',
    'mouth_delta',
]


class PhonemeMouthMapper:
    """将音素序列转换为 THA3 嘴型参数序列"""
    
    def __init__(self, pose_mapper):
        """
        Args:
            pose_mapper: THAPoseMapper 实例，用于获取参数索引
        """
        self.pose_mapper = pose_mapper
        self.name_to_idx = pose_mapper.name_to_index
    
    def phoneme_to_mouth_params(self, phoneme: str) -> Dict[str, float]:
        """
        将单个音素转换为嘴型参数字典
        """
        phoneme = phoneme.lower().strip()
        
        # 先尝试英语映射
        if phoneme in ENGLISH_PHONEME_TO_MOUTH:
            return ENGLISH_PHONEME_TO_MOUTH[phoneme].copy()
        
        # 再尝试中文拼音映射
        if phoneme in CHINESE_PINYIN_TO_MOUTH:
            return CHINESE_PINYIN_TO_MOUTH[phoneme].copy()
        
        # 尝试部分匹配
        base_phoneme = phoneme.rstrip('0123456789')
        if base_phoneme in ENGLISH_PHONEME_TO_MOUTH:
            return ENGLISH_PHONEME_TO_MOUTH[base_phoneme].copy()
        if base_phoneme in CHINESE_PINYIN_TO_MOUTH:
            return CHINESE_PINYIN_TO_MOUTH[base_phoneme].copy()
        
        # 默认返回（静音/闭嘴）
        return {'mouth_aaa': 0.0, 'mouth_iii': 0.0, 'mouth_uuu': 0.0, 'mouth_eee': 0.0, 'mouth_ooo': 0.0}
    
    def build_mouth_pose(self, mouth_params: Dict[str, float]) -> List[float]:
        """
        将嘴型参数转换为完整的 pose 向量
        
        Args:
            mouth_params: 嘴型参数字典
            
        Returns:
            完整的 pose 向量列表
        """
        pose = [0.0] * self.pose_mapper.num_params
        
        for param_name, value in mouth_params.items():
            if param_name in self.name_to_idx:
                idx = self.name_to_idx[param_name]
                lo, hi = self.pose_mapper.name_to_range.get(param_name, (0.0, 1.0))
                pose[idx] = max(lo, min(hi, value))
        
        return pose
    
    def get_mouth_pose_for_time(self, phoneme_frames: List[PhonemeFrame], 
                                  current_time: float) -> List[float]:
        """
        根据当前时间获取对应的嘴型 pose
        
        Args:
            phoneme_frames: 音素帧列表
            current_time: 当前播放时间 (秒)
            
        Returns:
            嘴型 pose 向量
        """
        current_phoneme = None
        for frame in phoneme_frames:
            if frame.start_time <= current_time < frame.end_time:
                current_phoneme = frame.phoneme
                break
        
        if current_phoneme is None:
            if phoneme_frames:
                current_phoneme = phoneme_frames[-1].phoneme
            else:
                current_phoneme = 'sil'
        
        mouth_params = self.phoneme_to_mouth_params(current_phoneme)
        return self.build_mouth_pose(mouth_params)
    
    def interpolate_mouth_pose(self, phoneme_frames: List[PhonemeFrame],
                                current_time: float,
                                lookahead: float = 0.02) -> List[float]:
        """
        获取带插值的嘴型 pose (更平滑)

        Args:
            phoneme_frames: 音素帧列表
            current_time: 当前播放时间 (秒)
            lookahead: 向前看的时间窗口 (秒)，默认 20ms

        Returns:
            插值后的嘴型 pose 向量
        """
        if not phoneme_frames:
            return [0.0] * self.pose_mapper.num_params

        current_frame = None
        next_frame = None
        current_idx = -1

        for i, frame in enumerate(phoneme_frames):
            if frame.start_time <= current_time < frame.end_time:
                current_frame = frame
                current_idx = i
                if i + 1 < len(phoneme_frames):
                    next_frame = phoneme_frames[i + 1]
                break

        if current_frame is None:
            if current_time < phoneme_frames[0].start_time:
                # 段头 pre-roll（首音素 start_time > 0，或 lookahead 把
                # 渲染时刻预跳到了首音素之前）。这段时间音频还在 silence，
                # 不要预取首音素张嘴口型 —— 否则会出现"莫名张嘴"
                # （音频没声但嘴已经张开 ~80ms）。
                if os.environ.get("THA_LIPSYNC_DEBUG"):
                    print(f"[lipsync] current_time={current_time:.3f}s before "
                          f"first phoneme start={phoneme_frames[0].start_time:.3f}s. "
                          f"Closing mouth (pre-roll silence).")
                return [0.0] * self.pose_mapper.num_params
            else:
                # current_time 超出了所有 phoneme 帧覆盖范围。两种情况：
                #   1) current_time >= 最后一帧 end_time：段已结束。
                #   2) current_time 落在两帧之间的"空洞"。理论上 align 阶段
                #      已把所有间隙补成 sil 或拉伸上一帧，正常不会出现；
                #      若仍出现，沿用上一帧的口型避免突然闭嘴抖动。
                last = phoneme_frames[-1]
                if current_time >= last.end_time:
                    if os.environ.get("THA_LIPSYNC_DEBUG"):
                        print(f"[lipsync] current_time={current_time:.2f}s past "
                              f"last phoneme end_time={last.end_time:.2f}s "
                              f"({len(phoneme_frames)} frames). Closing mouth.")
                    return [0.0] * self.pose_mapper.num_params
                # 空洞：找最近的前一帧延用
                prev = None
                for f in phoneme_frames:
                    if f.end_time <= current_time:
                        prev = f
                    else:
                        break
                if prev is None:
                    return [0.0] * self.pose_mapper.num_params
                if os.environ.get("THA_LIPSYNC_DEBUG"):
                    print(f"[lipsync] current_time={current_time:.2f}s in inter-frame "
                          f"gap, holding ph={prev.phoneme} (end={prev.end_time:.2f}s)")
                return self.build_mouth_pose(
                    self.phoneme_to_mouth_params(prev.phoneme)
                )

        duration = current_frame.duration
        if duration > 0:
            progress = (current_time - current_frame.start_time) / duration
        else:
            progress = 0.5

        current_params = self.phoneme_to_mouth_params(current_frame.phoneme)

        # ===== sil release tail (自适应) =====
        # 当前帧是 sil 且前一帧是发音（非 sil/非闭口辅音），在 sil 起始
        # 的一段时间里平滑从前一帧的口型衰减到 0，避免元音"瞬间闭嘴"
        # 看起来口型没做到位。
        # 自适应：
        #   - 短 sil（句中逗号/字间，<0.4s）→ 100ms 释放，连读流畅
        #   - 长 sil（句末，>=0.4s）→ 50ms 释放，干净闭嘴不拖泥带水
        if current_frame.phoneme == 'sil' and current_idx > 0:
            prev_frame = phoneme_frames[current_idx - 1]
            if prev_frame.phoneme != 'sil':
                _RELEASE = 0.10 if current_frame.duration < 0.4 else 0.05
                elapsed = current_time - current_frame.start_time
                if elapsed < _RELEASE:
                    k = 1.0 - elapsed / _RELEASE
                    k = k * k * (3 - 2 * k)  # smoothstep
                    prev_params = self.phoneme_to_mouth_params(prev_frame.phoneme)
                    return self.build_mouth_pose(
                        {key: v * k for key, v in prev_params.items()}
                    )

        # 自适应混合窗口：短音素用更大的混合区间，避免生硬跳变
        # 长音素（>120ms）：末尾 15% 开始混合
        # 短音素（<60ms）：末尾 40% 开始混合（约 20-25ms 过渡）
        if duration > 0.12:
            blend_start = 0.85
        elif duration > 0.06:
            blend_start = 0.70
        else:
            blend_start = 0.60

        # 若下一帧是 sil，不在当前元音窗里向 0 衰减 —— 让元音保持峰值
        # 到 end_time，由 sil 的 release tail 接管闭口动作。否则短促的句末
        # 元音（如"坡/啊/吗"）峰值时间被切掉看起来口型没到位。
        if next_frame and next_frame.phoneme != 'sil' and progress > blend_start:
            next_params = self.phoneme_to_mouth_params(next_frame.phoneme)

            blend = (progress - blend_start) / (1.0 - blend_start)
            blend = max(0.0, min(1.0, blend))
            # 用 smoothstep 曲线让过渡更自然
            blend = blend * blend * (3 - 2 * blend)

            blended_params = {}
            all_keys = set(current_params.keys()) | set(next_params.keys())
            for key in all_keys:
                curr_val = current_params.get(key, 0.0)
                next_val = next_params.get(key, 0.0)
                blended_params[key] = curr_val * (1 - blend) + next_val * blend

            current_params = blended_params

        return self.build_mouth_pose(current_params)