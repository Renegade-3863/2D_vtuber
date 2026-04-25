import json
import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from ai_runtime.phoneme_mouth_mapper import PhonemeFrame


# 标点符号 → 静音标记映射
# TTS 在标点处会产生自然停顿，对应闭嘴状态
PUNCTUATION_TO_SILENCE = {
    # 短停顿
    ',': 'sil_short', '，': 'sil_short', '、': 'sil_short',
    # 中停顿
    ';': 'sil_medium', '；': 'sil_medium',
    ':': 'sil_medium', '：': 'sil_medium',
    # 长停顿（仅句中使用，句尾会被移除）
    '.': 'sil_long', '。': 'sil_long',
    '?': 'sil_long', '？': 'sil_long',
    '!': 'sil_long', '！': 'sil_long',
    '…': 'sil_long', '⋯': 'sil_long',
    '—': 'sil_medium', '–': 'sil_medium',
}

# 静音标记的时长权重（相对于正常音素）
# TTS 在逗号处约停顿 200-350ms，句号约 300-500ms
SILENCE_WEIGHTS = {
    'sil_short': 2.5,   # 逗号等短停顿
    'sil_medium': 3.5,  # 分号、冒号等中停顿
    'sil_long': 4.5,    # 句号、问号等长停顿（句中）
}


@dataclass
class PhonemeAlignment:
    """音素对齐结果"""
    text: str
    phoneme_frames: List[PhonemeFrame]
    audio_path: Optional[str] = None


class AudioPhonemeAligner:
    """
    支持中文和英文的音素对齐器
    
    依赖安装:
        pip install pypinyin
    """
    
    def __init__(self, language: str = "auto"):
        """
        Args:
            language: 'auto' | 'zh' | 'en'
                - 'auto': 自动检测（默认）
                - 'zh': 中文
                - 'en': 英文
        """
        self.language = language
        self._pypinyin = None
        self._init_pypinyin()
        self._g2p = None
        self._init_g2p()

    def _init_pypinyin(self):
        """初始化中文拼音"""
        try:
            from pypinyin import pinyin, Style
            self._pypinyin = pinyin
            self._pinyin_style = Style.NORMAL
        except ImportError:
            print("[警告] pypinyin 未安装，音素转换将不可用")
            print("安装：pip install pypinyin")

    def _init_g2p(self):
        """初始化英文 g2p（CMU dict + neural fallback）。
        没装时回退到字典 + 字母规则切分（不准但能跑）。"""
        try:
            from g2p_en import G2p
            self._g2p = G2p()
        except ImportError:
            print("[警告] g2p_en 未安装，英文将走字母规则回退（嘴型不准）")
            print("安装：pip install g2p-en")
        except Exception as e:
            # 首次会触发 nltk 下载 cmudict / averaged_perceptron_tagger，
            # 网络不通时构造失败，降级到字典回退。
            print(f"[警告] g2p_en 初始化失败 ({e})，英文走回退")
    
    def _detect_language(self, text: str) -> str:
        """
        自动检测文本语言
        
        简单规则：包含中文字符则为中文，否则为英文
        """
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return 'zh'
        return 'en'
    
    def text_to_phonemes(self, text: str, language: str = None) -> List[str]:
        """
        将文本转换为音素列表
        
        Args:
            text: 输入文本
            language: 'auto' | 'zh' | 'en'
            
        Returns:
            音素列表
        """
        if language is None:
            language = self.language
        
        if language == 'auto':
            language = self._detect_language(text)
        
        if language == 'zh':
            return self._chinese_to_phonemes(text)
        else:
            # 英文：使用简单规则（按字母/音节划分）
            return self._english_to_phonemes_simple(text)
    
    def _build_english_dict(self) -> dict:
        """常见单词的音素字典（CMU dict 简化版）"""
        return {
            'hello': ['hh', 'ax', 'l', 'ow'],
            'hi': ['hh', 'ay'],
            'yes': ['y', 'eh', 's'],
            'no': ['n', 'ow'],
            'okay': ['ow', 'k', 'ey'],
            'thanks': ['th', 'ae', 'ng', 'k', 's'],
            'thank': ['th', 'ae', 'ng', 'k'],
            'you': ['y', 'uw'],
            'your': ['y', 'ao', 'r'],
            'name': ['n', 'ey', 'm'],
            'my': ['m', 'ay'],
            'is': ['ih', 'z'],
            'it': ['ih', 't'],
            'the': ['dh', 'ax'],
            'a': ['ax'],
            'an': ['ae', 'n'],
            'and': ['ae', 'n', 'd'],
            'in': ['ih', 'n'],
            'on': ['aa', 'n'],
            'at': ['ae', 't'],
            'to': ['t', 'uw'],
            'for': ['f', 'ao', 'r'],
            'of': ['ah', 'v'],
            'with': ['w', 'ih', 'dh'],
            'have': ['hh', 'ae', 'v'],
            'has': ['hh', 'ae', 'z'],
            'had': ['hh', 'ae', 'd'],
            'will': ['w', 'ih', 'l'],
            'would': ['w', 'uh', 'd'],
            'could': ['k', 'uh', 'd'],
            'should': ['sh', 'uh', 'd'],
            'can': ['k', 'ae', 'n'],
            'do': ['d', 'uw'],
            'does': ['d', 'ah', 'z'],
            'did': ['d', 'ih', 'd'],
            'am': ['ae', 'm'],
            'are': ['aa', 'r'],
            'was': ['w', 'aa', 'z'],
            'were': ['w', 'er'],
            'be': ['b', 'iy'],
            'been': ['b', 'iy', 'n'],
            'being': ['b', 'iy', 'ih', 'ng'],
            'i': ['ay'],
            'we': ['w', 'iy'],
            'they': ['dh', 'ey'],
            'he': ['hh', 'iy'],
            'she': ['sh', 'iy'],
            'what': ['w', 'aa', 't'],
            'where': ['w', 'eh', 'r'],
            'when': ['w', 'eh', 'n'],
            'why': ['w', 'ay'],
            'how': ['hh', 'aw'],
            'who': ['hh', 'uw'],
            'which': ['w', 'ih', 'ch'],
            'that': ['dh', 'ae', 't'],
            'this': ['dh', 'ih', 's'],
            'these': ['dh', 'iy', 'z'],
            'those': ['dh', 'ow', 'z'],
            'here': ['hh', 'iy', 'r'],
            'there': ['dh', 'eh', 'r'],
            'know': ['n', 'ow'],
            'like': ['l', 'ay', 'k'],
            'want': ['w', 'aa', 'n', 't'],
            'need': ['n', 'iy', 'd'],
            'think': ['th', 'ih', 'ng', 'k'],
            'say': ['s', 'ey'],
            'said': ['s', 'eh', 'd'],
            'see': ['s', 'iy'],
            'look': ['l', 'uh', 'k'],
            'good': ['g', 'uh', 'd'],
            'bad': ['b', 'ae', 'd'],
            'happy': ['hh', 'ae', 'p', 'iy'],
            'sad': ['s', 'ae', 'd'],
            'please': ['p', 'l', 'iy', 'z'],
            'sorry': ['s', 'aa', 'r', 'iy'],
            'excuse': ['ih', 'k', 's', 'k', 'y', 'uw', 'z'],
            'help': ['hh', 'eh', 'l', 'p'],
            'question': ['k', 'w', 'eh', 'sh', 'ax', 'n'],
            'answer': ['ae', 'n', 's', 'er'],
            'one': ['w', 'ah', 'n'],
            'two': ['t', 'uw'],
            'three': ['th', 'r', 'iy'],
            'plus': ['p', 'l', 'ah', 's'],
            'equals': ['iy', 'k', 'w', 'ax', 'l', 'z'],
            'equal': ['iy', 'k', 'w', 'ax', 'l'],
        }

    def _english_to_phonemes_simple(self, text: str) -> List[str]:
        """
        英语转音素：g2p_en 优先，未装时回退到内置字典 + 规则切分。
        """
        result = []
        text = text.lower()

        # 简单分词
        words = text.split()
        for word in words:
            # 提取尾部标点符号
            trailing_punct = None
            raw_word = word
            while raw_word and raw_word[-1] in PUNCTUATION_TO_SILENCE:
                trailing_punct = raw_word[-1]
                raw_word = raw_word[:-1]

            clean_word = ''.join(c for c in raw_word if c.isalpha())
            if clean_word:
                # 走统一的单词→音素入口（内部已优先 g2p_en）
                result.extend(self._english_word_to_phonemes(clean_word))

            # 在标点处插入静音帧（避免连续静音）
            if trailing_punct:
                sil_type = PUNCTUATION_TO_SILENCE[trailing_punct]
                if not result or not result[-1].startswith('sil'):
                    result.append(sil_type)

        # 移除末尾静音（句尾标点不需要停顿，语音直接结束）
        while result and result[-1].startswith('sil'):
            result.pop()

        return result
    
    def _split_unknown_word(self, word: str) -> List[str]:
        """
        将未知单词按音节规则划分
        
        返回近似的音素序列
        """
        # 元音字母（决定嘴型）
        vowels = ['a', 'e', 'i', 'o', 'u', 'y']
        
        # 双字母音素
        digraphs = ['sh', 'ch', 'th', 'ph', 'wh', 'ng', 'ck', 'qu', 'ss', 'tt', 'll', 'ff']
        
        result = []
        i = 0
        while i < len(word):
            # 检查双字母组合
            if i + 1 < len(word):
                two_char = word[i:i+2]
                if two_char in digraphs:
                    result.append(two_char)
                    i += 2
                    continue
            
            char = word[i]
            if char in vowels:
                # 元音：可能组合成双元音
                phoneme = char
                j = i + 1
                # 合并连续元音
                while j < len(word) and word[j] in vowels:
                    phoneme += word[j]
                    j += 1
                # 简化双元音
                if len(phoneme) > 2:
                    phoneme = phoneme[:2]
                result.append(phoneme)
                i = j
            else:
                # 辅音
                result.append(char)
                i += 1
        
        return result
    
    def _chinese_to_phonemes(self, text: str) -> List[str]:
        """
        中文转拼音音素
        
        将汉字转换为拼音，然后进一步分解为声母 + 韵母。
        标点符号转为静音标记 (sil_short/sil_medium/sil_long)。
        """
        if self._pypinyin is None:
            # 降级方案：过滤标点后返回字符
            return [c for c in text if c.isalpha() or '\u4e00' <= c <= '\u9fff']
        
        # 获取拼音（不带声调）
        pinyin_list = self._pypinyin(
            text, 
            style=self._pinyin_style,
            heteronym=False  # 不处理多音字，取默认读音
        )
        
        # 扁平化：[['ni'], ['hao']] -> ['ni', 'hao']
        phonemes = [p[0] for p in pinyin_list if p]
        
        # 进一步分解拼音为声母 + 韵母
        # 标点符号转换为静音标记
        result = []
        for pinyin in phonemes:
            # 检查是否是标点符号
            if pinyin in PUNCTUATION_TO_SILENCE:
                # 避免连续静音
                if result and not result[-1].startswith('sil'):
                    result.append(PUNCTUATION_TO_SILENCE[pinyin])
                continue
            
            # 跳过非拼音字符（数字、特殊符号等）
            if not pinyin.isalpha():
                continue
            
            initials, finals = self._split_pinyin(pinyin)
            if initials:
                result.append(initials)
            if finals:
                result.append(finals)
        
        # 移除末尾静音（句尾标点不需要停顿）
        while result and result[-1].startswith('sil'):
            result.pop()
        
        return result
    
    def _split_pinyin(self, pinyin: str) -> tuple:
        """
        将拼音分解为声母和韵母
        
        Args:
            pinyin: 拼音字符串，如 'ni', 'hao', 'zhong'
            
        Returns:
            (声母，韵母) 元组
        """
        # 零声母规范化：拼音 yu/yuan/yun/wo/wei... 在 pypinyin 输出里
        # 用 y/w 开头，但音节本质是 ü 介韵或 u 介韵。直接按"声母+韵母"
        # 拆 (y, u) 会让"鱼/元/运"等字嘴型走到 U 类而不是 ü 类。
        # 这里把零声母字直接映射到标准韵母，再返回 ('', 韵母)。
        ZERO_INITIAL_CANONICAL = {
            # ü 类（yu/yue/yuan/yun → v/ve/van/vn）
            'yu': 'v', 'yue': 've', 'yuan': 'van', 'yun': 'vn',
            # i 介韵（yi/ya/ye/yao/you/yan/yin/yang/ying/yong）
            'yi': 'i', 'ya': 'ia', 'ye': 'ie', 'yao': 'iao',
            'you': 'iu', 'yan': 'ian', 'yin': 'in',
            'yang': 'iang', 'ying': 'ing', 'yong': 'iong',
            # u 介韵（wu/wa/wo/wai/wei/wan/wen/wang/weng）
            'wu': 'u', 'wa': 'ua', 'wo': 'uo', 'wai': 'uai',
            'wei': 'ui', 'wan': 'uan', 'wen': 'un',
            'wang': 'uang', 'weng': 'ueng',
        }
        if pinyin in ZERO_INITIAL_CANONICAL:
            return '', ZERO_INITIAL_CANONICAL[pinyin]

        # 声母表
        initials = [
            'zh', 'ch', 'sh',  # 双字母声母
            'z', 'c', 's', 'b', 'p', 'm', 'f', 
            'd', 't', 'n', 'l', 'g', 'k', 'h', 
            'j', 'q', 'x', 'r', 'y', 'w'
        ]
        
        # 查找声母
        initial = ''
        for ini in initials:
            if pinyin.startswith(ini):
                initial = ini
                break
        
        # 剩余部分是韵母
        final = pinyin[len(initial):] if initial else pinyin
        
        return initial, final
    
    def align_text_to_audio(self, text: str, audio_path: str, 
                            estimated_duration: float,
                            word_boundaries=None) -> PhonemeAlignment:
        """
        将文本与音频对齐，生成带时间戳的音素序列
        
        如果提供了 word_boundaries（来自 TTS 引擎的精确词边界），
        则使用精确对齐；否则退回到加权估算。
        
        Args:
            text: 原始文本
            audio_path: 音频文件路径
            estimated_duration: 音频时长（秒）
            word_boundaries: TTSClient.word_boundaries 列表（可选）
        """
        language = self._detect_language(text) if self.language == 'auto' else self.language

        # 优先使用精确词边界对齐
        if word_boundaries:
            # 覆盖率检查：CosyVoice 偶尔会漏掉句首若干字 (如 "从前有只猫，"
            # 整段缺失, 第一个 timestamp 落在 1s 之后)。如果头部缺口太大,
            # 用部分 timestamp 强行对齐会出现长时间闭嘴洞, 不如退回到全文估算。
            # 注意: 句尾的 tail_gap 通常是 0.3~0.5s (标点/收尾静音不算发音),
            # 这是正常的尾部静默期, 不算漏字, 不应触发降级。
            HEAD_TOLERANCE = 0.30  # 秒 — 句首必须严格
            TAIL_TOLERANCE = 1.50  # 秒 — 句尾宽松, 让 _align_with_word_boundaries
                                   #       自己补尾部静音帧
            first_off = word_boundaries[0].offset
            last_end = word_boundaries[-1].end
            head_gap = first_off
            tail_gap = max(0.0, estimated_duration - last_end)
            if head_gap > HEAD_TOLERANCE or tail_gap > TAIL_TOLERANCE:
                print(f"[align] word_boundaries 覆盖不全 "
                      f"(head_gap={head_gap:.2f}s, tail_gap={tail_gap:.2f}s, "
                      f"total={estimated_duration:.2f}s) → 降级估算")
            else:
                frames = self._align_with_word_boundaries(
                    word_boundaries, language, estimated_duration
                )
                if frames:
                    return PhonemeAlignment(
                        text=text,
                        phoneme_frames=frames,
                        audio_path=audio_path
                    )

        # 退回到加权估算
        return self._align_weighted(text, audio_path, estimated_duration, language)

    def _smooth_word_boundaries(self, word_boundaries):
        """吸收 CosyVoice timestamp 字间假静音, 把 gap 50/50 分给两侧字。

        CosyVoice timestamp 的 begin/end 是「字音 onset 中心」, 不是字音的可听
        完整范围。后果: 短字 (40~80ms) + 长 sil (300~500ms) 交替, 渲染出来
        嘴一闪 + 假闭嘴。两个真正发音的字之间不会有真静音, 只有标点才会。
        所以: 所有「非标点-非标点」之间的 gap (>60ms) 都 50/50 吸收。
        """
        from ai_runtime.tts_client import WordBoundary
        if not word_boundaries or len(word_boundaries) < 2:
            return word_boundaries

        MIN_KEEP_GAP = 0.06  # 60ms 以下是自然连读, 不动
        MAX_ABSORB = 1.50    # gap > 1.5s 兜底当真停顿

        def is_punct(s: str) -> bool:
            s = (s or "").strip()
            if not s:
                return True
            if len(s) > 1:
                return False
            return not ('\u4e00' <= s <= '\u9fff' or s.isalpha() or s.isdigit())

        smoothed = [WordBoundary(text=wb.text, offset=wb.offset,
                                 duration=wb.duration, end=wb.end)
                    for wb in word_boundaries]

        # 开头字特殊处理: CosyVoice 经常把首字时长压到 40~80ms。
        # 如果首字非标点且 duration < 100ms, 让它往前吃到 t=0。
        first = smoothed[0]
        if not is_punct(first.text) and first.duration < 0.10 and first.offset > 0.0:
            first.offset = 0.0
            first.duration = first.end - first.offset

        for i in range(len(smoothed) - 1):
            cur = smoothed[i]
            nxt = smoothed[i + 1]
            gap = nxt.offset - cur.end
            if gap <= MIN_KEEP_GAP or gap >= MAX_ABSORB:
                continue
            if is_punct(cur.text) or is_punct(nxt.text):
                continue
            half = gap / 2.0
            cur.end = cur.end + half
            cur.duration = cur.end - cur.offset
            nxt.offset = nxt.offset - half
            nxt.duration = nxt.end - nxt.offset
        return smoothed

    def _align_with_word_boundaries(
        self, word_boundaries, language: str, total_duration: float
    ) -> List[PhonemeFrame]:
        """
        使用 TTS 引擎提供的精确词边界进行音素对齐
        
        策略：
        1. 每个词有精确的 [offset, end] 时间
        2. 词内按音素权重分配时间
        3. 词与词之间的间隙自动变为静音帧（闭嘴）
        """
        # CosyVoice quirk: timestamp 标的是「字音 onset 中心」, 不是字音的可听完整范围。
        # 常见症状: '今' @0.64-0.68s (40ms) 后跟 0.44s "静音", 然后 '天' @1.12s。
        # 实际上"今"字的 in 韵母会持续到 1.0s 左右, 所谓"静音"其实是字音延续。
        # 直接用原始 boundary 会让短字嘴型一闪而过 + 中间假闭嘴。
        # 修法: 字间 gap < ABSORB_GAP 时, 把 gap 50/50 分给前后两字 (吃掉假静音)。
        word_boundaries = self._smooth_word_boundaries(word_boundaries)

        frames: List[PhonemeFrame] = []
        MIN_SILENCE_GAP = 0.06  # 小于 60ms 的间隙不算停顿

        for i, wb in enumerate(word_boundaries):
            word_start = wb.offset
            word_end = wb.end
            word_text = wb.text

            # 如果与上一个词之间有间隙：
            #  - gap > MIN_SILENCE_GAP: 插入显式静音帧（闭嘴）
            #  - gap <= MIN_SILENCE_GAP: 拉伸上一帧填满，避免出现"无帧空洞"
            #    (CosyVoice word_timestamp 切得很细，频繁有 10~50ms 微间隙，
            #     不补会让 interpolate_mouth_pose 找不到帧 → 闪烁闭嘴)
            if frames:
                prev_end = frames[-1].end_time
                gap = word_start - prev_end
                if gap > MIN_SILENCE_GAP:
                    frames.append(PhonemeFrame(
                        phoneme='sil',
                        start_time=prev_end,
                        end_time=word_start,
                        duration=gap,
                    ))
                elif gap > 0:
                    last = frames[-1]
                    frames[-1] = PhonemeFrame(
                        phoneme=last.phoneme,
                        start_time=last.start_time,
                        end_time=word_start,
                        duration=word_start - last.start_time,
                    )

            # 将词分解为音素
            # 关键: 不能盲目用 segment 级 language。中英混排时一个 zh 段里
            # 可能夹着 "WiFi" / "PPT" / "AI" 这种英文 token, 必须按 word_text
            # 自身重新判断 (含中文字符 → zh, 否则 → en)。否则 zh 分支会把
            # 英文当 pinyin 处理, 直接出空音素。
            has_zh = any('\u4e00' <= c <= '\u9fff' for c in word_text)
            word_lang = 'zh' if has_zh else 'en'
            phonemes = self._word_to_phonemes(word_text, word_lang)
            if os.environ.get("THA_LIPSYNC_DEBUG"):
                print(f"  [align] '{word_text}' [{word_lang}] "
                      f"@{word_start:.2f}-{word_end:.2f}s ({wb.duration:.2f}s) "
                      f"→ {phonemes}")
            if not phonemes:
                # 区分两种 "音素为空" 的情况:
                #  1) 标点 (～？，。…) → 应该闭嘴 (sil)
                #  2) 真 OOV 中文字 / 拼音库失败 → 用 'a' 兜底, 至少嘴动一下
                stripped = word_text.strip()
                is_punct = (not stripped) or (
                    len(stripped) == 1 and not (
                        '\u4e00' <= stripped <= '\u9fff'
                        or stripped.isalpha() or stripped.isdigit()
                    )
                )
                fallback_ph = 'sil' if is_punct else 'a'
                frames.append(PhonemeFrame(
                    phoneme=fallback_ph,
                    start_time=word_start,
                    end_time=word_end,
                    duration=wb.duration,
                ))
                continue

            # 词内按权重分配时间
            weights = [self._get_phoneme_weight(p, language) for p in phonemes]
            total_weight = sum(weights)
            if total_weight <= 0:
                total_weight = 1.0

            word_duration = wb.duration
            current_time = word_start

            for j, (phoneme, weight) in enumerate(zip(phonemes, weights)):
                dur = word_duration * (weight / total_weight)
                end_t = current_time + dur
                # 最后一个音素延伸到词结束
                if j == len(phonemes) - 1:
                    end_t = word_end
                    dur = end_t - current_time
                frames.append(PhonemeFrame(
                    phoneme=phoneme,
                    start_time=current_time,
                    end_time=end_t,
                    duration=dur,
                ))
                current_time = end_t

        # 如果最后一帧在 total_duration 之前结束，补一个尾部静音
        if frames and frames[-1].end_time < total_duration - MIN_SILENCE_GAP:
            frames.append(PhonemeFrame(
                phoneme='sil',
                start_time=frames[-1].end_time,
                end_time=total_duration,
                duration=total_duration - frames[-1].end_time,
            ))

        return frames

    def _word_to_phonemes(self, word: str, language: str) -> List[str]:
        """将单个词转换为音素列表（不含标点和静音标记）"""
        if language == 'zh':
            return self._chinese_word_to_phonemes(word)
        else:
            return self._english_word_to_phonemes(word)

    def _chinese_word_to_phonemes(self, word: str) -> List[str]:
        """
        中文词转音素
        
        "你好" -> ['n', 'i', 'h', 'ao']
        "AI"   -> ['a', 'i']  (英文回退)
        """
        result = []
        # 过滤掉标点
        word = ''.join(c for c in word if '\u4e00' <= c <= '\u9fff' or c.isalpha())
        if not word:
            return result

        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in word)

        if has_chinese and self._pypinyin:
            pinyin_list = self._pypinyin(
                word, style=self._pinyin_style, heteronym=False
            )
            for p in pinyin_list:
                py = p[0] if p else ''
                if not py or not py.isalpha():
                    continue
                initials, finals = self._split_pinyin(py)
                if initials:
                    result.append(initials)
                if finals:
                    result.append(finals)
        else:
            # 英文回退（如 "AI"、"OK" 等）
            result = self._english_word_to_phonemes(word)

        return result

    def _english_word_to_phonemes(self, word: str) -> List[str]:
        """英文单词转音素：优先 g2p_en (CMU dict + neural)，回退到内置字典。"""
        clean = ''.join(c for c in word.lower() if c.isalpha())
        if not clean:
            return []
        # 优先走 g2p_en：返回 ARPABET 大写带重音 (e.g. 'AH0', 'EH1')。
        # 我们的映射表用小写无重音键，所以做一次规范化。
        if self._g2p is not None:
            try:
                raw = self._g2p(clean)
                phs: List[str] = []
                for tok in raw:
                    if not tok or not tok.strip():
                        continue
                    t = tok.strip()
                    # 标点 / 空格 g2p 会原样返回，跳过
                    if not t[0].isalpha():
                        continue
                    # 去掉重音标记 0/1/2 + 转小写
                    t = t.lower().rstrip('0123456789')
                    phs.append(t)
                if phs:
                    return phs
            except Exception as e:
                print(f"[g2p] failed for '{clean}': {e}, fallback to dict")
        # 回退：内置字典 → 字母规则
        from_dict = self._get_english_dict().get(clean)
        if from_dict:
            return list(from_dict)
        return self._split_unknown_word(clean)

    def _get_english_dict(self) -> dict:
        """返回英文常用词音素字典（懒加载缓存）"""
        if not hasattr(self, '_en_dict_cache'):
            self._en_dict_cache = self._build_english_dict()
        return self._en_dict_cache

    def _align_weighted(self, text, audio_path, estimated_duration, language):
        """退回方案：按权重估算对齐（无词边界时使用）"""
        phonemes = self.text_to_phonemes(text, language)

        if not phonemes:
            return PhonemeAlignment(
                text=text,
                phoneme_frames=[],
                audio_path=audio_path
            )

        total_duration = estimated_duration

        weights = []
        for p in phonemes:
            if p in SILENCE_WEIGHTS:
                weights.append(SILENCE_WEIGHTS[p])
            elif p.startswith('sil'):
                weights.append(3.0)
            else:
                weights.append(self._get_phoneme_weight(p, language))

        total_weight = sum(weights)
        if total_weight <= 0:
            total_weight = 1.0

        phoneme_frames = []
        current_time = 0.0

        for phoneme, weight in zip(phonemes, weights):
            duration = total_duration * (weight / total_weight)
            frame = PhonemeFrame(
                phoneme=phoneme,
                start_time=current_time,
                end_time=current_time + duration,
                duration=duration
            )
            phoneme_frames.append(frame)
            current_time += duration

        if phoneme_frames and total_duration > 0:
            phoneme_frames[-1].end_time = total_duration
            phoneme_frames[-1].duration = (
                phoneme_frames[-1].end_time - phoneme_frames[-1].start_time
            )

        return PhonemeAlignment(
            text=text,
            phoneme_frames=phoneme_frames,
            audio_path=audio_path
        )
    
    def _is_vowel(self, phoneme: str, language: str) -> bool:
        """判断是否是元音"""
        phoneme = phoneme.lower()
        
        if language == 'zh':
            vowels = ['a', 'o', 'e', 'i', 'u', 'v', 'ü',
                     'ai', 'ei', 'ui', 'ao', 'ou', 'iu',
                     'ie', 've', 'er', 'an', 'en', 'in', 'un', 'vn',
                     'ang', 'eng', 'ing', 'ong']
            return phoneme in vowels
        else:
            vowels = ['a', 'aa', 'ah', 'ae', 'ao', 'e', 'eh', 'ey',
                     'i', 'ih', 'iy', 'o', 'oh', 'ow', 'u', 'uh', 'uw',
                     'ax', 'ix', 'ay', 'aw', 'oy', 'er', 'ar']
            return phoneme in vowels
    
    def _get_phoneme_weight(self, phoneme: str, language: str) -> float:
        """
        获取音素时长权重（权重越高，分配时间越长）
        
        元音通常比辅音长，但不要过于极端
        """
        phoneme = phoneme.lower()
        
        if language == 'zh':
            # 中文拼音
            # 韵母（长）
            vowels = ['a', 'o', 'e', 'i', 'u', 'v', 'ü',
                     'ai', 'ei', 'ui', 'ao', 'ou', 'iu',
                     'ie', 've', 'ue', 'er', 'an', 'en', 'in', 'un', 'vn',
                     'ang', 'eng', 'ing', 'ong',
                     'ia', 'iao', 'ian', 'iang', 'iong',
                     'ua', 'uo', 'uai', 'uan', 'uang', 'ueng',
                     'van', 'vang']
            if phoneme in vowels:
                return 1.8  # 降低权重差异
            # 声母（短）
            return 1.0
        else:
            # 英文音素：拉大元/辅音权重差距，让元音吃掉词内大部分时间。
            # CosyVoice 英文 word_timestamp 是「词级」(中文是字级)，一个
            # 词如 "reflection" 9 音素挤进 0.6s，等权分配每个音素 60ms，
            # 嘴型还没张到目标值就被切走 → 视觉上"嘴在抖"。
            # 经验值：长元音 3.5、双元音 3.0、短元音 2.2、流音 1.0、
            # 鼻音 0.9、辅音 0.6。这样 reflection 的 3 个元音吃掉
            # 约 70% 词时长，嘴型才张得开。
            long_vowels = ['aa', 'ah', 'ae', 'ao', 'eh',
                          'ih', 'iy', 'uh', 'uw', 'ow', 'oh']
            if phoneme in long_vowels:
                return 3.5
            diphthongs = ['ey', 'ay', 'aw', 'oy', 'er', 'ar']
            if phoneme in diphthongs:
                return 3.0
            short_vowels = ['a', 'e', 'i', 'o', 'u', 'ax', 'ix']
            if phoneme in short_vowels:
                return 2.2
            liquids = ['l', 'r', 'w', 'y']
            if phoneme in liquids:
                return 1.0
            nasals = ['m', 'n', 'ng']
            if phoneme in nasals:
                return 0.9
            # 辅音（爆破/摩擦/破擦音都很短）
            return 0.6
    
    def _estimate_duration(self, text: str) -> float:
        """估算语音时长 (秒)"""
        # 中文：约 4-5 字/秒
        # 英文：约 12-15 字符/秒
        language = self._detect_language(text) if self.language == 'auto' else self.language
        
        if language == 'zh':
            return max(1.0, len([c for c in text if '\u4e00' <= c <= '\u9fff']) / 4.5)
        else:
            return max(1.2, len(text) / 12.0)