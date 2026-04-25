"""
THA3 嘴型参数定义

这个模块定义了 THA3 系统支持的所有嘴型参数及其含义。
用于文档化和辅助音素映射。
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MouthShape:
    """嘴型定义"""
    name: str
    description: str
    example_phonemes: List[str]  # 示例音素


# THA3 支持的嘴型参数
MOUTH_SHAPES = {
    "mouth_aaa": MouthShape(
        name="mouth_aaa",
        description="张大嘴，如发 'ah' 音",
        example_phonemes=["a", "aa", "ah", "ae"]
    ),
    "mouth_iii": MouthShape(
        name="mouth_iii",
        description="咧嘴笑，如发 'ee' 音",
        example_phonemes=["i", "ii", "iy", "ih"]
    ),
    "mouth_uuu": MouthShape(
        name="mouth_uuu",
        description="圆嘴，如发 'oo' 音",
        example_phonemes=["u", "uu", "uw", "uh"]
    ),
    "mouth_eee": MouthShape(
        name="mouth_eee",
        description="扁平嘴，如发 'eh' 音",
        example_phonemes=["e", "eh"]
    ),
    "mouth_ooo": MouthShape(
        name="mouth_ooo",
        description="小圆嘴，如发 'oh' 音",
        example_phonemes=["o", "oo", "oh"]
    ),
    "mouth_delta": MouthShape(
        name="mouth_delta",
        description="嘴部微张，中性状态",
        example_phonemes=["m", "n", "ng"]
    ),
    "mouth_raised_corner_left": MouthShape(
        name="mouth_raised_corner_left",
        description="左嘴角上扬（微笑）",
        example_phonemes=[]
    ),
    "mouth_raised_corner_right": MouthShape(
        name="mouth_raised_corner_right",
        description="右嘴角上扬（微笑）",
        example_phonemes=[]
    ),
    "mouth_lowered_corner_left": MouthShape(
        name="mouth_lowered_corner_left",
        description="左嘴角下垂（悲伤）",
        example_phonemes=[]
    ),
    "mouth_lowered_corner_right": MouthShape(
        name="mouth_lowered_corner_right",
        description="右嘴角下垂（悲伤）",
        example_phonemes=[]
    ),
    "mouth_smirk": MouthShape(
        name="mouth_smirk",
        description="假笑/得意的笑",
        example_phonemes=[]
    ),
}


def get_mouth_shape_names() -> List[str]:
    """获取所有嘴型参数名称"""
    return list(MOUTH_SHAPES.keys())


def get_mouth_shape_info(name: str) -> MouthShape | None:
    """获取指定嘴型的详细信息"""
    return MOUTH_SHAPES.get(name)