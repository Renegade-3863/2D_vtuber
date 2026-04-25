from dataclasses import dataclass
from typing import Dict, Any, List, Union

import torch
from tha3.poser.modes.load_poser import load_poser


@dataclass
class LLMState:
    emotion: str = "neutral"                # neutral|happy|sad|angry|surprised|thinking|serious
    intensity: float = 0.5                  # 0~1
    motion_hint: str = "none"               # none|nod|shake|tilt_left|tilt_right


class THAPoseMapper:
    def __init__(self, model_name: str = "separable_half", device: Union[str, torch.device] = "cuda"):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.poser = load_poser(model_name, self.device)
        self.dtype = self.poser.get_dtype()

        self.num_params = self.poser.get_num_parameters()
        self.name_to_index = {}
        self.name_to_range = {}

        idx = 0
        for group in self.poser.get_pose_parameter_groups():
            r = group.get_range()
            for pname in group.get_parameter_names():
                self.name_to_index[pname] = idx
                self.name_to_range[pname] = r
                idx += 1

        self.idx = lambda n: self.name_to_index[n]

    def _clamp_to_range(self, name: str, v: float) -> float:
        lo, hi = self.name_to_range.get(name, (0.0, 1.0))
        return max(lo, min(hi, v))

    def _set(self, pose: List[float], name: str, value: float):
        if name not in self.name_to_index:
            return
        pose[self.idx(name)] = self._clamp_to_range(name, value)

    def build_pose_vector(self, llm_state: Dict[str, Any], mouth_open: float = 0.0) -> List[float]:
        emotion = (llm_state.get("emotion") or "neutral").lower()
        intensity = float(llm_state.get("intensity", 0.5))
        intensity = max(0.0, min(1.0, intensity))
        motion_hint = (llm_state.get("motion_hint") or "none").lower()

        pose = [0.0] * self.num_params

        # Mouth (placeholder, replace with viseme later)
        self._set(pose, "mouth_aaa", mouth_open)

        # Emotion templates
        if emotion == "happy":
            self._set(pose, "mouth_raised_corner_left", 0.8 * intensity)
            self._set(pose, "mouth_raised_corner_right", 0.8 * intensity)
            self._set(pose, "eye_happy_wink_left", 0.6 * intensity)
            self._set(pose, "eye_happy_wink_right", 0.6 * intensity)
            self._set(pose, "eyebrow_happy_left", 0.7 * intensity)
            self._set(pose, "eyebrow_happy_right", 0.7 * intensity)

        elif emotion == "sad":
            self._set(pose, "mouth_lowered_corner_left", 0.8 * intensity)
            self._set(pose, "mouth_lowered_corner_right", 0.8 * intensity)
            self._set(pose, "eyebrow_troubled_left", 0.6 * intensity)
            self._set(pose, "eyebrow_troubled_right", 0.6 * intensity)
            self._set(pose, "eye_relaxed_left", 0.5 * intensity)
            self._set(pose, "eye_relaxed_right", 0.5 * intensity)

        elif emotion == "angry":
            self._set(pose, "eyebrow_angry_left", 0.55 * intensity)
            self._set(pose, "eyebrow_angry_right", 0.55 * intensity)
            self._set(pose, "eye_unimpressed_left", 0.35 * intensity)
            self._set(pose, "eye_unimpressed_right", 0.35 * intensity)
            self._set(pose, "mouth_delta", 0.20 * intensity)

        elif emotion == "surprised":
            self._set(pose, "eye_surprised_left", 0.70 * intensity)
            self._set(pose, "eye_surprised_right", 0.70 * intensity)
            self._set(pose, "mouth_ooo", 0.50 * intensity)

        elif emotion == "thinking":
            self._set(pose, "eye_relaxed_left", 0.30 * intensity)
            self._set(pose, "eye_relaxed_right", 0.30 * intensity)
            self._set(pose, "mouth_smirk", 0.20 * intensity)

        # Motion hint (small amplitude)
        if motion_hint == "nod":
            self._set(pose, "head_x", 0.10)
        elif motion_hint == "shake":
            self._set(pose, "head_y", 0.10)
        elif motion_hint == "tilt_left":
            self._set(pose, "neck_z", -0.12)
        elif motion_hint == "tilt_right":
            self._set(pose, "neck_z", 0.12)

        # Tiny breathing baseline
        self._set(pose, "breathing", 0.15)

        return pose

    def to_tensor(self, pose: List[float]) -> torch.Tensor:
        return torch.tensor(pose, device=self.device, dtype=self.dtype)
