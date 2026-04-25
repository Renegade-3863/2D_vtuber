"""
待机动画交互式调试器

实时预览和调试 IdlePerformance 中的每个待机动画，支持：
  - 数字键 1-8 即时触发对应动画（随时切换，自动打断当前动画）
  - ↑/↓ 调整播放速度 (0.25x ~ 3.0x)
  - SPACE 暂停/恢复
  - → 暂停时单帧步进
  - Tab 切换交互状态 (idle → listening → thinking → speaking)
  - B 切换 Baseline idle 是否叠加(默认开启，关闭后只看纯表演动画)
  - R 重放当前动画
  - I 进入空闲模式等待随机触发（测试自然触发逻辑）
  - Q 退出

屏幕 OSD 信息：
  - 动画名 + 进度条
  - 播放速度、交互状态
  - 当前帧所有非零参数及其数值

用法：
    python ai_runtime/debug_idle_animation.py
    python ai_runtime/debug_idle_animation.py --image data/images/MyChar.png
"""

import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tha3.poser.modes.load_poser import load_poser
from tha3.util import extract_pytorch_image_from_PIL_image, rgba_to_numpy_image
from ai_runtime.tha_pose_mapper import THAPoseMapper

# 从 tha_lip_sync_render 导入动画类
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "tha_module",
    str(Path(__file__).parent / "tha_lip_sync_render.py"),
)
_tha = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tha)

IdlePerformance = _tha.IdlePerformance
IdleAnimator = _tha.IdleAnimator
add_pose = _tha.add_pose
smooth_pose = _tha.smooth_pose

# ── 动画名 → 数字键映射 ─────────────────────────────────
PERF_NAMES = [p[0] for p in IdlePerformance.PERFORMANCES]
PERF_KEYS = {ord(str(i + 1)): name for i, name in enumerate(PERF_NAMES)}
# 9/0 留给未来扩展

INTERACTION_STATES = ["idle", "listening", "thinking", "speaking", "post_speak"]

HELP_TEXT = [
    "1-8: trigger anim  |  Up/Down: speed",
    "SPACE: pause  |  Right: step  |  R: replay",
    "Tab: state  |  B: baseline  |  I: idle  |  Q: quit",
]


PANEL_W = 320  # 右侧调试面板宽度


def build_debug_panel(char_h: int, info: dict, params: dict) -> np.ndarray:
    """生成独立的调试面板图像（纯黑底），不遮挡角色"""
    panel = np.zeros((char_h, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (25, 25, 25)  # 深灰底

    font = cv2.FONT_HERSHEY_SIMPLEX
    x0 = 10  # 左边距
    y = 22

    # ── 动画名 + 状态 ──
    anim_name = info.get("anim", "none")
    state = info.get("state", "idle")
    speed = info.get("speed", 1.0)
    paused = info.get("paused", False)
    baseline = info.get("baseline", True)

    cv2.putText(panel, f"Anim: {anim_name}", (x0, y), font, 0.55, (0, 255, 100), 1, cv2.LINE_AA)
    y += 22
    status_str = "PAUSED" if paused else f"{speed:.2f}x"
    cv2.putText(panel, f"{status_str}  state={state}  base={'ON' if baseline else 'OFF'}",
                (x0, y), font, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    y += 20

    # ── 进度条 ──
    progress = info.get("progress", 0.0)
    duration = info.get("duration", 0.0)
    bar_w = PANEL_W - 20
    cv2.rectangle(panel, (x0, y), (x0 + bar_w, y + 10), (60, 60, 60), -1)
    fill = int(bar_w * min(1.0, progress))
    if fill > 0:
        cv2.rectangle(panel, (x0, y), (x0 + fill, y + 10), (0, 220, 100), -1)
    pct = f"{progress * 100:.0f}%  ({progress * duration:.2f}s / {duration:.1f}s)"
    cv2.putText(panel, pct, (x0, y + 22), font, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
    y += 32

    # ── 快捷键映射 ──
    cv2.line(panel, (x0, y), (PANEL_W - 10, y), (60, 60, 60), 1)
    y += 14
    for i, name in enumerate(PERF_NAMES):
        color = (0, 255, 180) if anim_name == name else (120, 120, 120)
        cv2.putText(panel, f"[{i + 1}] {name}", (x0, y), font, 0.38, color, 1, cv2.LINE_AA)
        y += 15
    y += 6

    # ── 帮助 ──
    cv2.line(panel, (x0, y), (PANEL_W - 10, y), (60, 60, 60), 1)
    y += 14
    for ht in HELP_TEXT:
        cv2.putText(panel, ht, (x0, y), font, 0.30, (130, 130, 130), 1, cv2.LINE_AA)
        y += 13
    y += 8

    # ── 参数面板 ──
    cv2.line(panel, (x0, y), (PANEL_W - 10, y), (60, 60, 60), 1)
    y += 16
    cv2.putText(panel, "Active Params:", (x0, y), font, 0.42, (100, 200, 255), 1, cv2.LINE_AA)
    y += 18

    if params:
        sorted_params = sorted(params.items(), key=lambda kv: abs(kv[1]), reverse=True)
        sorted_params = [(k, v) for k, v in sorted_params if abs(v) > 0.005]

        for name, val in sorted_params:
            if y > char_h - 8:
                break
            color = (100, 255, 100) if val > 0 else (100, 150, 255)
            cv2.putText(panel, f"{name}", (x0, y), font, 0.34, color, 1, cv2.LINE_AA)
            cv2.putText(panel, f"{val:+.3f}", (x0 + 185, y), font, 0.34, color, 1, cv2.LINE_AA)
            # 条形图
            bar_max = 90
            bar_len = int(min(1.0, abs(val)) * bar_max)
            bar_color = (0, 200, 0) if val > 0 else (200, 100, 50)
            cv2.rectangle(panel, (x0 + 230, y - 7), (x0 + 230 + bar_len, y - 2), bar_color, -1)
            y += 15

    return panel


def main():
    parser = argparse.ArgumentParser(description="待机动画交互式调试器")
    parser.add_argument("--image", type=str, default=None, help="角色图片路径")
    parser.add_argument("--model", type=str, default="separable_half", help="THA3 模型")
    args = parser.parse_args()

    # ── 加载模型 ──────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[debug] Loading THA3 model '{args.model}' on {device}...")
    poser = load_poser(args.model, device)
    dtype = poser.get_dtype()
    num_params = poser.get_num_parameters()

    repo_root = Path(__file__).resolve().parents[1]
    img_path = Path(args.image) if args.image else repo_root / "data/images/Talking-Head-Anime-①.png"
    if not img_path.exists():
        print(f"找不到图片: {img_path}")
        return
    print(f"[debug] Loading image: {img_path.name}")
    pil = Image.open(img_path).convert("RGBA")
    torch_image = extract_pytorch_image_from_PIL_image(pil).to(device).to(dtype)

    mapper = THAPoseMapper(model_name=args.model, device=device)

    # ── 初始化动画系统 ────────────────────────────────────
    idle_anim = IdleAnimator()
    idle_perf = IdlePerformance()
    idle_anim._idle_perf = idle_perf  # 为了让 idle_anim.sample 能触发表演

    current_pose = [0.0] * num_params

    # ── 状态变量 ──────────────────────────────────────────
    speed = 1.0
    paused = False
    baseline_on = True          # 是否叠加 idle 基础动画
    interaction_state = "idle"
    state_idx = 0
    forced_anim = None          # 手动触发的动画名
    idle_mode = False           # True = 等待自然随机触发

    # 虚拟时钟（可暂停/变速）
    virtual_time = time.time()
    last_real_time = time.time()
    step_once = False           # 单帧步进标记

    # ── 窗口 ──────────────────────────────────────────────
    win_name = "Idle Animation Debugger"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)

    print("\n" + "=" * 56)
    print("  待机动画交互式调试器")
    print("=" * 56)
    print("  数字键 1-8 触发动画：")
    for i, name in enumerate(PERF_NAMES):
        dur = IdlePerformance.PERFORMANCES[i][1]
        print(f"    [{i + 1}] {name:<15s} ({dur:.1f}s)")
    print("  SPACE=暂停  →=步进  ↑↓=速度  Tab=状态")
    print("  B=基线开关  R=重播  I=空闲模式  Q=退出")
    print("=" * 56 + "\n")

    fps_target = 30
    frame_count = 0

    while True:
        real_now = time.time()
        dt_real = real_now - last_real_time
        last_real_time = real_now

        # 更新虚拟时钟
        if not paused or step_once:
            virtual_time += dt_real * speed
            step_once = False

        now = virtual_time

        # ── 手动触发动画 ──
        if forced_anim is not None:
            idle_perf._current = forced_anim
            idle_perf._duration = next(
                p[1] for p in IdlePerformance.PERFORMANCES if p[0] == forced_anim
            )
            idle_perf._start = now
            idle_perf._interrupted = False
            idle_mode = False
            print(f"▶ {forced_anim} ({idle_perf._duration:.1f}s)")
            forced_anim = None

        # ── 获取动画参数 ──
        idle_dict = {}
        if baseline_on:
            idle_dict = idle_anim.sample(now, speaking_active=(interaction_state == "speaking"),
                                         interaction_state=interaction_state)

        if idle_mode:
            perf_dict = idle_perf.sample(now, interaction_state=interaction_state)
        elif idle_perf._current != "none":
            elapsed = now - idle_perf._start
            if elapsed < idle_perf._duration:
                t = elapsed / idle_perf._duration
                perf_dict = idle_perf._animate(idle_perf._current, t, elapsed)
            else:
                perf_dict = {}
                idle_perf._current = "none"
        else:
            perf_dict = {}

        # 合并到 pose
        combined = {**idle_dict}
        for k, v in perf_dict.items():
            combined[k] = combined.get(k, 0.0) + v

        target_pose = [0.0] * num_params
        for name, val in combined.items():
            idx = mapper.name_to_index.get(name)
            if idx is not None:
                lo, hi = mapper.name_to_range.get(name, (0.0, 1.0))
                target_pose[idx] = max(lo, min(hi, float(val)))

        current_pose = smooth_pose(current_pose, target_pose, alpha=0.35)

        # ── 渲染 ──
        pose_tensor = mapper.to_tensor(current_pose)
        with torch.inference_mode():
            out = poser.pose(torch_image, pose_tensor)[0].detach().cpu()

        img = out.clamp(-1, 1)
        img = (img + 1) / 2
        img = torch.pow(img, 1 / 2.2)
        img_np = img.permute(1, 2, 0).numpy()
        img_bgr = (img_np[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

        # ── OSD ──
        anim_name = idle_perf._current if idle_perf._current != "none" else ("waiting..." if idle_mode else "none")
        progress = 0.0
        duration = 0.0
        if idle_perf._current != "none":
            duration = idle_perf._duration
            elapsed_anim = now - idle_perf._start
            progress = min(1.0, max(0.0, elapsed_anim / duration)) if duration > 0 else 0.0

        osd_info = {
            "anim": anim_name,
            "state": interaction_state,
            "speed": speed,
            "paused": paused,
            "baseline": baseline_on,
            "progress": progress,
            "duration": duration,
        }

        # 角色画面（干净）+ 右侧调试面板
        panel = build_debug_panel(img_bgr.shape[0], osd_info, combined)
        display = np.hstack([img_bgr, panel])

        cv2.imshow(win_name, display)
        frame_count += 1

        # ── 按键处理 ──
        wait_ms = max(1, int(1000 / fps_target))
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q'):
            break

        # 数字键 1-8 触发动画
        if key in PERF_KEYS:
            forced_anim = PERF_KEYS[key]

        # SPACE 暂停/恢复
        if key == ord(' '):
            paused = not paused
            print(f"{'⏸ 暂停' if paused else '▶ 恢复'}")

        # → 单帧步进 (key code 83 = right arrow in OpenCV)
        if key == 83 or key == ord('d'):
            if paused:
                step_once = True

        # ↑ 加速 (key code 82 = up)
        if key == 82 or key == ord('w'):
            speed = min(3.0, speed + 0.25)
            print(f"速度: {speed:.2f}x")

        # ↓ 减速 (key code 84 = down)
        if key == 84 or key == ord('s'):
            speed = max(0.25, speed - 0.25)
            print(f"速度: {speed:.2f}x")

        # Tab 切换交互状态
        if key == 9:  # Tab
            state_idx = (state_idx + 1) % len(INTERACTION_STATES)
            interaction_state = INTERACTION_STATES[state_idx]
            print(f"状态: {interaction_state}")

        # B 切换基线
        if key == ord('b'):
            baseline_on = not baseline_on
            print(f"基线 idle: {'ON' if baseline_on else 'OFF'}")

        # R 重播当前动画
        if key == ord('r'):
            current_name = idle_perf._current
            if current_name != "none":
                forced_anim = current_name
            else:
                print("没有当前动画可重播，按 1-8 选择一个")

        # I 空闲模式（等待自然触发）
        if key == ord('i'):
            idle_mode = not idle_mode
            if idle_mode:
                idle_perf._current = "none"
                idle_perf._next_trigger = now + 2.0
                print("🎲 空闲模式 ON — 等待随机触发")
            else:
                print("🎲 空闲模式 OFF")

        # 检测窗口关闭
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print("调试器已退出")


if __name__ == "__main__":
    main()
