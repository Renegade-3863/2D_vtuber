"""
待机动画测试工具

用法（在 tha 环境中运行）：
    # 列出所有动画
    python ai_runtime/test_idle_animation.py --list

    # 测试单个动画（循环播放）
    python ai_runtime/test_idle_animation.py --anim stretch

    # 测试打瞌睡动画（慢放）
    python ai_runtime/test_idle_animation.py --anim doze --speed 0.5

    # 依次播放所有动画
    python ai_runtime/test_idle_animation.py --all

控制：Q 退出 | SPACE 跳过
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tha3.poser.modes.load_poser import load_poser
from tha3.util import extract_pytorch_image_from_PIL_image, rgba_to_numpy_image

# 从 tha_lip_sync_render 导入 IdlePerformance 和 IdleAnimator
import importlib.util
spec = importlib.util.spec_from_file_location(
    "tha_module",
    str(Path(__file__).parent / "tha_lip_sync_render.py")
)
tha_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tha_module)

IdlePerformance = tha_module.IdlePerformance
IdleAnimator = tha_module.IdleAnimator


def build_pose_vector(idle_dict, perf_dict, num_params, name_to_index):
    """将 idle + perf 字典合并为 THA3 pose vector"""
    pose = [0.0] * num_params
    combined = {**idle_dict, **perf_dict}
    for name, value in combined.items():
        idx = name_to_index.get(name)
        if idx is not None:
            pose[idx] = value
    return pose


def play_animation(poser, torch_image, mapper, anim_name, speed=1.0, num_loops=3):
    idle_perf = IdlePerformance()
    idle_anim = IdleAnimator()

    # 强制立即触发指定动画
    idle_perf._current = "none"
    idle_perf._next_trigger = time.time()

    original_pick = idle_perf._pick_performance
    def forced_pick(now):
        idle_perf._current = anim_name
        idle_perf._duration = next(p[1] for p in IdlePerformance.PERFORMANCES if p[0] == anim_name)
        idle_perf._start = now
        print(f"\n▶ 播放: {anim_name} ({idle_perf._duration:.1f}秒, 速度={speed}x, 循环 {loop_count+1}/{num_loops})")
    idle_perf._pick_performance = forced_pick

    fps = 30
    global loop_count
    loop_count = 0

    print(f"\n{'='*50}")
    print(f"测试动画: {anim_name}")
    print(f"按 Q 退出 | SPACE 跳过")
    print(f"{'='*50}")

    while loop_count < num_loops:
        now = time.time()

        idle = idle_anim.sample(now, speaking_active=False, interaction_state="idle")
        perf = idle_perf.sample(now, interaction_state="idle")

        pose_vec = build_pose_vector(idle, perf, poser.get_num_parameters(), mapper.name_to_index)
        pose_tensor = mapper.to_tensor(pose_vec)

        with torch.no_grad():
            out = poser.pose(torch_image, pose_tensor)[0].detach().cpu()

        img = rgba_to_numpy_image(out)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img)

        # 转 BGR 给 OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        cv2.putText(img_bgr, f"{anim_name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(img_bgr, f"Loop {loop_count+1}/{num_loops}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Idle Animation Test", img_bgr)

        key = cv2.waitKey(int(1000 / fps / speed)) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return
        elif key == ord(' '):
            idle_perf._current = "none"
            idle_perf._next_trigger = now + 0.1

        if idle_perf._current == "none" and loop_count < num_loops:
            loop_count += 1
            if loop_count < num_loops:
                idle_perf._next_trigger = now + 0.5

    cv2.destroyAllWindows()
    print(f"\n✓ 完成 {num_loops} 次循环")


def list_animations():
    print("\n🎭 可用待机动画:\n")
    print(f"{'名称':<15} {'时长(秒)':<10} {'权重':<8} {'描述'}")
    print("-" * 70)
    descriptions = {
        "stretch":     "伸懒腰（仰头→后仰→深呼吸→微笑回正）",
        "head_tilt":   "好奇歪头（歪头→停留观察→眨眼→回正）",
        "sigh":        "无聊叹气（低头→撅嘴呼气→回正）",
        "hair_touch":  "撩头发（左歪头→俏皮微笑→回正）",
        "look_around": "张望四周（眼珠左右扫视→头部跟随）",
        "doze":        "打瞌睡（低头闭眼→惊醒→愣住→微笑）",
        "hum":         "哼歌摇摆（左右摇晃+微笑+嘴微动）",
        "pout":        "撅嘴卖萌（歪头+嘟嘴+委屈眨眼）",
    }
    for name, dur, w in IdlePerformance.PERFORMANCES:
        print(f"{name:<15} {dur:<10.1f} {w:<8.1f} {descriptions.get(name, '')}")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="待机动画测试工具")
    parser.add_argument("--list", action="store_true", help="列出所有动画")
    parser.add_argument("--anim", type=str, help="指定动画名称")
    parser.add_argument("--all", action="store_true", help="播放所有动画")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度 (0.5=慢放)")
    parser.add_argument("--loops", type=int, default=3, help="循环次数")
    args = parser.parse_args()

    if args.list:
        list_animations()
        return

    if not args.anim and not args.all:
        parser.print_help()
        list_animations()
        return

    print("[1/3] 加载 THA3 模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    poser = load_poser("separable_half", device)

    print("[2/3] 加载角色图片...")
    repo_root = Path(__file__).resolve().parents[1]
    img_path = repo_root / "data/images/Talking-Head-Anime-①.png"
    if not img_path.exists():
        print(f"❌ 找不到图片: {img_path}")
        return
    pil = Image.open(img_path).convert("RGBA")
    torch_image = extract_pytorch_image_from_PIL_image(pil).to(device).to(poser.get_dtype())

    print("[3/3] 初始化 pose mapper...")
    from ai_runtime.tha_pose_mapper import THAPoseMapper
    mapper = THAPoseMapper(model_name="separable_half", device=device)

    if args.all:
        for name, dur, w in IdlePerformance.PERFORMANCES:
            play_animation(poser, torch_image, mapper, name, args.speed, 2)
            time.sleep(0.5)
    else:
        play_animation(poser, torch_image, mapper, args.anim, args.speed, args.loops)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
