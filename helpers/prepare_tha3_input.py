# save as helpers/prepare_tha3_input.py
# pip install pillow numpy opencv-python

import cv2
import numpy as np
from PIL import Image


def alpha_bbox(rgba):
    a = rgba[:, :, 3]
    ys, xs = np.where(a > 0)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def detect_face_gray(gray):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        return None
    # pick largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return x, y, w, h


def prepare(
    input_png,
    output_png,
    out_size=512,
    target_head=180,
    target_face_y_ratio=0.38,
    min_body_fill=0.85,
):
    im = Image.open(input_png).convert("RGBA")
    arr = np.array(im)

    # crop to character bbox by alpha
    box = alpha_bbox(arr)
    if box is None:
        raise ValueError("No non-transparent pixels found.")
    x0, y0, x1, y1 = box
    crop = arr[y0:y1 + 1, x0:x1 + 1]

    # estimate face size
    gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_RGB2GRAY)
    face = detect_face_gray(gray)

    # Base scale: make face reasonably large
    if face is not None:
        _, _, fw, fh = face
        face_size = max(fw, fh)
        scale = target_head / max(face_size, 1)
    else:
        # Fallback: fill enough vertical space
        scale = (out_size * min_body_fill) / max(crop.shape[0], 1)

    # Prevent tiny character output
    min_scale_for_body = (out_size * min_body_fill) / max(crop.shape[0], 1)
    scale = max(scale, min_scale_for_body)

    new_w = max(1, int(crop.shape[1] * scale))
    new_h = max(1, int(crop.shape[0] * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # paste into 512 canvas, place face around upper-middle
    canvas = np.zeros((out_size, out_size, 4), dtype=np.uint8)

    if face is not None:
        fx, fy, fw, fh = face
        fx = int(fx * scale)
        fy = int(fy * scale)
        fw = int(fw * scale)
        fh = int(fh * scale)
        face_cx = fx + fw // 2
        face_cy = fy + fh // 2
    else:
        face_cx = new_w // 2
        face_cy = int(new_h * 0.33)

    target_cx = out_size // 2
    target_cy = int(out_size * target_face_y_ratio)
    ox = target_cx - face_cx
    oy = target_cy - face_cy

    # alpha blend paste
    x_start = max(0, ox)
    y_start = max(0, oy)
    x_end = min(out_size, ox + new_w)
    y_end = min(out_size, oy + new_h)
    sx0 = max(0, -ox)
    sy0 = max(0, -oy)
    sx1 = sx0 + (x_end - x_start)
    sy1 = sy0 + (y_end - y_start)

    if x_end > x_start and y_end > y_start:
        src = resized[sy0:sy1, sx0:sx1]
        dst = canvas[y_start:y_end, x_start:x_end]
        a = src[:, :, 3:4].astype(np.float32) / 255.0
        dst[:, :, :3] = (src[:, :, :3] * a + dst[:, :, :3] * (1 - a)).astype(np.uint8)
        dst[:, :, 3] = np.maximum(dst[:, :, 3], src[:, :, 3])

    Image.fromarray(canvas, "RGBA").save(output_png)
    print(f"Saved: {output_png}")


if __name__ == "__main__":
    prepare(
    "data/images/testPic_transparent.png",
    "data/images/testPic_tha3_ready.png",
    out_size=512,
    target_head=150,          # 从 180 降到 150
    target_face_y_ratio=0.42, # 脸往下放一点
    min_body_fill=0.78        # 整体别撑太满
)
