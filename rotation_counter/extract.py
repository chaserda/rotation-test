# Video preprocessing: sample JPEG frames (no tracking / optical flow).

from __future__ import annotations

import cv2


# Decode/sample/resize only. Returns JPEG byte strings.
def extract_frames(
    video_path: str,
    fps_target: float = 5.0,
    max_width: int = 640,
) -> list[bytes]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip = max(1, int(round(original_fps / fps_target)))

    frames: list[bytes] = []
    idx = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if idx % skip == 0:
            h, w = image.shape[:2]
            if w > max_width:
                scale = max_width / w
                image = cv2.resize(image, (max_width, int(h * scale)))
            ok_enc, buf = cv2.imencode(
                ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            if not ok_enc:
                raise RuntimeError("Failed to encode frame")
            frames.append(buf.tobytes())
        idx += 1

    cap.release()
    return frames
