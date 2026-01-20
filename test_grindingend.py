import os
import cv2
import numpy as np
import time

import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

import pyrealsense2 as rs


# =========================
# CONFIG
# =========================
MODEL_PATH = "/home/ho/BEADtrain/modelresult/2026-01-19_06-31-27/unet_best.pth"

IMAGE_SIZE = 1024
THRESH = 0.5

# 오탐 작은 조각 제거
REMOVE_SMALL_BLOBS = True
MIN_AREA = 800

# RealSense 설정
CAM_W, CAM_H, CAM_FPS = 1280, 720, 30
PLAY_AT_CAMERA_FPS = True


def remove_small_components(mask01, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    out = np.zeros_like(mask01)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out


# =========================
# device / model
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(state_dict, strict=True)
model.eval()

print("✅ model loaded:", MODEL_PATH)

transform = A.Compose([
    A.LongestMaxSize(max_size=IMAGE_SIZE),
    A.PadIfNeeded(
        min_height=IMAGE_SIZE,
        min_width=IMAGE_SIZE,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=0
    ),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2(),
])


# =========================
# RealSense camera
# =========================
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    CAM_W,
    CAM_H,
    rs.format.bgr8,
    CAM_FPS
)

pipeline.start(config)
frame_delay = 1.0 / CAM_FPS

print(f"✅ RealSense started: {CAM_W}x{CAM_H} @ {CAM_FPS}fps")
print("Controls: [q]=quit, [space]=pause/resume")

paused = False
last_time = time.time()


# =========================
# main loop
# =========================
try:
    while True:
        if not paused:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame_bgr = np.asanyarray(color_frame.get_data())
            H, W = frame_bgr.shape[:2]

            t0 = time.time()

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            aug = transform(
                image=frame_rgb,
                mask=np.zeros((H, W, 1), dtype=np.float32)
            )
            x = aug["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x)
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

            pred_1024 = (prob > THRESH).astype(np.uint8)
            pred = cv2.resize(pred_1024, (W, H), interpolation=cv2.INTER_NEAREST)

            if REMOVE_SMALL_BLOBS:
                pred = remove_small_components(pred, MIN_AREA)

            overlay = frame_bgr.copy()
            overlay[pred == 1] = (0, 255, 0)
            out = cv2.addWeighted(frame_bgr, 0.75, overlay, 0.25, 0)

            ratio = float(pred.sum()) / float(pred.size)
            infer_ms = (time.time() - t0) * 1000.0

            cv2.putText(
                out,
                f"bead_ratio={ratio:.4f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
            cv2.putText(
                out,
                f"infer={infer_ms:.1f}ms",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            cv2.imshow("Bead Segmentation (RealSense)", out)

            if PLAY_AT_CAMERA_FPS:
                elapsed = time.time() - last_time
                sleep_t = frame_delay - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
                last_time = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == 32:  # space
            paused = not paused

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
