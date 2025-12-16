import os
import cv2
from glob import glob
from tqdm import tqdm

# 원본 디렉토리
IMG_DIR = "/home/ho/BEADtrain/DATA/Test/image"
MASK_DIR = "/home/ho/BEADtrain/DATA/Test/mask"

# 저장 디렉토리
IMG_OUT_DIR = "/home/ho/BEADtrain/DATA/Resized/image_1920x1080"
MASK_OUT_DIR = "/home/ho/BEADtrain/DATA/Resized/mask_1920x1080"

os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(MASK_OUT_DIR, exist_ok=True)

# 타겟 크기
TARGET_SIZE = (1920, 1080)  # (width, height)

# 이미지 리스트 기준으로 처리
image_paths = sorted(glob(os.path.join(IMG_DIR, "*.jpg")))

for img_path in tqdm(image_paths, desc="🔄 이미지 & 마스크 resize 중"):
    fname = os.path.splitext(os.path.basename(img_path))[0]

    # 이미지 읽고 resize
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, TARGET_SIZE)
    cv2.imwrite(os.path.join(IMG_OUT_DIR, fname + ".jpg"), resized_img)

    # 마스크 읽고 resize (INTER_NEAREST 유지!)
    mask_path = os.path.join(MASK_DIR, fname + ".png")
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        resized_mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(MASK_OUT_DIR, fname + ".png"), resized_mask)
    else:
        print(f"⚠️ 마스크 없음: {fname}.png")

