import os

# image_dir = "/home/ho/BEADtrain/DATA/Training/image"
# mask_dir = "/home/ho/BEADtrain/DATA/Training/mask"

# image_dir = "/home/ho/BEADtrain/REAL/image"
# mask_dir = "/home/ho/BEADtrain/REAL/mask"

image_dir = "/home/ho/BEADtrain/REAL/fitimage"
mask_dir = "/home/ho/BEADtrain/REAL/fitmask"

image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]

image_names = {os.path.splitext(f)[0] for f in image_files}
mask_names = {os.path.splitext(f)[0] for f in mask_files}

common_names = image_names & mask_names

print(f"📷 이미지 파일 수: {len(image_files)}개")
print(f"🧷 마스크 파일 수: {len(mask_files)}개")
print(f"✅ 짝이 맞는 파일 수: {len(common_names)}개")

# 짝이 없는 이미지(.jpg) 삭제
for name in image_names - common_names:
    img_path = os.path.join(image_dir, f"{name}.jpg")
    print(f"삭제 대상 이미지: {img_path}")
    if os.path.exists(img_path):
        os.remove(img_path)
        print(f"🗑️ 삭제된 이미지 파일: {img_path}")

# 짝이 없는 마스크(.png) 삭제
for name in mask_names - common_names:
    mask_path = os.path.join(mask_dir, f"{name}.png")
    print(f"삭제 대상 마스크: {mask_path}")
    if os.path.exists(mask_path):
        os.remove(mask_path)
        print(f"🗑️ 삭제된 마스크 파일: {mask_path}")

