from PIL import Image

img = Image.open("/home/ho/BEADtrain/DATA/Test/image/VT_ST_00_14561917.jpg")
print("모드:", img.mode)


import torch
from torch.cuda.amp import autocast, GradScaler

print("PyTorch 버전:", torch.__version__)
print("AMP 지원 여부:", hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"))


from PIL import Image
import numpy as np

# 마스크 경로
mask_path =  "/home/ho/Downloads/bead/SegmentationClass/KakaoTalk_20250819_163330552.png"

# 흑백(Grayscale)로 불러오기
mask = Image.open(mask_path).convert("L")
mask_np = np.array(mask)

# ✅ 마스크에 존재하는 고유 픽셀 값 출력
unique_values = np.unique(mask_np)
print("🧾 마스크에 존재하는 고유 픽셀 값:", unique_values)



# # source venv/bin/activate 한다음에 이거 픽셀값 0과 255로 바꾸는 코드임
# import os
# import cv2
# import numpy as np

# # 입력/출력 경로
# input_dir = "/home/ho/Downloads/bead/SegmentationObject/"
# output_dir = "/home/ho/Downloads/bead/SegmentationObject_255/"

# # 출력 폴더 없으면 생성
# os.makedirs(output_dir, exist_ok=True)

# # 폴더 안 모든 파일 순회
# for fname in os.listdir(input_dir):
#     if fname.lower().endswith(".png"):
#         in_path = os.path.join(input_dir, fname)
#         out_path = os.path.join(output_dir, fname)

#         # 마스크 읽기 (그레이스케일)
#         mask = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)

#         # 값이 38이면 255로, 나머지는 0으로
#         binary_mask = np.where(mask == 38, 255, 0).astype(np.uint8)

#         # 저장
#         cv2.imwrite(out_path, binary_mask)
#         print(f"✅ {fname} 변환 완료 → {out_path}")

# print("🎉 모든 PNG 변환 완료!")

