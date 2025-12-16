# import os
# import numpy as np
# from PIL import Image
# from glob import glob

# #MASK_DIR = "/home/ho/BEADtrain/"    
# # MASK_DIR = "/home/ho/Downloads/bead/SegmentationObject/"
# MASK_DIR = "/home/ho/BEADtrain/REAL/mask"


# mask_paths = glob(os.path.join(MASK_DIR, "*.png"))

# # 고유 픽셀값들 수집
# all_values = set()

# for path in mask_paths:
#     mask = Image.open(path).convert("L")
#     mask_np = np.array(mask)
#     unique = np.unique(mask_np)
#     all_values.update(unique)

# print("🧪 전체 마스크에서 발견된 픽셀 값들:", sorted(list(all_values)))



# #-----------------손상된 이미지 찾기
# import os
# import numpy as np
# from PIL import Image
# from glob import glob

# MASK_DIR = "/home/ho/BEADtrain/REAL/mask"
# mask_paths = glob(os.path.join(MASK_DIR, "*.png")) + glob(os.path.join(MASK_DIR, "*.PNG"))

# all_values = set()

# for path in mask_paths:
#     try:
#         mask = Image.open(path).convert("L")
#         mask_np = np.array(mask)
#         mask.thumbnail((512, 512))  # 임시로 크기 축소

#         unique = np.unique(mask_np)
#         all_values.update(unique)
#     except Exception as e:
#         print(f"⚠️ 파일 오류: {path}, {e}")

# print("🧪 전체 마스크에서 발견된 픽셀 값들:", sorted(list(all_values)))

from PIL import Image
import numpy as np

path = "/home/ho/BEADtrain/REAL/mask/KakaoTalk_20250819_164238229.png"
img = Image.open(path).convert("L")
arr = np.array(img)
print(np.unique(arr))