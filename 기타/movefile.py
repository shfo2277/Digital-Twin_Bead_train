# import os
# import shutil
# import glob

# # 원본 이미지/마스크 디렉토리
# src_dir = "/home/ho/BEADtrain/DATA/Training/label/TL_VTST_정상"

# # 이동 대상 디렉토리
# dst_dir = "/home/ho/BEADtrain/DATA/Test/json"

# # 디렉토리가 없다면 생성
# os.makedirs(dst_dir, exist_ok=True)

# # .jpg, .png, .json 파일 리스트 수집 및 정렬
# all_files = sorted(
#     glob.glob(os.path.join(src_dir, "*.jpg")) +
#     glob.glob(os.path.join(src_dir, "*.png")) +
#     glob.glob(os.path.join(src_dir, "*.json"))
# )

# # 앞에서부터 2000개 선택
# files_to_move = all_files[:2000]

# print(f"🔍 이동할 파일 수: {len(files_to_move)}개")

# # 파일 이동
# for path in files_to_move:
#     filename = os.path.basename(path)
#     dst_path = os.path.join(dst_dir, filename)
#     shutil.move(path, dst_path)

# print("✅ 총 2000개의 .jpg/.png/.json 파일이 성공적으로 이동되었습니다!")



#--------------이미지 복사 
import os
import shutil
import glob

# 원본 마스크 디렉토리
src_dir = "/home/ho/BEADtrain/DATA/Training/mask"

# 복사할 대상 디렉토리
dst_dir = "/home/ho/BEADtrain/REAL/mask"

# 대상 디렉토리 없으면 생성
os.makedirs(dst_dir, exist_ok=True)

# PNG 파일 경로 가져오기
mask_paths = glob.glob(os.path.join(src_dir, "*.png"))

# 복사 실행
for path in mask_paths:
    filename = os.path.basename(path)
    dst_path = os.path.join(dst_dir, filename)
    shutil.copy(path, dst_path)

print(f"✅ 총 {len(mask_paths)}개의 PNG 마스크가 복사되었습니다.")
