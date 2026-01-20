# 이건 걍 실제 카메라 이미지로 가져와서 이미지 얻기 위한거 라벨링하기 위해
import cv2
import os


# 화면에 표시할 스케일 비율
display_scale = 0.5  # 50% 크기

# ✅ 저장 경로 설정
save_dir = "/home/ho/BEADtrain/REAL/endgrinding"

# ✅ 폴더가 없으면 자동 생성
os.makedirs(save_dir, exist_ok=True)

# ✅ 카메라 열기
#이건 415
#cap = cv2.VideoCapture(4)  # 필요하면 번호 변경
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# ✅ 해상도 요청 (카메라가 지원하면 반영됨)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# ✅ 실제 적용된 해상도 확인
ret, test_frame = cap.read()
if ret:
    print("📏 Actual Resolution:", test_frame.shape)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

print("🎥 Press 's' to SAVE image, 'q' to QUIT")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 읽기 실패")
        break

    # 화면에만 축소
    display_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)

    cv2.imshow("Camera View", display_frame)

    key = cv2.waitKey(1) & 0xFF

    # ✅ 이미지 저장
    if key == ord('s'):
        filename = f"capture_112{count}.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"✅ Saved: {filepath}")
        count += 1

    # ✅ 종료
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 종료")
