import cv2
import os

# [중요] 카메라 장치 번호 확인
# /dev/video0 이었으므로 '0'을 입력
CAP_INDEX = 0 

# 저장할 폴더 생성
save_dir = "calibration_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cap = cv2.VideoCapture(CAP_INDEX)
if not cap.isOpened():
    print(f"오류: 카메라 {CAP_INDEX}번을 열 수 없습니다.")
    print("팁: VM에서 카메라가 사용 중인지 확인하세요 (예: cheese)")
    exit()

# [중요] v4l2-ctl에서 확인한 yuyv 포맷과 해상도 설정
# (640x480이 목록에 있었으므로)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

img_count = 0
print("캘리브레이션 이미지 수집 시작.")
print("체커보드를 들고 다양한 각도, 거리, 위치에서 [s] 키를 누르세요.")
print("총 15~20장 정도 저장하세요.")
print("[q] 키를 누르면 종료합니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        # 이미지 저장
        img_name = os.path.join(save_dir, f"calib_{img_count:02d}.png")
        cv2.imwrite(img_name, frame)
        print(f"{img_name} 저장됨!")
        img_count += 1

cap.release()
cv2.destroyAllWindows()