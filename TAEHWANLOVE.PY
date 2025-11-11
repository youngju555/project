from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

# -------------------------
# ⚙️ 설정
# -------------------------
model = YOLO(r"C:\Users\peo00\runs\segment\blue_box_seg_training3\weights\best.pt")  # YOLO 모델 경로
cap = cv2.VideoCapture(1)
# 안정화 파라미터
alpha = 0.2                 # EMA 필터 강도
angle_buffer = deque(maxlen=10)  # 최근 N프레임 평균
angle_lock_threshold = 0.8  # 각도 변화 허용 한계
lock_counter = 0
locked_angle = None
smooth_angle = None
last_angle = None

print("🎯 실행 중: 'Q'키로 종료")

# -------------------------
# ⚙️ 각도 계산 함수
# -------------------------
def get_angle_from_roi(frame, x1, y1, x2, y2):
    roi = frame[int(y1):int(y2), int(x1):int(x2)]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 300:
        return None

    rect = cv2.minAreaRect(c)
    ((_, _), (w, h), angle) = rect

    # 보정
    if angle < -45:
        angle += 90
    if w < h:
        angle = angle
    else:
        angle = angle + 90

    return angle % 90  # 0~180도로 제한
    # return angle % 180  # 0~180도로 제한


# -------------------------
# ⚙️ 메인 루프
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        if len(boxes) == 0:
            continue

        x1, y1, x2, y2 = boxes[0].astype(int)
        conf = confs[0]
        angle = get_angle_from_roi(frame, x1, y1, x2, y2)

        # ===== 안정화 필터 =====
        if angle is not None:
            angle_buffer.append(angle)
            avg_angle = np.mean(angle_buffer)

            if smooth_angle is None:
                smooth_angle = avg_angle
            else:
                smooth_angle = alpha * avg_angle + (1 - alpha) * smooth_angle

            # LOCK
            if abs(smooth_angle - avg_angle) < angle_lock_threshold:
                lock_counter += 1
                if lock_counter > 10:
                    locked_angle = smooth_angle
            else:
                lock_counter = 0
                locked_angle = None

            last_angle = smooth_angle

        display_angle = locked_angle if locked_angle is not None else (smooth_angle or 0)

        # ===== 정확도 기반 분류 =====
        classify = "불량품" if conf < 0.7 else "양품"

        # ===== 시각화 =====
        color = (0, 0, 255) if classify == "불량품" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Angle: {display_angle:.2f}°", (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, classify, (x1, y2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLO + OpenCV Angle Test (Yellow Removed)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
