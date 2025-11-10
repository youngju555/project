    # -*- coding: utf-8 -*-
"""
MyCobot 320 M5 + YOLO 기반 자동 픽업 (그리퍼 각도보정 send_angles 적용 버전)
"""

import cv2, time, argparse, threading, numpy as np
from ultralytics import YOLO

try:
    from pymycobot.mycobot320 import MyCobot320 as CobotClass
except Exception:
    from pymycobot.mycobot import MyCobot as CobotClass


# ===================== 설정 =====================
POSE_HOME = [-212.7, -175.8, 331.0, -168.64, 7.73, 91.65]
DEFAULT_SPEED = 25

# 실측 보정값
PIXEL_TO_MM = 0.04
OFFSET_X = 70.0
OFFSET_Y = 0.0
OFFSET_Z = 0.0


# ===================== 변환 함수 =====================
def pixel_to_robot(cx, cy, frame_w, frame_h):
    """픽셀 중심(cx, cy)을 로봇 좌표(mm)로 단순 변환"""
    dx = (cx - frame_w / 2) * PIXEL_TO_MM
    dy = (cy - frame_h / 2) * 0.08
    robot_x = POSE_HOME[0] + OFFSET_X - dy
    robot_y = POSE_HOME[1] + OFFSET_Y - dx
    robot_z = POSE_HOME[2]
    return [robot_x, robot_y, robot_z, POSE_HOME[3], POSE_HOME[4], POSE_HOME[5]]


# ===================== YOLO 감지 + 각도 추정 =====================
def detect_object(model, frame):
    results = model.predict(frame, conf=0.7, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    frame_vis = results[0].plot()

    if len(boxes) == 0:
        return frame_vis, None, None

    x1, y1, x2, y2 = boxes[0]
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

    # ROI 잘라서 회전각 검출
    roi = frame[int(y1):int(y2), int(x1):int(x2)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angle = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        ((_, _), (_, _), angle) = rect
        if angle < -45:
            angle = 90 + angle

        # 시각화
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        box[:, 0] += int(x1)
        box[:, 1] += int(y1)
        cv2.drawContours(frame_vis, [box], 0, (255, 255, 0), 2)
        cv2.putText(frame_vis, f"{angle:.1f}°", (cx, cy + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame_vis, (cx, cy), angle


# ===================== 카메라 스레드 =====================
def camera_thread(stop_event, frame_container):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라 열기 실패")
        return
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            frame_container["frame"] = frame
    cap.release()


# ===================== 메인 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--model", type=str, default="best.pt")
    args = parser.parse_args()

    model = YOLO(args.model)
    print("✅ YOLO 모델 로드 완료")

    mc = CobotClass(args.port, args.baud)
    mc.power_on()
    time.sleep(0.5)
    mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
    print("🏠 홈 포즈 도달")
    # mc.set_gripper_value(80, 30, 1)  # 열기
    mc.set_gripper_mode(0)
    mc.set_electric_gripper(0)
    mc.set_gripper_value(50, 20, 1)  # 열림

    # === 카메라 스레드 시작 ===
    frame_container = {"frame": None}
    stop_event = threading.Event()
    cam_thread = threading.Thread(target=camera_thread, args=(stop_event, frame_container), daemon=True)
    cam_thread.start()

    print("📷 카메라 감지 시작 (q 누르면 종료)")
    detect_start, detected, detected_angle = None, None, None

    while not stop_event.is_set():
        frame = frame_container.get("frame")
        if frame is None:
            continue

        frame, result, angle = detect_object(model, frame)
        h, w, _ = frame.shape
        cv2.imshow("Camera", frame)

        if result:
            cx, cy = result
            if detect_start is None:
                detect_start = time.time()
            elif time.time() - detect_start > 2.0:
                print(f"🟢 물체 감지됨 (cx={cx}, cy={cy}, angle={angle:.2f}°)")
                detected = pixel_to_robot(cx, cy, w, h)
                detected_angle = angle
                break
        else:
            detect_start = None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    stop_event.set()
    cam_thread.join()
    cv2.destroyAllWindows()

    # ===================== 픽업 절차 =====================
    if detected:
        print("🎯 변환된 로봇 좌표:", detected)
        print(f"📐 감지된 물체 각도: {detected_angle:.2f}°")

        x, y, z, r, p, yaw = detected

        # 🔄 send_angles() 기반 그리퍼 회전 보정
        if detected_angle is not None:
            angles = mc.get_angles()
            if angles:
                angles[5] += detected_angle * 1.0  # 회전 보정 강도 (1.0~1.5 실험)
                mc.send_angles(angles, 25)
                time.sleep(2)
                print(f"🧭 그리퍼 회전 완료 ({detected_angle:.1f}° 반영됨)")

        # === 1️⃣ 접근 (XY 정렬, 위로 이동) ===
        mc.send_coords([x, y, z + 70, r, p, yaw], 25, 0)
        time.sleep(1.2)

        # === 2️⃣ 점진적 하강 ===
        for step in [10, 20]:
            down = mc.get_coords()
            if down:
                down[0], down[1] = x, y
                down[2] -= step
                mc.send_coords(down, 20, 0)
                time.sleep(0.8)

        # === 3️⃣ 집기 ===
        mc.set_gripper_value(10, 30, 1)
        time.sleep(1.5)

        # === 4️⃣ 점진적 상승 ===
        for step in [30, 60]:
            up = mc.get_coords()
            if up:
                up[2] += step
                mc.send_coords(up, 25, 0)
                time.sleep(0.8)

        # === 5️⃣ 홈 복귀 ===
        mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
        print("🏁 픽업 완료 → 홈 복귀")

    print("🔒 종료 완료")


if __name__ == "__main__":
    main()
