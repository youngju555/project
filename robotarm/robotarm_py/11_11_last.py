# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 + YOLO 감지 + OpenCV 기반 각도 추정
(세그멘테이션 제거 / 과부하 최소화 버전)
"""

import time, argparse, threading, cv2, math, numpy as np
from ultralytics import YOLO

try:
    from pymycobot.mycobot320 import MyCobot320 as CobotClass
except Exception:
    from pymycobot.mycobot import MyCobot as CobotClass


# ===================== 포즈 설정 =====================
# POSE_HOME   = [-254.1, -107.3, 332.5, -169.37, 9.44, 91.75]
POSE_HOME   = [-248.8, 45.9, 291.6, -175.41, 3.67, -0.74]
POSE_CLEAR  = [-222.1,  -30.3, 346.7, -176.15, -2.44, 119.98]
POSE_PLACE1 = [-152.4,  181.4, 228.1, -170.3,   6.5,   38.8]   # 양품
POSE_PLACE2 = [ -37.3,  318.6, 166.2,  162.81, -2.81, -29.21]  # 불량품
DEFAULT_SPEED = 25

# ===================== 보정 파라미터 =====================
SCALE_X = -(1) * -(0.6)
SCALE_Y = -(1) * -(0.05)
OFFSET_X = 70.0
OFFSET_Y = 0.0
is_defect = False


# ===================== 픽셀 → 로봇 변환 =====================
def pixel_to_robot(cx, cy, frame_w, frame_h):
    dx = (cx - frame_w / 2) * SCALE_X
    dy = (cy - frame_h / 2) * SCALE_Y
    robot_x = POSE_HOME[0] + OFFSET_X - dy
    robot_y = POSE_HOME[1] + OFFSET_Y - dx
    robot_z = POSE_HOME[2]
    return [robot_x, robot_y, robot_z, POSE_HOME[3], POSE_HOME[4], POSE_HOME[5]]


# ===================== OpenCV 기반 각도 계산 =====================
def get_angle_from_roi(frame, x1, y1, x2, y2):
    """ROI 잘라서 minAreaRect 기반 각도 계산"""
    roi = frame[int(y1):int(y2), int(x1):int(x2)]
    if roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    ((_, _), (_, _), angle) = rect

    # 보정
    if angle < -45:
        angle = 90 + angle

    return angle


# ===================== YOLO 감지 =====================
def detect_object(model, frame):
    global accuracy
    results = model.predict(frame, imgsz=640, conf=0.55, verbose=False)
    r = results[0]
    boxes = r.boxes
    frame_vis = frame.copy()

    if len(boxes) == 0:
        return frame_vis, None, None, None, None, None, False

    box = max(boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = box.xyxy[0]
    conf = float(box.conf[0])
    cls = int(box.cls[0])
    name = r.names[cls]
    accuracy = conf

    # 중심좌표 및 각도 계산
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    angle = get_angle_from_roi(frame, x1, y1, x2, y2)

    # === 🔍 불량(노란점) 탐지 ===
    roi = frame[int(y1):int(y2), int(x1):int(x2)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 노란색 범위
    lower_yellow = np.array([20, 100, 150])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    yellow_ratio = np.sum(mask_yellow > 0) / (roi.shape[0] * roi.shape[1])

    is_defect = yellow_ratio > 0.002   # ROI 중 0.2% 이상 노란 픽셀이면 불량
    if is_defect:
        status_text = "DEFECT (YELLOW DOT)"
        color_box = (0, 0, 255)
    else:
        status_text = "OK"
        color_box = (0, 255, 0)

    # === 시각화 ===
    cv2.rectangle(frame_vis, (int(x1), int(y1)), (int(x2), int(y2)), color_box, 2)
    cv2.circle(frame_vis, (cx, cy), 5, (0, 255, 255), -1)
    cv2.putText(frame_vis,
                f"{name} ({conf:.2f}) | {angle:.1f}° | {status_text}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)

    print(f"🧩 감지됨: {name} | conf={conf:.2f} | angle={angle:.1f}° | defect={is_defect} | yellow_ratio={yellow_ratio:.4f}")

    return frame_vis, (cx, cy), angle, conf, frame.shape[1], frame.shape[0], is_defect




# ===================== 카메라 스레드 =====================
def camera_thread(stop_event, frame_container):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라 열기 실패")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            frame_container["frame"] = frame
        time.sleep(0.03)
    cap.release()


# ===================== 메인 루틴 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--model", type=str, default="best.pt")
    args = parser.parse_args()

    model = YOLO(args.model)
    print("✅ YOLO 모델 로드 완료 (세그멘테이션 OFF, 과부하 최소화)")

    mc = CobotClass(args.port, args.baud)
    mc.power_on()
    time.sleep(3)
    print(mc.is_all_servo_enable())
    # mc.send_angles([0,0,0,0,0,0],20)
    time.sleep(0.5)
    mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
    mc.set_gripper_mode(0)
    mc.set_electric_gripper(0)
    mc.set_gripper_value(50, 20, 1)
    print("🏠 홈 포즈 도달 및 초기화 완료")

    frame_container, stop_event = {"frame": None}, threading.Event()
    cam_thread = threading.Thread(target=camera_thread, args=(stop_event, frame_container), daemon=True)
    cam_thread.start()

    print("📷 감지 중 (3초 이상 유지 시 픽업 시작)")
    detect_start, detected, detected_angle = None, None, None

    while not stop_event.is_set():
        frame = frame_container.get("frame")
        if frame is None:
            continue
        frame_vis, result, angle, conf, fw, fh, is_defect = detect_object(model, frame)


        if result:
            cx, cy = result
            if detect_start is None:
                detect_start = time.time()
            elif time.time() - detect_start > 3.0:
                print(f"🟢 물체 확정: (cx={cx:.1f}, cy={cy:.1f}), angle={angle:.1f}°")
                detected = pixel_to_robot(cx, cy, fw, fh)
                detected_angle = angle
                stop_event.set()
                break
        else:
            detect_start = None

        cv2.imshow("Camera", frame_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cam_thread.join()
    cv2.destroyAllWindows()

    if not detected:
        print("❌ 감지 실패 또는 중단됨")
        return

    x, y, z, r, p, yaw = detected
    print(f"🎯 이동 목표 좌표: {detected}")

    # === 1️⃣ 좌표 이동 (Z+70 지점)
    mc.send_coords([x, y, z + 70, r, p, yaw], 25, 0)
    time.sleep(1.5)

    # === 2️⃣ 픽업 직전 각도 재검출 및 보정 ===
    frame = frame_container.get("frame")
    if frame is not None:
        re_frame, _, new_angle, _, _, _, _ = detect_object(model, frame)
        if new_angle is not None:
            print(f"📐 픽업 직전 각도 재측정: {new_angle:.1f}°")
            angles = mc.get_angles()
            if angles:
                angles[5] += new_angle
                mc.send_angles(angles, 25)
                time.sleep(2)
                print(f"🧭 그리퍼 회전 보정 완료 ({new_angle:.1f}°)")

    # === 3️⃣ 점진적 하강 ===

    down = mc.get_coords()
    print("down : ", down)
    if down:
        down[0], down[1] = x, y
        down[2] -= 0
        mc.send_coords(down, 20, 0)
        time.sleep(2.5)
        print("📉 하강 중:", mc.get_coords())
        time.sleep(2.5)
        ###############################################################################
        exit()

    # === 4️⃣ 집기 ===
    mc.set_gripper_value(20, 30, 1)
    time.sleep(1.5)

    # === 5️⃣ 상승 ===
    up = mc.get_coords()
    up[2] += 100
    mc.send_coords(up, 25, 0)
    time.sleep(1.5)

    # === 6️⃣ 분류 이동 ===
    mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
    time.sleep(1)



    if is_defect:
        print("🔴 불량품 (노란 점 감지됨)")
        mc.send_coords(POSE_PLACE2, DEFAULT_SPEED, 0)
    else:
        print("🟢 양품 (정상)")
        mc.send_coords(POSE_PLACE1, DEFAULT_SPEED, 0)

    time.sleep(2)
    mc.set_gripper_value(50, 20, 1)
    mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
    time.sleep(1)
    mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
    print("🏁 픽업 및 분류 완료 → 홈 복귀")


if __name__ == "__main__":
    main()