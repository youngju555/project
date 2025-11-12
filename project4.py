#*- coding: utf-8 -*-
"""
MyCobot 320 M5 + YOLO 감지 및 분류 + OpenCV 기반 각도 추정
(카메라 내부 파라미터 직접 내장 버전)
[MERGE] YOLO가 '양품'/'불량'을 직접 분류하는 로직으로 변경
"""
import time, argparse, threading, cv2, math, numpy as np
from ultralytics import YOLO
try:
    from pymycobot.mycobot320 import MyCobot320 as CobotClass
except Exception:
    from pymycobot.mycobot import MyCobot as CobotClass

# ===================== 포즈 설정 =====================
POSE_HOME   = [-264.3, 66.4, 325.0, -177.3, 7.78, 1.83]
POSE_CLEAR  = [-254.4, -17.4, 350.6, -178.78, 15.16, 1.6]
POSE_PLACE1 = [-152.4, 181.4, 228.1, -170.3, 6.5, 38.8]    # 양품
POSE_PLACE2 = [ -37.3, 318.6, 170.2, 162.81, -2.81, -29.21] # 불량품
DEFAULT_SPEED = 25

# ===================== 보정 파라미터 =====================
# [MODIFIED] 실측 비율 조정 — 1픽셀당 약 0.25~0.3mm 수준으로 변경
SCALE_X = 0.33
SCALE_Y = 0.35
OFFSET_X = -5.0
OFFSET_Y = -85.0

# ===================== 카메라 내부 보정 파라미터 =====================
K = np.array([
    [539.1372906745268, 0.0, 329.02126025840977],
    [0.0, 542.3421738705956, 242.1099554052592],
    [0.0, 0.0, 1.0]
])
D = np.array([[0.20528603028454656, -0.766640680691422,
               -0.0009661402178902956, 0.0011189160210831846,
               0.9763000357883636]])

# ===================== 픽셀 → 로봇 변환 =====================
def pixel_to_robot(cx, cy, frame_w, frame_h):
    """
    픽셀 좌표(cx, cy)를 로봇 좌표(mm)로 변환
    - 카메라 중심(frame_w/2, frame_h/2)을 기준
    - SCALE_X, SCALE_Y: mm per pixel
    """
    dx = (cx - frame_w / 2) * SCALE_X
    dy = (cy - frame_h / 2) * SCALE_Y

    # [MODIFIED] 방향 보정: 실제 세팅에 따라 부호가 반대일 수 있음
    # 로봇 기준 X는 전후, Y는 좌우로 보통 배치되므로 아래처럼 구성
    robot_x = POSE_HOME[0] + OFFSET_X + dx    # 위/아래 → 전후
    robot_y = POSE_HOME[1] + OFFSET_Y - dy    # 좌/우 → 좌우
    robot_z = POSE_HOME[2]

    # [DEBUG] 로그 출력
    print(f"[DEBUG] pixel(cx={cx:.1f}, cy={cy:.1f}) "
          f"→ Δ(dx={dx:.1f}, dy={dy:.1f}) "
          f"→ robot(x={robot_x:.1f}, y={robot_y:.1f})")

    return [robot_x, robot_y, robot_z, POSE_HOME[3], POSE_HOME[4], POSE_HOME[5]]

# ===================== OpenCV 기반 각도 계산 =====================
def get_angle_from_roi(frame, x1, y1, x2, y2):
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
    ((_, _), (width, height), angle) = rect
    if width < height:
        angle = 0 + angle
    else:
        if angle < -45:
            angle = 0 + angle
    return angle

# ===================== YOLO 감지 =====================
def detect_object(model, frame):
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
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    angle = get_angle_from_roi(frame, x1, y1, x2, y2)
    is_defect = (name != "OK")
    status_text = "DEFECT" if is_defect else "OK"
    color_box = (0, 0, 255) if is_defect else (0, 255, 0)

    cv2.rectangle(frame_vis, (int(x1), int(y1)), (int(x2), int(y2)), color_box, 2)
    cv2.circle(frame_vis, (cx, cy), 5, (0, 255, 255), -1)
    cv2.putText(frame_vis, f"{name} ({conf:.2f}) | {angle:.1f}° | {status_text}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)

    print(f"[YOLO] 감지됨: {name} | conf={conf:.2f} | angle={angle:.1f}° | defect={is_defect}")
    return frame_vis, (cx, cy), angle, conf, frame.shape[1], frame.shape[0], is_defect

# ===================== 카메라 스레드 =====================
def camera_thread(stop_event, frame_container):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(":x: 카메라 열기 실패")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w, h = 640, 480
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), 5)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            frame_container["frame"] = undistorted
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
    print(":흰색_확인_표시: YOLO 모델 로드 완료 (분류 로직: YOLO)")

    mc = CobotClass(args.port, args.baud)
    mc.power_on()
    time.sleep(1)
    mc.send_angles([0,0,0,0,0,0],20)
    time.sleep(3)
    mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
    time.sleep(2)
    mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
    mc.set_gripper_mode(0)
    mc.set_electric_gripper(0)
    mc.set_gripper_value(50, 20, 1)
    print(":집: 홈 포즈 도달 및 초기화 완료")

    frame_container, stop_event = {"frame": None}, threading.Event()
    cam_thread = threading.Thread(target=camera_thread, args=(stop_event, frame_container), daemon=True)
    cam_thread.start()
    print(":카메라: 감지 중 (3초 이상 유지 시 픽업 시작)")

    detect_start, detected, detected_angle = None, None, None
    confirmed_is_defect = False

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
                print(f":큰_초록색_원: 물체 확정: (cx={cx:.1f}, cy={cy:.1f}), angle={angle:.1f}°")
                detected = pixel_to_robot(cx, cy, fw, fh)
                detected_angle = angle
                confirmed_is_defect = is_defect
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
        print(":x: 감지 실패 또는 중단됨")
        return

    x, y, z, r, p, yaw = detected
    print(f":다트: 이동 목표 좌표: {detected}")

    # === :일: 좌표 이동 (Z+70 지점)
    mc.send_coords([x, y, 325, r, p, yaw], 25, 1)
    time.sleep(3)

    # === :셋: 픽업 ===
    SAFE_Z = 325
    PICK_Z = 282
    mc.send_coords([x, y, SAFE_Z, r, p, yaw], 25, 1)
    time.sleep(3)
    mc.send_coords([x, y, PICK_Z, r, p, yaw], 20, 1)
    time.sleep(3)
    mc.set_gripper_value(10, 30, 1)
    time.sleep(3)
    up = mc.get_coords()
    up[2] += 100
    mc.send_coords(up, 25, 0)
    time.sleep(3)
    mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
    time.sleep(3)

    if confirmed_is_defect:
        print(":빨간색_원: 불량품 (YOLO가 'DEFECT'로 분류)")
        mc.send_coords(POSE_PLACE2, DEFAULT_SPEED, 0)
    else:
        print(":큰_초록색_원: 양품 (YOLO가 'OK'로 분류)")
        mc.send_coords(POSE_PLACE1, DEFAULT_SPEED, 0)

    time.sleep(2)
    mc.set_gripper_value(50, 20, 1)
    mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
    time.sleep(1)
    mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
    print(":체크무늬_깃발: 픽업 및 분류 완료 → 홈 복귀")

if __name__ == "__main__":
    main()
