# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 + YOLO 위치 감지 + Confidence(정확도) 기반 불량 판정
[v2.7-stack] 9회 반복 쌓기형 배치 + 픽업 안정화 강화 버전
------------------------------------------------------------
- 기존 v2.6 로직 유지
- 양품 / 불량품 각각 쌓기형 (Z축 +STACK_HEIGHT씩 상승)
- send_coords에서 회전은 1단계로 처리
- time.sleep() 대신 wait_for_robot_stop() 적용 (그리퍼 동작 제외)
- [NEW] Z축 하강을 '절대 275' → '현재 위치에서 상대 하강 GRAB_DELTA_Z' 로 변경
- [NEW] J6 회전 후 (x,y) 재보정 미세 이동
- [NEW] 그리퍼 2단계로 꽉 닫기 (헛집기 방지)
- [NEW] 불량 기준 conf < 0.9 (CONF_GOOD_THRESHOLD 상수로 관리)
------------------------------------------------------------
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

POSE_PLACE1 = [-13.7, 345.8, 167.3, 177.77, 2.49, 5.53]    # 양품
POSE_PLACE1_UP = [-33.5, 208.1, 349.7, -147.75, -3.1, 4.54]
POSE_PLACE2 =  [-269.7, 244.4, 204.6, -152.37, -3.53, 7.72] # 불량품
POSE_PLACE2_UP = [-253.0, 170.2, 366.4, -121.08, 0.6, 4.78]

DEFAULT_SPEED = 15

# ===================== 쌓기 설정 =====================
STACK_HEIGHT = 27  # 한번 놓을 때마다 27mm씩 상승
stack_count_good = 0
stack_count_bad = 0

# ===================== 보정 파라미터 =====================
SCALE_X = 0.35
SCALE_Y = 0.36
OFFSET_X = -5.0
OFFSET_Y = -83.0

# ===================== 카메라 내부 보정 파라미터 =====================
K = np.array([
    [539.1372906745268, 0.0, 329.02126025840977],
    [0.0, 542.3421738705956, 242.1099554052592],
    [0.0, 0.0, 1.0]
])
D = np.array([[0.20528603028454656, -0.766640680691422,
               -0.0009661402178902956, 0.0011189160210831846,
               0.9763000357883636]])

# ===================== 분류 / 픽업 관련 상수 =====================
CONF_GOOD_THRESHOLD = 0.90   # 이 이상이면 양품, 이하면 불량
GRAB_DELTA_Z = 50.0          # 현재 Z에서 이만큼 상대 하강
APPROACH_Z = 325.0           # 픽업 전 접근 높이

# ===================== 픽셀 → 로봇 변환 =====================
def pixel_to_robot(cx, cy, frame_w, frame_h):
    dx = (cx - frame_w / 2) * SCALE_X
    dy = (cy - frame_h / 2) * SCALE_Y

    robot_x = POSE_HOME[0] + OFFSET_X + dx
    robot_y = POSE_HOME[1] + OFFSET_Y - dy
    robot_z = POSE_HOME[2]

    print(f"[DEBUG] pixel→robot: (cx={cx:.1f}, cy={cy:.1f}) → ({robot_x:.1f}, {robot_y:.1f})")
    return [robot_x, robot_y, robot_z, POSE_HOME[3], POSE_HOME[4], POSE_HOME[5]]

# ===================== 로봇 이동 완료 대기 =====================
def wait_for_robot_stop(mc, pos_tolerance=0.8, ang_tolerance=0.5, poll_interval=0.1, max_wait_time=15.0):
    """
    로봇이 이동을 완료하고 멈출 때까지 대기합니다.
    get_coords()를 반복적으로 폴링하여 좌표 변화가 없을 때를 감지합니다.
    """
    print("⏳ 로봇 이동 완료 대기 중...")
    start_time = time.time()

    last_coords = mc.get_coords()
    if not last_coords:
        print("⚠️ 로봇 좌표 초기값 읽기 실패. 2초 대기(Fallback).")
        time.sleep(2.0)
        return

    while True:
        if time.time() - start_time > max_wait_time:
            print(f"⚠️ {max_wait_time}초 타임아웃. 대기 종료.")
            break

        time.sleep(poll_interval)

        current_coords = mc.get_coords()
        if not current_coords:
            print("⚠️ 로봇 좌표 읽기 실패. 재시도...")
            continue

        pos_diff = [abs(c - l) for c, l in zip(current_coords[:3], last_coords[:3])]
        ang_diff = [abs(c - l) for c, l in zip(current_coords[3:], last_coords[3:])]

        is_stopped = all(p < pos_tolerance for p in pos_diff) and \
                     all(a < ang_tolerance for a in ang_diff)

        if is_stopped:
            print("✅ 로봇 정지 확인.")
            break

        last_coords = current_coords

# ===================== OpenCV 기반 각도 계산 =====================
def get_angle_from_roi(frame, x1, y1, x2, y2):
    roi = frame[int(y1):int(y2), int(x1):int(x2)]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    angle = rect[2] 
    if angle < -45:
        angle = 90 + angle
    return angle

# ===================== YOLO 감지 + Confidence 기반 불량 판별 =====================
def detect_object(model, frame):
    results = model.predict(frame, imgsz=640, conf=0.55, verbose=False)
    r = results[0]
    boxes = r.boxes
    frame_vis = frame.copy()

    if len(boxes) == 0:
        return frame_vis, None, None, None, None, None, None, False

    box = max(boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = box.xyxy[0]
    conf = float(box.conf[0])
    cls = int(box.cls[0])
    name = r.names[cls]

    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    angle = get_angle_from_roi(frame, x1, y1, x2, y2)
    is_defect = conf < CONF_GOOD_THRESHOLD

    color_box = (0, 0, 255) if is_defect else (0, 255, 0)
    cv2.rectangle(frame_vis, (int(x1), int(y1)), (int(x2), int(y2)), color_box, 2)
    cv2.circle(frame_vis, (cx, cy), 5, (0, 255, 255), -1)
    cv2.putText(
        frame_vis,
        f"{name} ({conf:.2f}) | {angle:.1f}° | {'DEFECT' if is_defect else 'OK'}",
        (int(x1), int(y1) - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2
    )

    print(f"[YOLO->CONF] 감지: {name} | conf={conf:.2f} | angle={angle:.1f}° | defect={is_defect}")
    return frame_vis, (cx, cy), angle, conf, frame.shape[1], frame.shape[0], name, is_defect

# ===================== 카메라 스레드 =====================
def camera_thread(stop_event, frame_container):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ 카메라 열기 실패")
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
    global stack_count_good, stack_count_bad

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--model", type=str, default="/home/young/Downloads/best.pt")
    args = parser.parse_args()

    model = YOLO(args.model)
    print(f"✅ YOLO 모델 로드 완료 (분류 로직: conf >= {CONF_GOOD_THRESHOLD:.2f} → 양품, 미만 → 불량품)")

    mc = CobotClass(args.port, args.baud)
    mc.power_on()
    time.sleep(1)  # 부팅 대기

    mc.send_angles([0, 0, 0, 0, 0, 0], 20)
    wait_for_robot_stop(mc)

    mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
    wait_for_robot_stop(mc)

    mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
    mc.set_gripper_mode(0)
    mc.set_electric_gripper(0)
    mc.set_gripper_value(50, 20, 1)  # 완전 열기
    print("🏠 홈 포즈 도달 및 초기화 완료")
    wait_for_robot_stop(mc)

    # ==================== 9회 반복 ====================
    for i in range(9):
        print(f"\n--- 🔁 사이클 {i+1}/9 시작 ---")

        frame_container, stop_event = {"frame": None}, threading.Event()
        cam_thread = threading.Thread(target=camera_thread, args=(stop_event, frame_container), daemon=True)
        cam_thread.start()

        detect_start = None
        detected = None
        detected_angle = None
        confirmed_is_defect = False
        detected_name = None

        while not stop_event.is_set():
            frame = frame_container.get("frame")
            if frame is None:
                continue

            frame_vis, result, angle, conf, fw, fh, name, is_defect = detect_object(model, frame)

            if result:
                cx, cy = result
                if detect_start is None:
                    detect_start = time.time()
                elif time.time() - detect_start > 4.0:
                    detected = pixel_to_robot(cx, cy, fw, fh)
                    detected_angle = angle
                    confirmed_is_defect = is_defect
                    detected_name = name
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
            print("❌ 감지 실패. 홈에서 대기 후 다음 사이클 시도.")
            mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
            wait_for_robot_stop(mc)
            continue

        # === [수정된 픽업 로직 시작] ===
        x, y, z, r, p, yaw = detected  # yaw는 POSE_HOME[5]

        # 1. (x, y) 접근 (회전 없이, 고정 Z=APPROACH_Z)
        print(f"🧭 1단계: (x, y)로 접근 이동 ({x:.1f}, {y:.1f}, Z={APPROACH_Z})")
        mc.send_coords([x, y, APPROACH_Z, r, p, yaw], 25, 1)
        wait_for_robot_stop(mc)

        # 2. J6 회전 보정 (감지된 각도만큼 상대 회전)
        angles = mc.get_angles()
        if angles:
            print(f"[DEBUG] 현재 J6 각도: {angles[5]:.1f}, 보정할 각도: {detected_angle:.1f}")
            angles[5] += detected_angle
            print(f"🧭 2단계: J6 회전 보정. 목표 각도: {angles[5]:.1f}")
            mc.send_angles(angles, 25)
            wait_for_robot_stop(mc)
            print("🧭 각도 보정 완료!")
        else:
            print("❌ 에러: 로봇 각도를 읽을 수 없습니다. 픽업을 중단합니다.")
            continue

        # 2-1. 회전 후 (x, y) 재보정 (미세 보정)
        current_coords = mc.get_coords()
        if current_coords:
            refine_coords = current_coords.copy()
            refine_coords[0] = x
            refine_coords[1] = y
            print(f"🧭 2-1단계: 회전 후 (x, y) 재보정 이동 ({x:.1f}, {y:.1f})")
            mc.send_coords(refine_coords, 15, 0)
            wait_for_robot_stop(mc)

        # 3. Z 상대 하강
        print("🧭 3단계: 현재 좌표 읽어오기 (Z 하강 준비)")
        current_coords = mc.get_coords()
        if not current_coords:
            print("❌ 에러: 로봇 좌표 읽기 실패. 하강 중단.")
            continue

        original_z = current_coords[2]
        target_z = original_z - GRAB_DELTA_Z

        down_coords = current_coords.copy()
        down_coords[2] = target_z

        print(f"🧭 3단계: Z축 상대 하강 (Z: {original_z:.1f} → {target_z:.1f})")
        mc.send_coords(down_coords, 20, 0)
        wait_for_robot_stop(mc)

        # 4. 그리퍼 2단계로 꽉 닫기
        print("🖐️ 4단계: 그리퍼 1차 닫기 (대략 위치)")
        mc.set_gripper_value(20, 40, 1)  # 박스 두께보다 약간 큰 값
        time.sleep(1.0)

        print("🖐️ 4-1단계: 그리퍼 2차 꽉 닫기")
        mc.set_gripper_value(5, 40, 1)   # 최대에 가깝게 닫기
        time.sleep(1.0)                  # 그리퍼 모터 동작 대기

        # 5. Z 상대 상승 (원래 높이로 복귀)
        print(f"🖐️ 5단계: Z축 상대 상승 (Z: {target_z:.1f} → {original_z:.1f})")
        up_coords = mc.get_coords()
        if not up_coords:
            up_coords = down_coords
        up_coords[2] = original_z
        mc.send_coords(up_coords, 25, 0)
        wait_for_robot_stop(mc)

        # === [수정된 픽업 로직 끝] ===

        mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
        wait_for_robot_stop(mc)

        # === 쌓기 로직 ===
        if confirmed_is_defect:
            z_offset = STACK_HEIGHT * stack_count_bad
            print(f"🔴 불량품({detected_name}) 쌓기 {stack_count_bad+1}단 (Z +{z_offset})")
            place_up = POSE_PLACE2_UP.copy()
            place_down = POSE_PLACE2.copy()
            place_up[2] += z_offset
            place_down[2] += z_offset
            stack_count_bad += 1
        else:
            z_offset = STACK_HEIGHT * stack_count_good
            print(f"🟢 양품({detected_name}) 쌓기 {stack_count_good+1}단 (Z +{z_offset})")
            place_up = POSE_PLACE1_UP.copy()
            place_down = POSE_PLACE1.copy()
            place_up[2] += z_offset
            place_down[2] += z_offset
            stack_count_good += 1

        mc.send_coords(place_up, DEFAULT_SPEED, 0)
        wait_for_robot_stop(mc)

        mc.send_coords(place_down, DEFAULT_SPEED, 0)
        wait_for_robot_stop(mc)

        mc.set_gripper_value(50, 20, 1)  # 박스 내려놓고 완전 열기
        time.sleep(1.5)  # 그리퍼 동작 대기

        mc.send_coords(place_up, DEFAULT_SPEED, 0)
        wait_for_robot_stop(mc)

        mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
        wait_for_robot_stop(mc)

        mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
        wait_for_robot_stop(mc)

        print(f"✅ 사이클 {i+1}/9 완료 → 홈 복귀")

    print("\n🎉 총 9회 쌓기 완료. 프로그램 종료.")
    mc.power_off()

if __name__ == "__main__":
    main()
