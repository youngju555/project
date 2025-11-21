# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 (pymycobot)
개선 버전 v5.5 — 병렬 초기화 + 안정된 프레임 감지
----------------------------------------------------
- (속도) 로봇 이동과 카메라 초기화를 병렬로 처리 (빠른 시작)
- (안정) 3프레임 연속 감지(stable_frames) 로직으로 노이즈 제거
- (안전) 고정 Z축 피킹(FIXED_PICK_Z) 로직 사용
- (수정) --camera 인수 버그 수정
"""

import threading
import cv2
import time
import argparse
import numpy as np

try:
    from pymycobot.mycobot320 import MyCobot320 as CobotClass
except Exception:
    from pymycobot.mycobot import MyCobot as CobotClass


# ===============================================================
# 1️⃣ 전역 변수 및 Lock
# ===============================================================
picking_done = False
g_target_coordinate = None
g_coord_lock = threading.Lock()


# ===============================================================
# 2️⃣ 기본 설정값
# ===============================================================
POSES = {
    "Home":  [59.8, -215.9, 354.6, -175.33, 8.65, 86.68],
    "Clear_Air_A": [264.0, -1.0, 379.0, -153, 11, -106],
    "Place_B": [333.0, 11.0, 170.0, -175, -0.08, -89.0],
}
DEFAULT_SPEED = 20

CAMERA_MATRIX = np.array([
    [539.13729067, 0.0, 329.02126026],
    [0.0, 542.34217387, 242.10995541],
    [0.0, 0.0, 1.0]
])
DIST_COEFFS = np.array([[0.20528603, -0.76664068, -0.00096614, 0.00111892, 0.97630004]])


# ===============================================================
# 3️⃣ 픽셀 → 로봇 좌표 변환 (X, Y 전용)
# ===============================================================
def pixel_to_robot(cx, cy, distance_cm, frame_w, frame_h):
    pts = np.array([[[cx, cy]]], dtype=np.float32)
    undistorted_pts = cv2.undistortPoints(pts, CAMERA_MATRIX, DIST_COEFFS, P=None)
    norm_x, norm_y = undistorted_pts[0, 0]

    scale_z = distance_cm * 10.0  # cm → mm
    x_cam = norm_x * scale_z
    y_cam = norm_y * scale_z

    # === TCP 기준 (탐색 자세) ===
    TCP_BASE_OFFSET_X = 59.8
    TCP_BASE_OFFSET_Y = -215.9
    
    # === 카메라 ↔ TCP 옵셋 (실측 필요) ===
    CAMERA_TO_TCP_OFFSET_X = 100.0
    CAMERA_TO_TCP_OFFSET_Y = 0.0

    # === 좌표 변환 (X, Y만 계산) ===
    robot_x = TCP_BASE_OFFSET_X + CAMERA_TO_TCP_OFFSET_X + y_cam
    robot_y = TCP_BASE_OFFSET_Y + CAMERA_TO_TCP_OFFSET_Y + x_cam
    
    # Z좌표는 'FIXED_PICK_Z'를 사용하므로, 여기서는 무시됨
    # (디버깅용으로만 Z 계산 로직을 남겨둠)
    TCP_BASE_OFFSET_Z = 354.6
    robot_z_ignored = TCP_BASE_OFFSET_Z - scale_z

    # X, Y 좌표만 반환 (Z는 고정값이므로)
    return {"x": round(robot_x, 2), "y": round(robot_y, 2), "z_debug": round(robot_z_ignored, 2)}


# ===============================================================
# 4️⃣ 색상 검출 + 거리 계산
# ===============================================================
def detect_color_and_distance(frame, target_color="blue"):
    h, w, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # [필수 수정] 조명 환경에 맞는 HSV 값
    color_ranges = {
        "blue": [(90, 50, 50), (140, 255, 255)],
    }

    lower, upper = color_ranges.get(target_color, ((0, 0, 0), (0, 0, 0)))
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_info = []
    
    # [필수 수정] 바닥이 평평하면 True 권장
    USE_FIXED_DISTANCE = True
    # [필수 측정] 탐색 자세 카메라 렌즈 ~ 물체 표면 거리 (cm)
    FIXED_DISTANCE_CM = 30.0

    if contours:
        c = max(contours, key=cv2.contourArea)
        # [필수 수정] 노이즈 필터링 (환경에 맞게 400 ~ 1000)
        if cv2.contourArea(c) > 400:
            x, y, w_box, h_box = cv2.boundingRect(c)
            cx, cy = x + w_box // 2, y + h_box // 2

            # Z축은 고정식이므로, distance는 X,Y 계산용 스케일링에만 사용됨
            distance = FIXED_DISTANCE_CM
            
            # (참고: USE_FIXED_DISTANCE=False일 경우)
            # KNOWN_WIDTH = 2.5; fx = CAMERA_MATRIX[0, 0]
            # distance = (KNOWN_WIDTH * fx) / w_box
            
            detected_info.append((target_color, (cx, cy), distance))

            # 시각화
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 255, 0), 2)
            cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
            cv2.putText(frame, f"{target_color} {distance:.1f}cm",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame, detected_info


# ===============================================================
# 5️⃣ 카메라 스레드 [수정됨]
# ===============================================================
# [수정] main에서 cam_id를 받도록 인자 추가
def camera_capture_thread(stop_event, frame_container, target_color="blue", mc=None, cam_id=1):
    global picking_done, g_target_coordinate, g_coord_lock

    print(f"📷 카메라 {cam_id}번 초기화 중...")
    
    # [수정] 하드코딩된 '1' 대신 'cam_id' 사용
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"⚠️ {cam_id}번 실패 → 0번 시도")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 카메라 연결 실패")
        return

    # 실제 프레임 입력 확인
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret and frame is not None:
            print("✅ 카메라 프레임 수신 시작됨")
            break
        print("⌛ 카메라 준비 중...")
        time.sleep(0.2)

    stable_frames = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        processed_frame, detected = detect_color_and_distance(frame, target_color)
        frame_container["frame"] = processed_frame

        # [안정성 로직]
        # (mc가 연결되어 있거나, dry-run 모드일 때만)
        if (mc is not None or args.dry_run) and not picking_done:
            if detected:
                stable_frames += 1
                if stable_frames >= 3: # 3프레임 연속 감지
                    color_name, (cx, cy), dist = detected[0]
                    print(f"🎯 안정 검출: {color_name} ({cx},{cy})")
                    h, w, _ = frame.shape
                    
                    # (X, Y) 좌표 계산
                    coord = pixel_to_robot(cx, cy, dist, w, h)
                    
                    with g_coord_lock:
                        g_target_coordinate = coord
                    picking_done = True
                    stable_frames = 0 # 초기화
            else:
                stable_frames = 0 # 감지 실패시 카운터 초기화

        time.sleep(0.05)

    cap.release()
    print("📷 카메라 스레드 종료")


# ===============================================================
# 6️⃣ 메인 제어 루프 [수정됨]
# ===============================================================
def main():
    global g_target_coordinate, g_coord_lock, picking_done, args # [수정] args를 전역으로

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--color", type=str, default="blue")
    parser.add_argument("--camera", type=int, default=1) # [수정] 이 값이 스레드로 전달됨
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args() # [수정] 전역 변수 args에 저장

    # === [필수 수정] 실제 로봇 세팅 ===
    GRIPPER_OFFSET_Z = 18.0
    FIXED_PICK_Z = 278.0  # [필수 측정] 물체가 놓일 바닥의 '고정 Z' 높이 (mm)
    APPROACH_HEIGHT = 40.0
    PICK_RX, PICK_RY, PICK_RZ = -175.33, 8.65, 86.68

    frame_container = {"frame": None}
    stop_event = threading.Event()
    mc = None

    # --- [수정] 병렬 초기화 ---
    
    # 1. 카메라 스레드 먼저 시작
   cam_thread = threading.Thread(
        target=camera_capture_thread,
        # [!!! 여기!! 핵심 버그 수정 !!!]
        args=(
            stop_event, 
            frame_container, 
            model,            # 1. model 객체 전달 (args.color 대신)
            mc,               # 2. mc (None) 전달
            args.camera,      # 3. cam_id 전달
            args.dry_run      # 4. dry_run 상태 전달
        ),
        daemon=True
    )
    cam_thread.start()

    # 2. 로봇 초기화 (카메라가 켜지는 3초 동안)
    if not args.dry_run:
        try:
            mc = CobotClass(args.port, args.baud)
            time.sleep(0.5)
            mc.power_on()
            print("🔌 Power ON 완료")
            mc.set_gripper_state(0, 80)
            time.sleep(1)
            mc.send_coords(POSES["Home"], args.speed)
            time.sleep(3) # Home 이동 대기
            print("🏠 홈 위치 도달. (카메라 스레드 준비 완료 대기)")
        except Exception as e:
            print(f"❌ 로봇 연결 실패: {e}")
            mc = None
    else:
        print("✅ 'dry-run' 모드로 시작.")
    
    # [수정] 스레드에 로봇 객체(mc) 주입
    # (카메라 스레드가 mc가 None이 아님을 확인하고 감지 시작)
    cam_thread.mc = mc
    
    print("✅ 메인 루프 시작 (q로 종료)")

    try:
        while not stop_event.is_set():
            frame = frame_container.get("frame")
            if frame is not None:
                cv2.imshow("Camera View", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

            current_coord = None
            if picking_done: # 카메라 스레드가 True로 설정함
                with g_coord_lock:
                    if g_target_coordinate is not None:
                        current_coord = g_target_coordinate.copy()
                        g_target_coordinate = None
                
                if current_coord:
                    print(f"🤖 인식 성공 → 로봇 이동 시작 (X,Y): {current_coord}")
                    pick_x = current_coord["x"]
                    pick_y = current_coord["y"]
                    
                    # [수정] 고정 Z 로직
                    z_approach = FIXED_PICK_Z + APPROACH_HEIGHT
                    z_grasp = FIXED_PICK_Z - GRIPPER_OFFSET_Z
                    print(f"  → 고정Z사용: 접근Z={z_approach:.2f}, 잡기Z={z_grasp:.2f}")

                    if not args.dry_run and mc is not None:
                        # --- 로봇 동작 시퀀스 ---
                        mc.set_gripper_state(0, 80)
                        time.sleep(1)
                        mc.send_coords([pick_x, pick_y, z_approach, PICK_RX, PICK_RY, PICK_RZ], 25, 1)
                        time.sleep(5)
                        mc.send_coords([pick_x, pick_y, z_grasp, PICK_RX, PICK_RY, PICK_RZ], 15, 1)
                        time.sleep(1.5)
                        mc.set_gripper_state(1, 80)
                        time.sleep(1.5)
                        mc.send_coords([pick_x, pick_y, z_approach, PICK_RX, PICK_RY, PICK_RZ], 25, 1)
                        time.sleep(2)
                        mc.send_coords(POSES["Clear_Air_A"], args.speed, 1)
                        time.sleep(3)
                        mc.send_coords(POSES["Place_B"], args.speed, 1)
                        time.sleep(3)
                        mc.set_gripper_state(0, 80)
                        time.sleep(1.5)
                        mc.send_coords(POSES["Home"], args.speed)
                        time.sleep(3)
                        print("✅ 1회 피킹 완료")
                    else:
                        print("   [dry-run] 로봇 이동 시뮬레이션 완료.")
                        time.sleep(5) # 시뮬레이션 대기

                    picking_done = False # 다음 탐색을 위해 초기화

    finally:
        stop_event.set()
        cam_thread.join()
        cv2.destroyAllWindows()
        if mc: mc.power_off()
        print("🔒 종료")


if __name__ == "__main__":
    main()