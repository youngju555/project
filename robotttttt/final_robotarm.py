# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 (pymycobot)
[개선 버전 v6.1] — 최종 버그 수정
----------------------------------------------------
- (수정) main -> camera_thread로 'model', 'cam_id', 'dry_run' 인수가
         정확히 전달되도록 치명적인 버그 수정
- (수정) 'args.color' (불필요한 인수) 제거
"""

import threading
import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO

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
args = None # [수정] args를 전역으로 (스레드에서 dry_run 참조용)

# ===============================================================
# 2️⃣ 기본 설정값 (사용자 설정 유지)
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
# 3️⃣ 픽셀 → 로봇 좌표 변환 (사용자 설정 유지)
# ===============================================================
def pixel_to_robot(cx, cy, distance_cm, frame_w, frame_h):
    pts = np.array([[[cx, cy]]], dtype=np.float32)
    undistorted_pts = cv2.undistortPoints(pts, CAMERA_MATRIX, DIST_COEFFS, P=None)
    norm_x, norm_y = undistorted_pts[0, 0]

    scale_z = distance_cm * 10.0  # cm → mm
    x_cam = norm_x * scale_z
    y_cam = norm_y * scale_z

    # === TCP 기준 (탐색 자세, Home) ===
    TCP_BASE_OFFSET_X = 59.8
    TCP_BASE_OFFSET_Y = -215.9
    # === 카메라 ↔ TCP 옵셋 (실측) ===
    CAMERA_TO_TCP_OFFSET_X = 100.0
    CAMERA_TO_TCP_OFFSET_Y = 0.0
    
    # === 좌표 변환 (90도 회전 적용됨) ===
    robot_x = TCP_BASE_OFFSET_X + CAMERA_TO_TCP_OFFSET_X + y_cam
    robot_y = TCP_BASE_OFFSET_Y + CAMERA_TO_TCP_OFFSET_Y + x_cam
    
    return {"x": round(robot_x, 2), "y": round(robot_y, 2)}

# ===============================================================
# 4️⃣ YOLO 큐브 검출 (사용자 설정 유지)
# ===============================================================
def detect_yolo_cube(frame, model):
    detected_info = []
    FIXED_DISTANCE_CM = 30.0
    
    results = model(frame, verbose=False)
    
    if results and results[0].boxes:
        for box in results[0].boxes:
            conf = float(box.conf[0])
            
            # 양품 필터 (93%)
            if conf > 0.93:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                distance = FIXED_DISTANCE_CM
                
                detected_info.append(("Cube", (cx, cy), distance))

                # 시각화
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Good Cube: {conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
                
                break 
    return frame, detected_info

# ===============================================================
# 5️⃣ 카메라 스레드 [수정됨]
# ===============================================================
# [수정] 인자 리스트 변경 (model, cam_id, dry_run 추가)
def camera_capture_thread(stop_event, frame_container, model, mc=None, cam_id=1, dry_run=False):
    global picking_done, g_target_coordinate, g_coord_lock

    print(f"📷 카메라 {cam_id}번 초기화 중...")
    cap = cv2.VideoCapture(cam_id) # [수정] 하드코딩된 '1' 대신 'cam_id' 사용
    if not cap.isOpened():
        print(f"⚠️ {cam_id}번 실패 → 0번 시도")
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라 연결 실패"); return

    # (워밍업 로직)
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

        processed_frame, detected = detect_yolo_cube(frame, model) # YOLO 사용
        frame_container["frame"] = processed_frame

        # [안정성 로직]
        # [수정] mc는 주입받으므로 cam_thread.mc로 참조
        if (getattr(cam_thread, 'mc', None) is not None or dry_run) and not picking_done:
            if detected:
                stable_frames += 1
                if stable_frames >= 3: 
                    obj_name, (cx, cy), dist = detected[0]
                    print(f"🎯 안정 검출: {obj_name} ({cx},{cy})")
                    
                    h, w, _ = frame.shape
                    coord_xy = pixel_to_robot(cx, cy, dist, w, h)
                    
                    with g_coord_lock:
                        g_target_coordinate = coord_xy
                    picking_done = True
                    stable_frames = 0
            else:
                stable_frames = 0 

        time.sleep(0.05)

    cap.release()
    print("📷 카메라 스레드 종료")

# ===============================================================
# 6️⃣ 메인 제어 루프 [수정됨]
# ===============================================================
def main():
    global g_target_coordinate, g_coord_lock, picking_done, args

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--speed", type=int, default=20)
    # [수정] color 인자 제거
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    # [수정] model 경로 인자 추가
    parser.add_argument("--model", type=str, default="best.pt", help="YOLO 모델 경로")
    args = parser.parse_args()

    print(f"🧠 YOLOv8 모델('{args.model}') 로드 중...")
    try:
        model = YOLO(args.model) # [수정] args.model 사용
        print("✅ 모델 로드 성공.")
    except Exception as e:
        print(f"❌ '{args.model}' 모델 로드 실패: {e}")
        return

    # === [필수 수정] 실제 로봇 세팅 (사용자 설정 유지) ===
    GRIPPER_OFFSET_Z = 18.0
    FIXED_PICK_Z = 278.0  
    APPROACH_HEIGHT = 40.0 # (사용자 v5.4에서 35.0 -> 40.0으로 복원)
    PICK_RX, PICK_RY, PICK_RZ = -175.33, 8.65, 86.68

    frame_container = {"frame": None}
    stop_event = threading.Event()
    mc = None

    # --- 병렬 초기화 ---
    cam_thread = threading.Thread(
        target=camera_capture_thread,
        # [!!! 여기!! 핵심 버그 수정 !!!]
        args=(
            stop_event, 
            frame_container, 
            model,            # 1. model 객체 전달
            mc,               # 2. mc (None) 전달
            args.camera,      # 3. cam_id 전달
            args.dry_run      # 4. dry_run 상태 전달
        ),
        daemon=True
    )
    cam_thread.start()

    if not args.dry_run:
        try:
            mc = CobotClass(args.port, args.baud); time.sleep(0.5)
            mc.power_on(); print("🔌 Power ON 완료")
            mc.set_gripper_state(0, 80); time.sleep(1) # 그리퍼 열기
            mc.send_coords(POSES["Home"], args.speed); time.sleep(3) # Home 이동
            print("🏠 홈 위치 도달. (카메라 스레드 준비 완료 대기)")
        except Exception as e:
            print(f"❌ 로봇 연결 실패: {e}"); mc = None
            args.dry_run = True # [수정] 로봇 연결 실패 시 dry-run으로 강제 전환
    else:
        print("✅ 'dry-run' 모드로 시작.")
    
    # 스레드에 로봇 객체(mc) 주입
    cam_thread.mc = mc
    
    print("✅ 메인 루프 시작 (q로 종료)")

    try:
        while not stop_event.is_set():
            frame = frame_container.get("frame")
            if frame is not None:
                cv2.imshow("Camera View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set(); break

            current_coord = None
            if picking_done: 
                with g_coord_lock:
                    if g_target_coordinate is not None:
                        current_coord = g_target_coordinate.copy()
                        g_target_coordinate = None
                
                if current_coord:
                    print(f"🤖 [양품 인식] → 로봇 이동 시작 (X,Y): {current_coord}")
                    pick_x = current_coord["x"]
                    pick_y = current_coord["y"]
                    
                    z_approach = FIXED_PICK_Z + APPROACH_HEIGHT
                    z_grasp = FIXED_PICK_Z - GRIPPER_OFFSET_Z
                    print(f"  → 고정Z사용: 접근Z={z_approach:.2f}, 잡기Z={z_grasp:.2f}")

                    if not args.dry_run and mc is not None:
                        # --- 로봇 동작 시퀀스 (그리퍼 API 사용) ---
                        mc.set_gripper_value(60, 80,1) # Open
                        time.sleep(1)
                        mc.send_coords([pick_x, pick_y, z_approach, PICK_RX, PICK_RY, PICK_RZ], 25, 1)
                        time.sleep(2)
                        mc.send_coords([pick_x, pick_y, z_grasp, PICK_RX, PICK_RY, PICK_RZ], 15, 1)
                        time.sleep(1.5)
                        mc.set_gripper_state(1, 80) # Close
                        time.sleep(1.5)
                        mc.send_coords([pick_x, pick_y, z_approach, PICK_RX, PICK_RY, PICK_RZ], 25, 1)
                        time.sleep(2)
                        mc.send_coords(POSES["Clear_Air_A"], args.speed, 1)
                        time.sleep(3)
                        mc.send_coords(POSES["Place_B"], args.speed, 1)
                        time.sleep(3)
                        mc.set_gripper_value(60, 80,1) # Open
                        time.sleep(1.5)
                        mc.send_coords(POSES["Home"], args.speed)
                        time.sleep(3)
                        print("✅ 1회 피킹 완료")
                    else:
                        print("   [dry-run] 로봇 이동 시뮬레이션 완료.")
                        time.sleep(5) 

                    picking_done = False # 다음 탐색을 위해 초기화

    finally:
        stop_event.set()
        cam_thread.join()
        cv2.destroyAllWindows()
        if mc: mc.power_off()
        print("🔒 종료")


if __name__ == "__main__":
    main()