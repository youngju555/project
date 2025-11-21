# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 (pymycobot)
[개선 버전 v5.9 - 실시간성 확보]

📌 v5.8 대비 핵심 변경점
----------------------------------------------------
1. (지연/Lag 해결) '카메라 읽기'와 'YOLO 예측' 스레드 분리
   - [신규] camera_read_thread: 1초에 30번씩 프레임만 읽어서 Queue에 넣음 (빠름)
   - [수정] yolo_process_thread: Queue에서 프레임을 꺼내서 predict (느림)
   - (결과) YOLO가 아무리 느려도, 카메라는 30fps로 부드럽게 보임

2. (멈춤/Freeze 해결) '로봇 제어' 로직을 별도 스레드로 분리
   - [신규] robot_control_thread: main()에 있던 모든 time.sleep()과
     mc.send_coords()를 전담하는 스레드.
   - (결과) 로봇이 20초간 움직여도, main()의 cv2.imshow()는 멈추지 않음.

3. (자원 관리) Queue와 Event 객체 도입
   - frame_queue: 카메라->YOLO 실시간 프레임 전달용
   - e_robot_task_ready: YOLO->로봇 "물건 찾았으니 움직여" 신호용
   - e_robot_task_done: 로봇->YOLO "동작 끝났으니 다시 찾아" 신호용
"""

import threading
import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO
import queue  # [!!! v5.9 추가 !!!]

# ---------------------------------------------------------------------------
# 0. 로봇 클래스 불러오기
# ---------------------------------------------------------------------------
try:
    from pymycobot.mycobot320 import MyCobot320 as CobotClass
except Exception:
    from pymycobot.mycobot import MyCobot as CobotClass

# ---------------------------------------------------------------------------
# 1. 전역 변수, Lock, Event [!!! v5.9 수정 !!!]
# ---------------------------------------------------------------------------
g_target_coordinate = None      # YOLO가 계산한 로봇 좌표
g_coord_lock = threading.Lock() # 위 좌표를 안전하게 읽고 쓰기 위한 Lock
args = None                     # argparse 결과

# [v5.9] 스레드 간 통신용 Event
e_robot_task_ready = threading.Event()  # YOLO -> Robot "물건 찾았다, 출발해"
e_robot_task_done = threading.Event()   # Robot -> YOLO "작업 끝났다, 다시 찾아도 돼"
e_robot_task_done.set() # 초기 상태는 "작업 완료" (즉시 탐지 시작 가능)

# [v5.9] 스레드 간 프레임 전달용 Queue
# Queue(maxsize=1): 큐에 1개만 저장. 최신 프레임을 유지하고 나머지는 버림
frame_queue = queue.Queue(maxsize=1) 
# 디버그 및 GUI 표시용 (YOLO가 처리한 최종 프레임)
processed_frame_buffer = {"frame": None}

# ---------------------------------------------------------------------------
# 2. 로봇 기본 자세/캘리브레이션 값 (v5.8 원본)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 3. 픽셀 좌표 → 로봇 좌표 변환 (v5.8 원본)
# ---------------------------------------------------------------------------
def pixel_to_robot(cx, cy, distance_cm, frame_w, frame_h):
    pts = np.array([[[cx, cy]]], dtype=np.float32)
    undistorted_pts = cv2.undistortPoints(pts, CAMERA_MATRIX, DIST_COEFFS, P=None)
    norm_x, norm_y = undistorted_pts[0, 0]
    scale_z = distance_cm * 10.0
    x_cam = norm_x * scale_z
    y_cam = norm_y * scale_z
    
    TCP_BASE_OFFSET_X = 59.8
    TCP_BASE_OFFSET_Y = -215.9
    CAMERA_TO_TCP_OFFSET_X = 90.0 # (v5.8 원본값)
    CAMERA_TO_TCP_OFFSET_Y = 0.0
    
    robot_x = TCP_BASE_OFFSET_X + CAMERA_TO_TCP_OFFSET_X + y_cam
    robot_y = TCP_BASE_OFFSET_Y + CAMERA_TO_TCP_OFFSET_Y + x_cam
    
    TCP_BASE_OFFSET_Z = 354.6
    robot_z_ignored = TCP_BASE_OFFSET_Z - scale_z
    
    return {"x": round(robot_x, 2), "y": round(robot_y, 2), "z_debug": round(robot_z_ignored, 2)}

# ---------------------------------------------------------------------------
# 4. [신규 v5.9] 카메라 '읽기' 스레드 (초고속 영상 수급)
# ---------------------------------------------------------------------------
def camera_read_thread(stop_event, cap, frame_queue):
    """오직 cap.read()만 반복하며 Queue에 최신 프레임을 공급"""
    print("📷 카메라 '읽기' 스레드 시작")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        try:
            # 큐가 꽉 찼으면(maxsize=1), 기존 것을 버리고 새 것을 넣음 (non-blocking)
            frame_queue.put_nowait(frame) 
        except queue.Full:
            # 큐가 꽉 차서 (YOLO 처리가 밀려서) 프레임을 넣지 못할 때
            # 그냥 이 프레임은 버리고 다음 프레임을 읽으러 감
            pass
        
        time.sleep(0.01) # 약 30~50fps 유지를 위한 최소한의 sleep
    print("📷 카메라 '읽기' 스레드 종료")

# ---------------------------------------------------------------------------
# 5. [수정 v5.9] 'YOLO 처리' 스레드 (느린 두뇌)
# ---------------------------------------------------------------------------
def yolo_process_thread(stop_event, frame_queue, model):
    """Queue에서 프레임을 꺼내서 YOLO 예측만 수행 (느리게 동작)"""
    global g_target_coordinate, g_coord_lock, processed_frame_buffer
    
    print("🧠 YOLO '처리' 스레드 시작")
    stable_frames = 0
    
    while not stop_event.is_set():
        # 1. 로봇이 작업 중(e_robot_task_done이 False)이면, 탐지 안 함
        if not e_robot_task_done.is_set():
            stable_frames = 0
            time.sleep(0.1)
            continue
            
        # 2. 로봇이 쉬고 있으면, Queue에서 최신 프레임 꺼내기
        try:
            frame = frame_queue.get(timeout=0.1) # 0.1초간 기다림
        except queue.Empty:
            continue # 큐가 비었으면 다시 대기

        # 3. YOLO 예측 (가장 느린 부분)
        results = model.predict(frame, imgsz=640, conf=0.6, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # 4. GUI 표시용 프레임 저장
        processed_frame_buffer["frame"] = results[0].plot()

        # 5. 물체 감지 및 좌표 계산
        if len(boxes) > 0:
            stable_frames += 1
            if stable_frames >= 3: # 3프레임 연속 감지 시 "확정"
                x1, y1, x2, y2 = boxes[0]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                distance_cm = 30.0 # 임시 고정거리

                print(f"🎯 YOLO 객체 중심: ({cx}, {cy})")
                h, w, _ = frame.shape
                coord = pixel_to_robot(cx, cy, distance_cm, w, h)

                with g_coord_lock:
                    g_target_coordinate = coord
                
                e_robot_task_ready.set()  # 로봇 스레드에게 "출발 신호"
                e_robot_task_done.clear() # "탐지 임무 완료, 로봇 끝날 때까지 대기"
                stable_frames = 0
        else:
            stable_frames = 0
            
    print("🧠 YOLO '처리' 스레드 종료")

# ---------------------------------------------------------------------------
# 6. [신규 v5.9] '로봇 제어' 스레드 (느린 팔다리)
# ---------------------------------------------------------------------------
def robot_control_thread(stop_event, mc, dry_run):
    """로봇의 모든 움직임(sleep 포함)을 전담"""
    global g_target_coordinate, g_coord_lock
    
    print("🤖 로봇 '제어' 스레드 시작")
    
    # 1. (딱 한 번) 홈 위치로 이동
    if not dry_run and mc is not None:
        print("🤖 로봇을 홈 위치로 이동합니다...")
        mc.send_coords(POSES["Home"], DEFAULT_SPEED)
        time.sleep(3)
        print("🏠 홈 위치 도달. 탐지를 시작합니다.")
    else:
        print("🏠 [dry-run] 홈 위치 도달. 탐지를 시작합니다.")
        
    e_robot_task_done.set() # YOLO가 탐지를 시작하도록 허용

    # 2. 메인 루프 (신호 대기)
    while not stop_event.is_set():
        # e_robot_task_ready 신호가 올 때까지 무한정 대기 (Blocking)
        if not e_robot_task_ready.wait(timeout=0.5):
            continue # 0.5초마다 stop_event 체크

        # 신호가 오면, 좌표를 가져와서 전체 시퀀스 실행
        current_coord = None
        with g_coord_lock:
            if g_target_coordinate is not None:
                current_coord = g_target_coordinate.copy()
                g_target_coordinate = None
        
        if current_coord:
            print(f"🤖 인식 성공 → 로봇 이동 시작: {current_coord}")
            pick_x = current_coord["x"]
            pick_y = current_coord["y"]

            # (v5.8 원본 로직)
            GRIPPER_OFFSET_Z = 18.0
            FIXED_PICK_Z = 278.0
            APPROACH_HEIGHT = 40.0
            PICK_RX, PICK_RY, PICK_RZ = -175.33, 8.65, 86.68
            
            z_approach = FIXED_PICK_Z + APPROACH_HEIGHT
            z_grasp = FIXED_PICK_Z - GRIPPER_OFFSET_Z
            print(f"  ↳ 접근Z={z_approach:.1f}, 잡기Z={z_grasp:.1f}")

            if not dry_run and mc is not None:
                # --- v5.8 픽업 시퀀스 (총 20~25초 소요) ---
                mc.set_gripper_state(0, 80)
                time.sleep(1)
                mc.send_coords([pick_x, pick_y, z_approach, PICK_RX, PICK_RY, PICK_RZ], 25, 1)
                time.sleep(5)
                mc.send_coords([pick_x, pick_y, z_grasp, PICK_RX, PICK_RY, PICK_RZ], 15, 1)
                time.sleep(1.5)
                mc.set_gripper_state(1, 80)
                time.sleep(1.5)
                
                # (v5.8 코드 수정 - z_grasp + 80 부분)
                mc.send_coords([pick_x, pick_y, z_grasp + 80, PICK_RX, PICK_RY, PICK_RZ], 25, 1)
                time.sleep(2)
                # (z_approach로 다시 올리기 - 이 코드는 중복 같지만 원본 유지)
                # mc.send_coords([pick_x, pick_y, z_approach, PICK_RX, PICK_RY, PICK_RZ], 25, 1)
                # time.sleep(2)
                
                mc.send_coords(POSES["Clear_Air_A"], DEFAULT_SPEED, 1)
                time.sleep(3)
                mc.send_coords(POSES["Place_B"], DEFAULT_SPEED, 1)
                time.sleep(3)
                mc.set_gripper_state(0, 80)
                time.sleep(1.5)
                mc.send_coords(POSES["Home"], DEFAULT_SPEED)
                time.sleep(3)
                print("✅ 1회 피킹 완료")
            else:
                print("  [dry-run] 로봇 없이 동작 흐름만 실행")
                time.sleep(5) # 시뮬레이션 대기

            # 작업이 끝났음을 알림
            e_robot_task_ready.clear() # "출발 신호" 끄기
            e_robot_task_done.set()  # YOLO에게 "다시 탐지 시작" 신호
            
    print("🤖 로봇 '제어' 스레드 종료")

# ---------------------------------------------------------------------------
# 7. 메인 루프 (GUI 담당)
# ---------------------------------------------------------------------------
def main():
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=str, default="best.pt")
    args = parser.parse_args()

    print(f"🧠 YOLOv8 모델('{args.model}') 로드 중...")
    try:
        model = YOLO(args.model, task="detect")
        print("✅ YOLO 모델 로드 성공")
    except Exception as e:
        print(f"❌ YOLO 모델 로드 실패: {e}")
        return
        
    stop_event = threading.Event()
    mc = None
    cap = None
    
    threads = [] # 실행된 스레드들을 담을 리스트

    try:
        # 1) 로봇 초기화 (v5.8 원본)
        if not args.dry_run:
            try:
                mc = CobotClass(args.port, args.baud)
                time.sleep(0.5)
                mc.power_on()
                print("🔌 로봇 Power ON 완료")
                mc.set_gripper_state(0, 80)
                time.sleep(1)
            except Exception as e:
                print(f"❌ 로봇 연결 실패: {e}")
                mc = None
                args.dry_run = True
        else:
            print("🟡 dry-run 모드로 시작")

        # 2) 카메라 초기화 (v5.8 원본)
        print(f"📷 메인: 카메라 {args.camera}번 열기 시도...")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"⚠️ {args.camera}번 카메라 실패 → 0번으로 재시도")
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("camera open failed")
        print("✅ 메인: 카메라 연결 성공")

        # 3) [!!! v5.9 !!!] 3개의 스레드 시작
        
        # Thread 1: 카메라 읽기
        t_cam = threading.Thread(
            target=camera_read_thread, 
            args=(stop_event, cap, frame_queue), 
            daemon=True
        )
        t_cam.start()
        threads.append(t_cam)

        # Thread 2: YOLO 처리
        t_yolo = threading.Thread(
            target=yolo_process_thread, 
            args=(stop_event, frame_queue, model), 
            daemon=True
        )
        t_yolo.start()
        threads.append(t_yolo)

        # Thread 3: 로봇 제어
        t_robot = threading.Thread(
            target=robot_control_thread, 
            args=(stop_event, mc, args.dry_run), 
            daemon=True
        )
        t_robot.start()
        threads.append(t_robot)

        print("✅ 메인 루프 시작 (GUI 표시 담당, q로 종료)")
        
        # 4) 메인 루프 (GUI만 담당)
        while not stop_event.is_set():
            # YOLO가 처리한 최종 결과 프레임을 가져옴
            frame = processed_frame_buffer.get("frame")
            
            if frame is None:
                # YOLO가 아직 첫 프레임을 처리 못했으면 Queue에서 원본이라도 가져옴
                try:
                    frame = frame_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.01)
                    continue

            cv2.imshow("Camera View", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
                
            time.sleep(0.01) # GUI 루프도 약 30~50fps로 실행

    except Exception as e:
        print(f"🚨 메인 루프에서 에러 발생: {e}")
    finally:
        # 7) 종료 처리
        print("🛑 종료 신호 감지... 모든 스레드 정리 중...")
        stop_event.set()
        
        for t in threads:
            t.join(timeout=1.0) # 스레드가 1초 안에 끝나길 기다림
            
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        if mc:
            mc.power_off()
        print("🔒 프로그램 종료")


if __name__ == "__main__":
    main()