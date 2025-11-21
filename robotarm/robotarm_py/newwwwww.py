# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 (pymycobot)
[개선 버전 v5.10 - 다중 클래스 분류]

📌 v5.9 대비 핵심 변경점
----------------------------------------------------
1. (로직) YOLO가 클래스 ID(색상)를 감지하여 로봇 스레드로 전달
   - [수정] g_target_object: (x, y, z) 좌표뿐만 아니라 "class_id"도 함께 저장

2. (로직) 로봇 스레드가 class_id에 따라 '다른' 위치에 물체를 내려놓음
   - (1=Blue -> Box1), (2=Red -> Box2), (3=Yellow -> Box3)

3. (좌표) 사용자 요청 POSES 딕셔너리 전체 반영
   - 기존 Clear_Air_A, Place_B 대신 Box1_up, Box1 등으로 대체
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
# 1. 전역 변수, Lock, Event [!!! v5.10 수정 !!!]
# ---------------------------------------------------------------------------
g_target_object = None          # [v5.10] YOLO가 계산한 로봇 좌표 + 클래스 ID
g_coord_lock = threading.Lock() # 위 좌표를 안전하게 읽고 쓰기 위한 Lock
args = None                     # argparse 결과

# [v5.9] 스레드 간 통신용 Event
e_robot_task_ready = threading.Event()  # YOLO -> Robot "물건 찾았다, 출발해"
e_robot_task_done = threading.Event()   # Robot -> YOLO "작업 끝났다, 다시 찾아도 돼"
# e_robot_task_done.set() # 초기 상태는 "작업 완료" (즉시 탐지 시작 가능)

# [v5.9] 스레드 간 프레임 전달용 Queue
frame_queue = queue.Queue(maxsize=1) 
# 디버그 및 GUI 표시용 (YOLO가 처리한 최종 프레임)
processed_frame_buffer = {"frame": None}

# ---------------------------------------------------------------------------
# 2. 로봇 기본 자세/캘리브레이션 값 [!!! v5.10 수정 !!!]
# ---------------------------------------------------------------------------
# [v5.10] 사용자가 요청한 POSES 딕셔너리로 전체 교체
POSES = {
    "Home":  [59.8, -215.9, 354.6, -175.33, 8.65, 86.68],  # 시작/대기 위치
    "Place": [105.8, -65.0, 483.4, -116.46, 4.87, -78.69],  # (사용자 정의 - 현재 로직에선 미사용)
    "Box1": [291.3, 210.0, 200, -172.57, -1.46, -87.15],  # 1. 파란색 놓는 곳
    "Box2": [333.4, 11.7, 200, -175.19, -0.08, -89.53],  # 2. 빨간색 놓는 곳
    "Box3": [319.9, -169.5, 200, -172.32, -2.86, -87.15],  # 3. 노란색 놓는 곳
    "Box1_up": [229.8, 132.6, 386.4, -147.34, 9.15, -74.66],  # Box1 접근(위)
    "Box2_up": [264.0, -1.3, 379.0, -153.71, 11.7, -106.33], # Box2 접근(위)
    "Box3_up": [228.0, -203.0, 362.8, -146.13, 15.2, -149.53], # Box3 접근(위)
}

DEFAULT_SPEED = 20
CAMERA_MATRIX = np.array([
    [539.13729067, 0.0, 329.02126026],
    [0.0, 542.34217387, 242.10995541],
    [0.0, 0.0, 1.0]
])
DIST_COEFFS = np.array([[0.20528603, -0.76664068, -0.00096614, 0.00111892, 0.97630004]])

# ---------------------------------------------------------------------------
# 3. 픽셀 좌표 → 로봇 좌표 변환 (v5.9 원본)
# ---------------------------------------------------------------------------
def pixel_to_robot(cx, cy, distance_cm, frame_w, frame_h):
    # (v5.9와 동일)
    pts = np.array([[[cx, cy]]], dtype=np.float32)
    undistorted_pts = cv2.undistortPoints(pts, CAMERA_MATRIX, DIST_COEFFS, P=None)
    norm_x, norm_y = undistorted_pts[0, 0]
    scale_z = distance_cm * 10.0
    x_cam = norm_x * scale_z
    y_cam = norm_y * scale_z
    
    TCP_BASE_OFFSET_X = 59.8
    TCP_BASE_OFFSET_Y = -215.9
    CAMERA_TO_TCP_OFFSET_X = 90.0 
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
    # (v5.9와 동일)
    print("📷 카메라 '읽기' 스레드 시작")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        try:
            frame_queue.put_nowait(frame) 
        except queue.Full:
            pass
        
        time.sleep(0.01) 
    print("📷 카메라 '읽기' 스레드 종료")

# ---------------------------------------------------------------------------
# 5. [수정 v5.10] 'YOLO 처리' 스레드 (느린 두뇌)
# ---------------------------------------------------------------------------
def yolo_process_thread(stop_event, frame_queue, model):
    """Queue에서 프레임을 꺼내서 YOLO 예측만 수행 (느리게 동작)"""
    global g_target_object, g_coord_lock, processed_frame_buffer
    
    print("🧠 YOLO '처리' 스레드 시작")
    stable_frames = 0
    
    # debug: 모델 클래스 이름 출력 (안 해두었다면 메인에서 이미 출력)
    print("YOLO classes:", model.names)

    while not stop_event.is_set():
        if not e_robot_task_done.is_set():
            stable_frames = 0
            time.sleep(0.1)
            continue

        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # --- convert BGR -> RGB before feeding model ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # use lower conf threshold for better recall during debug
        try:
            results = model.predict(frame_rgb, imgsz=640, conf=0.25, verbose=False)
        except Exception as e:
            print("YOLO predict error:", e)
            continue

        # boxes, classes
        if len(results) == 0:
            # no results? skip
            processed_frame_buffer["frame"] = frame  # raw BGR
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else np.array([])
        classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else np.array([])

        # results[0].plot() likely returns RGB; convert to BGR for cv2.imshow
        annotated = results[0].plot()
        if annotated is not None:
            # some versions return RGB, some BGR — ensure BGR for imshow
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR) if annotated.shape[2] == 3 else annotated
            processed_frame_buffer["frame"] = annotated_bgr
        else:
            processed_frame_buffer["frame"] = frame

        # debug: print how many boxes
        # print("debug boxes count:", len(boxes))

        if len(boxes) > 0:
            stable_frames += 1
            if stable_frames >= 3:
                x1, y1, x2, y2 = boxes[0]
                class_id = int(classes[0]) if len(classes) > 0 else -1

                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                distance_cm = 21.0

                print(f"🎯 YOLO 객체 중심: ({cx}, {cy}), 클래스 ID: {class_id}")
                h, w, _ = frame.shape
                coord = pixel_to_robot(cx, cy, distance_cm, w, h)

                with g_coord_lock:
                    g_target_object = {"coord": coord, "class_id": class_id}

                e_robot_task_ready.set()
                e_robot_task_done.clear()
                stable_frames = 0
        else:
            stable_frames = 0

    print("🧠 YOLO '처리' 스레드 종료")


# ---------------------------------------------------------------------------
# 6. [수정 v5.10] '로봇 제어' 스레드 (느린 팔다리)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 6. 로봇 제어 스레드 [안정화 + 안전한 Z축 적용]
# ---------------------------------------------------------------------------
def robot_control_thread(stop_event, mc, dry_run):
    global g_target_object, g_coord_lock

    print("🤖 로봇 '제어' 스레드 시작")

    # ✅ [v5.10a] 초기 상태: 탐지 금지
    e_robot_task_ready.clear()
    e_robot_task_done.clear()

    # --- 홈으로 이동 ---
    if not dry_run and mc is not None:
        print("🏠 홈 위치로 이동 중...")
        mc.send_coords(POSES["Home"], DEFAULT_SPEED, 1)
        time.sleep(4)
        print("✅ 홈 위치 도달!")
    else:
        print("🟡 [dry-run] 홈 위치 도달 (시뮬레이션).")

    # ✅ [v5.10a] 이제 YOLO 탐지 허용
    e_robot_task_done.set()
    print("🔓 YOLO 탐지 허용됨!")

    # --- 메인 루프 ---
    while not stop_event.is_set():
        if not e_robot_task_ready.wait(timeout=0.5):
            continue

        with g_coord_lock:
            current_task = g_target_object.copy() if g_target_object else None
            g_target_object = None

        e_robot_task_ready.clear() # 신호 받았으니 끔

        if not current_task:
            e_robot_task_done.set() # 🔓 (중요) 잡을게 없으면 즉시 YOLO 재허용
            continue

        # --- [v5.10a] 로직 시작 (탐지 금지) ---
        # (참고: dry-run이 아니면 e_robot_task_done.clear()가 아래에 있음)
        
        coord = current_task["coord"]
        class_id = current_task["class_id"]
        print(f"🤖 작업 시작: {coord}, class={class_id}")

        # [v8.0 로직 참고] Z축 각도(RPY)는 Home과 동일하게 고정
        PICK_RX, PICK_RY, PICK_RZ = -175.33, 8.65, 86.68 

        if class_id == 0:
            place_pose_name, approach_pose_name = "Box1", "Box1_up"
        elif class_id == 1:
            place_pose_name, approach_pose_name = "Box2", "Box2_up"
        else: # 2 또는 기타
            place_pose_name, approach_pose_name = "Box3", "Box3_up"
            
        place_pose = POSES[place_pose_name]
        approach_pose = POSES[approach_pose_name]


        # --- ⬇️ [핵심 수정] v8.0의 Z축 높이를 참고하여 파라미터 재설정 ---
        Z_GRASP = 300.0          # 1. 물체를 잡을 Z 높이 (v8.0의 260+40)
        Z_APPROACH_OFFSET = 60.0 # 2. 접근/후퇴 시 추가 높이 (v8.0의 360-300)
        Z_APPROACH = Z_GRASP + Z_APPROACH_OFFSET # 3. 최종 접근/후퇴 높이 (360.0)
        
        print(f"  ↳ [안전 Z 적용] 접근/후퇴 Z={Z_APPROACH:.1f}, 잡기 Z={Z_GRASP:.1f}")
        # --- ⬆️ [핵심 수정] 완료 ---


        if not dry_run and mc:
            # 🚫 [v5.10a] 로봇이 실제로 움직이기 직전에 탐지 금지
            e_robot_task_done.clear()

            # 1) 그리퍼 열기
            mc.set_gripper_state(0, 80)
            time.sleep(1)
            
            # 2) [HIGH] 물체 '위'의 안전한 높이로 먼저 이동 (Z_APPROACH)
            mc.send_coords(
                [coord["x"], coord["y"], Z_APPROACH, 
                PICK_RX, PICK_RY, PICK_RZ], 
                25, 1
            )
            time.sleep(4)
            
            # 3) [LOW] 잡을 높이로 내려가기 (Z_GRASP)
            mc.send_coords(
                [coord["x"], coord["y"], Z_GRASP, 
                PICK_RX, PICK_RY, PICK_RZ], 
                15, 1
            )
            time.sleep(2)
            
            # 4) 그리퍼 닫기
            mc.set_gripper_state(1, 80)
            time.sleep(2)
            
            # 5) [HIGH] 다시 안전한 높이로 들어올리기 (Z_APPROACH)
            mc.send_coords(
                [coord["x"], coord["y"], Z_APPROACH, 
                PICK_RX, PICK_RY, PICK_RZ], 
                25, 1
            )
            time.sleep(3)

            # 6) 상자 위치로 이동
            mc.send_coords(approach_pose, DEFAULT_SPEED, 1)
            time.sleep(4)
            mc.send_coords(place_pose, DEFAULT_SPEED, 1)
            time.sleep(3)
            
            # 7) 내려놓기
            mc.set_gripper_state(0, 80)
            time.sleep(1.5)
            
            # 8) 다시 홈으로 복귀
            mc.send_coords(POSES["Home"], DEFAULT_SPEED, 1)
            time.sleep(4)
            print("✅ 1회 피킹 완료")

        else:
            print("🟡 [dry-run] High-Low-High 피킹 시뮬레이션 완료")
            time.sleep(5)

        # 🔓 [v5.10a] 모든 작업 완료 후 다시 YOLO 허용
        e_robot_task_done.set()
        print("🔓 YOLO 탐지 허용됨!")

    print("🤖 로봇 '제어' 스레드 종료")

# ---------------------------------------------------------------------------
# 7. 메인 루프 (GUI 담당)
# ---------------------------------------------------------------------------
def main():
    # (v5.9와 동일)
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=str, default="/home/young/Downloads/best.pt")
    args = parser.parse_args()

    print(f"🧠 YOLOv8 모델('{args.model}') 로드 중...")
    try:
        model = YOLO(args.model, task="detect")
        print("Loaded model names:", model.names) 
        print("✅ YOLO 모델 로드 성공")
    except Exception as e:
        print(f"❌ YOLO 모델 로드 실패: {e}")
        return
        
    stop_event = threading.Event()
    mc = None
    cap = None
    
    threads = [] 

    try:
        # 1) 로봇 초기화 (v5.9 원본)
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

        # 2) 카메라 초기화 (v5.9 원본)
        print(f"📷 메인: 카메라 {args.camera}번 열기 시도...")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"⚠️ {args.camera}번 카메라 실패 → 0번으로 재시도")
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("camera open failed")
        print("✅ 메인: 카메라 연결 성공")

        # 3) [v5.9] 3개의 스레드 시작
        
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
            frame = processed_frame_buffer.get("frame")
            
            if frame is None:
                try:
                    frame = frame_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.01)
                    continue

            cv2.imshow("Camera View", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
                
            time.sleep(0.01)

    except Exception as e:
        print(f"🚨 메인 루프에서 에러 발생: {e}")
    finally:
        # 7) 종료 처리
        print("🛑 종료 신호 감지... 모든 스레드 정리 중...")
        stop_event.set()
        
        for t in threads:
            t.join(timeout=1.0) 
            
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        if mc:
            mc.power_off()
        print("🔒 프로그램 종료")


if __name__ == "__main__":
    main()