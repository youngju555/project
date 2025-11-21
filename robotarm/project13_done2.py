# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 + YOLO 위치 감지 + Confidence(정확도) 기반 불량 판정
[v2.8-drop-relative] 9회 반복 드롭형 배치 버전
------------------------------------------------------------
- [MODIFIED] 1순위 누적 오차 해결
    - pixel_to_robot이 mc 객체를 받아 '현재 좌표' 기준으로 상대 계산
- [MODIFIED] 5분 시간제한을 위한 속도 확보
    - '쌓기' 로직을 '드롭' 로직(UP 포즈에서 바로 열기)으로 변경
- [KEPT] v2.7의 wait_for_robot_stop() 유지
- [KEPT] v2.7의 카메라 루프 밖 초기화 유지
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

POSE_PLACE1 = [-13.7, 345.8, 167.3, 177.77, 2.49, 5.53]    # 양품 (이제 사용 안 함)
POSE_PLACE1_UP = [-33.5, 208.1, 349.7, -147.75, -3.1, 4.54] # 양품 드롭 위치
POSE_PLACE2 =  [-269.7, 244.4, 204.6, -152.37, -3.53, 7.72] # 불량품 (이제 사용 안 함)
POSE_PLACE2_UP = [-253.0, 170.2, 366.4, -121.08, 0.6, 4.78] # 불량품 드롭 위치
DEFAULT_SPEED = 15

# ===================== 쌓기 설정 (REMOVED) =====================
# STACK_HEIGHT = 27 # 제거
# stack_count_good = 0 # 제거
# stack_count_bad = 0 # 제거

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

# ===================== [MODIFIED] 픽셀 → 로봇 변환 (1순위 해결) =====================
def pixel_to_robot(mc, cx, cy, frame_w, frame_h):
    """
    [MODIFIED]
    POSE_HOME 상수가 아닌, mc.get_coords()로 읽어온 '현재' 좌표를 기준으로
    픽업 좌표를 계산하여 누적 오차를 방지합니다.
    """
    
    # [NEW] 현재 로봇의 좌표를 읽어온다 (이것이 POSE_HOME 대신 기준점이 됨)
    current_coords = mc.get_coords()
    if not current_coords:
        print("⚠️ pixel_to_robot에서 좌표 읽기 실패! POSE_HOME으로 대체합니다.")
        current_coords = POSE_HOME.copy() # .copy() 추가
    
    # [MODIFIED] POSE_HOME[0] 대신 current_coords[0] (현재 X) 사용
    dx = (cx - frame_w / 2) * SCALE_X
    dy = (cy - frame_h / 2) * SCALE_Y
    robot_x = current_coords[0] + OFFSET_X + dx 
    robot_y = current_coords[1] + OFFSET_Y - dy
    robot_z = current_coords[2] # 현재 Z_HOME
    
    print(f"[DEBUG] pixel→robot (Relative): (기준={current_coords[0]:.1f}, {current_coords[1]:.1f}) → (타겟={robot_x:.1f},{robot_y:.1f})")
    
    # [MODIFIED] 기준 자세도 현재 자세(읽어온 값)를 사용
    return [robot_x, robot_y, robot_z, current_coords[3], current_coords[4], current_coords[5]]

# <<< [NEW] 로봇 이동 완료 대기 함수 >>>
def wait_for_robot_stop(mc, pos_tolerance=0.8, ang_tolerance=0.5, poll_interval=0.2, max_wait_time=15.0):
    """
    로봇이 이동을 완료하고 멈출 때까지 대기합니다.
    get_coords()를 반복적으로 폴링하여 좌표 변화가 없을 때를 감지합니다.
    """
    print("⏳ 로봇 이동 완료 대기 중...")
    start_time = time.time()
    
    last_coords = mc.get_coords()
    if not last_coords:
        # 통신 실패 시, 이전처럼 고정 시간 대기 (Fallback)
        print("⚠️ 로봇 좌표 초기값 읽기 실패. 2초 대기합니다.")
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
    if roi.size == 0: return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, CV_THRESH_BINARY+cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0.0
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    angle = rect[2]
    if angle < -45: angle = 90 + angle
    return angle

# ===================== YOLO 감지 + Confidence 기반 불량 판별 =====================
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
    is_defect = conf < 0.94
    color_box = (0, 0, 255) if is_defect else (0, 255, 0)
    cv2.rectangle(frame_vis, (int(x1), int(y1)), (int(x2), int(y2)), color_box, 2)
    cv2.circle(frame_vis, (cx, cy), 5, (0, 255, 255), -1)
    cv2.putText(frame_vis, f"{name} ({conf:.2f}) | {angle:.1f}° | {'DEFECT' if is_defect else 'OK'}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)

    # print(f"[YOLO->CONF] 감지: {name} | conf={conf:.2f} | angle={angle:.1f}° | defect={is_defect}") # 너무 많이 출력되어 주석 처리
    return frame_vis, (cx, cy), angle, conf, frame.shape[1], frame.shape[0], is_defect

# ===================== 카메라 스레드 ===================== 
def camera_thread(stop_event, frame_container, cap, mapx, mapy):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            frame_container["frame"] = undistorted
        time.sleep(0.03)

# ===================== 메인 루틴 =====================
def main():
    # global stack_count_good, stack_count_bad # [REMOVED]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--model", type=str, default="/home/young/Downloads/best.pt")
    args = parser.parse_args()

    # --- 로봇 초기화 ---
    print("🤖 로봇 연결 시도...")
    mc = CobotClass(args.port, args.baud)
    mc.power_on()
    time.sleep(1) 
    mc.send_angles([0,0,0,0,0,0],20)
    wait_for_robot_stop(mc)
    mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
    wait_for_robot_stop(mc)
    mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
    mc.set_gripper_mode(0)
    mc.set_electric_gripper(0)
    mc.set_gripper_value(50, 20, 1)
    print("🏠 홈 포즈 도달 및 초기화 완료")
    wait_for_robot_stop(mc) # 홈 도착 대기
    
    # --- YOLO 모델 로드 ---
    print("🧠 YOLO 모델 로드 중...")
    model = YOLO(args.model)
    print("✅ YOLO 모델 로드 완료 (분류 로직: Confidence < 0.94)")

    # --- 카메라를 루프 밖에서 *한 번만* 엽니다 ---
    print("📷 카메라 초기화 시도...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(":x: 카메라 열기 실패. 프로그램을 종료합니다.")
        mc.power_off()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w, h = 640, 480
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), 5)
    print("✅ 카메라 초기화 완료.")
    # -------------------------------------------------

    # ==================== 9회 반복 ====================
    try:
        for i in range(9):
            print(f"\n--- 🔁 사이클 {i+1}/9 시작 ---")

            # [중요] 홈 포즈가 약간 틀어졌을 수 있으므로, 매 사이클마다 홈으로 '다시' 이동
            # 이것이 1순위 해결책 B(차선책)의 변형이며, 1순위 A와 함께 사용하면 더욱 강력함
            print("🏠 홈 포즈 정렬...")
            mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
            wait_for_robot_stop(mc)

            frame_container, stop_event = {"frame": None}, threading.Event()
            
            cam_thread = threading.Thread(
                target=camera_thread, 
                args=(stop_event, frame_container, cap, mapx, mapy), 
                daemon=True
            )
            cam_thread.start()

            detect_start, detected, detected_angle = None, None, None
            confirmed_is_defect = False

            print("👀 물체 감지 시작...")
            while not stop_event.is_set():
                frame = frame_container.get("frame")
                if frame is None:
                    continue
                frame_vis, result, angle, conf, fw, fh, is_defect = detect_object(model, frame)
                if result:
                    cx, cy = result
                    if detect_start is None:
                        print("...물체 감지됨. 4초간 위치 고정 대기...")
                        detect_start = time.time()
                    elif time.time() - detect_start > 3.0: # 4초간 흔들림 없이 고정되면
                        print("✅ 위치 고정 확인. 픽업 좌표 계산.")
                        
                        # [MODIFIED] 1순위 해결책 적용
                        # mc 객체를 전달하여 '현재 위치' 기준으로 좌표 계산
                        detected = pixel_to_robot(mc, cx, cy, fw, fh) 
                        
                        detected_angle = angle
                        confirmed_is_defect = is_defect
                        stop_event.set() # 카메라 스레드 중지 신호
                        break
                else:
                    if detect_start is not None:
                        print("...물체 사라짐. 감지 초기화...")
                    detect_start = None # 물체가 사라지면 타이머 리셋
                
                cv2.imshow("Camera", frame_vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

            cam_thread.join() 
            cv2.destroyAllWindows()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                 print("🛑 'q' 입력으로 사용자가 중지했습니다.")
                 break 
            if not detected:
                print("❌ 감지 실패. 홈에서 대기 후 다음 사이클 시도.")
                # mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0) # 이미 위에서 홈으로 감
                # wait_for_robot_stop(mc)
                continue

            # === [픽업 로직 (v2.7과 동일)] ===
            x, y, z, r, p, yaw = detected 
            print(f"🧭 1단계: (x,y)로 이동 ({x:.1f}, {y:.1f})")
            mc.send_coords([x, y, 325, r, p, yaw], 25, 1)
            wait_for_robot_stop(mc)

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

            print("🧭 3단계: 현재 좌표 읽어오기 (Z 하강 준비)")
            current_coords = mc.get_coords()
            if not current_coords:
                print("❌ 에러: 로봇 좌표 읽기 실패. 하강 중단.")
                continue
            
            down_coords = current_coords.copy()
            down_coords[2] = 275 # Z축만 275로 변경
            print(f"🧭 3단계: Z축 '그대로' 하강 (Z=275)")
            mc.send_coords(down_coords, 20, 0) 
            wait_for_robot_stop(mc)

            print("🖐️ 그리퍼 닫기")
            mc.set_gripper_value(10, 30, 1)
            time.sleep(2) # [KEPT] 그리퍼 작동 대기

            print("🖐️ Z축 '그대로' 상승 (Z=325)")
            up_coords = mc.get_coords()
            if not up_coords: up_coords = down_coords
            up_coords[2] = 325
            mc.send_coords(up_coords, 25, 0)
            wait_for_robot_stop(mc)
            
            mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
            wait_for_robot_stop(mc)
            # === [픽업 로직 끝] ===


            # === [NEW DROP LOGIC] (쌓기 로직 대체) ===
            if confirmed_is_defect:
                print(f"🔴 불량품 드롭 위치로 이동")
                target_drop_pose = POSE_PLACE2_UP.copy()
            else:
                print(f"🟢 양품 드롭 위치로 이동")
                target_drop_pose = POSE_PLACE1_UP.copy()

            # 'UP' 위치(상공)로 이동
            mc.send_coords(target_drop_pose, DEFAULT_SPEED, 0)
            wait_for_robot_stop(mc)
            
            # 그 자리에서 바로 드롭
            print("🖐️ 그리퍼 열기 (드롭)")
            mc.set_gripper_value(50, 20, 1)
            time.sleep(1.5) # [KEPT] 그리퍼 작동 대기
            
            # 하강, 상승 로직 모두 제거됨 (시간 단축)
            # === [드롭 로직 끝] ===

            mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
            wait_for_robot_stop(mc)
            
            # mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0) # 어차피 루프 시작 시 감
            # wait_for_robot_stop(mc)
            
            print(f"✅ 사이클 {i+1}/9 완료")

    except KeyboardInterrupt:
        print("\n🛑 (Ctrl+C) 사용자가 프로그램을 강제 종료했습니다.")
    
    finally:
        # --- 모든 작업이 끝나면 리소스 해제 ---
        print("\n🎉 모든 작업 종료. 리소스 해제 중...")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
            print("📷 카메라 해제 완료.")
        cv2.destroyAllWindows()
        mc.power_off()
        print("🤖 로봇 전원 차단 완료.")


if __name__ == "__main__":
    main()