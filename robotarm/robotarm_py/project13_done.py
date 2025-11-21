# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 + YOLO 위치 감지 + Confidence(정확도) 기반 불량 판정
[v2.6-stack] 9회 반복 쌓기형 배치 버전
------------------------------------------------------------
- 기존 로직은 그대로 유지
- 양품 / 불량품 각각 쌓기형 (Z축 +20씩 상승)
- send_coords에서 회전은 1단계로 처리
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
STACK_HEIGHT = 27 # 한번 놓을 때마다 20mm씩 상승
stack_count_good = 0
stack_count_bad = 0

# ===================== 보정 파라미터 =====================
SCALE_X = 0.33
SCALE_Y = 0.37
OFFSET_X = -5.0
OFFSET_Y = -82.0

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
    dx = (cx - frame_w / 2) * SCALE_X
    dy = (cy - frame_h / 2) * SCALE_Y
    robot_x = POSE_HOME[0] + OFFSET_X + dx
    robot_y = POSE_HOME[1] + OFFSET_Y - dy
    robot_z = POSE_HOME[2]
    print(f"[DEBUG] pixel→robot: (cx={cx:.1f},cy={cy:.1f}) → ({robot_x:.1f},{robot_y:.1f})")
    return [robot_x, robot_y, robot_z, POSE_HOME[3], POSE_HOME[4], POSE_HOME[5]]

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

    print(f"[YOLO->CONF] 감지: {name} | conf={conf:.2f} | angle={angle:.1f}° | defect={is_defect}")
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
    global stack_count_good, stack_count_bad

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--model", type=str, default="/home/young/Downloads/best.pt")
    args = parser.parse_args()

    model = YOLO(args.model)
    print("✅ YOLO 모델 로드 완료 (분류 로직: Confidence < 0.8)")

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
    print("🏠 홈 포즈 도달 및 초기화 완료")
    time.sleep(1)

    # ==================== 9회 반복 ====================
    for i in range(9):
        print(f"\n--- 🔁 사이클 {i+1}/9 시작 ---")

        frame_container, stop_event = {"frame": None}, threading.Event()
        cam_thread = threading.Thread(target=camera_thread, args=(stop_event, frame_container), daemon=True)
        cam_thread.start()

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

        # [v2.6-stack]의 감지 루프가 끝난 직후
# ...
        cam_thread.join()
        cv2.destroyAllWindows()

        if not detected:
            print("❌ 감지 실패. 홈에서 대기 후 다음 사이클 시도.")
            mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
            time.sleep(2)
            continue

        # === [수정된 픽업 로직 시작] ===

        x, y, z, r, p, yaw = detected # 'detected'는 pixel_to_robot의 결과 (yaw는 POSE_HOME[5])

        # 1. (x,y)로 우선 이동 (회전 없이)
        # (이때 yaw는 detected_angle이 더해지지 않은 POSE_HOME[5] 값)
        print(f"🧭 1단계: (x,y)로 이동 ({x:.1f}, {y:.1f})")
        mc.send_coords([x, y, 325, r, p, yaw], 25, 1) 
        time.sleep(3) # (이동 대기)

        # 2. 'send_angles'로 제자리에서 회전 (J6 상대 회전)
        angles = mc.get_angles()    
        if angles:
            print(f"[DEBUG] 현재 J6 각도: {angles[5]:.1f}, 보정할 각도: {detected_angle:.1f}")
    
            # [중요] 현재 관절 각도에 감지된 각도를 '더함'
            angles[5] += detected_angle 
    
            print(f"🧭 2단계: J6 회전 보정. 목표 각도: {angles[5]:.1f}")
            mc.send_angles(angles, 25)
            time.sleep(2) # (회전 대기)
            print("🧭 각도 보정 완료!")
        else:
            print("❌ 에러: 로봇 각도를 읽을 수 없습니다. 픽업을 중단합니다.")
            continue

        # --- [수정된 하강/상승 로직] ---
        # 3. 하강 (회전이 완료된 상태)
        print("🧭 3단계: 현재 좌표 읽어오기 (Z 하강 준비)")
        current_coords = mc.get_coords()
        if not current_coords:
            print("❌ 에러: 로봇 좌표 읽기 실패. 하강 중단.")
            continue
        
        down_coords = current_coords.copy()
        down_coords[2] = 275 # Z축만 275로 변경
        
        print(f"🧭 3단계: Z축 '그대로' 하강 (Z=275)")
        # (x,y,r,p,yaw)는 현재 값 그대로 두고, Z만 275로 가는 '선형' 이동(mode=0)
        mc.send_coords(down_coords, 20, 0) 
        time.sleep(2)

        # 4. 잡기
        print("🖐️ 그리퍼 닫기")
        mc.set_gripper_value(10, 30, 1)
        time.sleep(2) # 그리퍼 대기

        # 5. 상승 (하강과 동일한 원리)
        print("🖐️ Z축 '그대로' 상승 (Z=325)")
        up_coords = mc.get_coords()
        if not up_coords:
            up_coords = down_coords # 방금 전 좌표라도 사용
            
        up_coords[2] = 325 # Z축만 325로 변경
        mc.send_coords(up_coords, 25, 0)
        time.sleep(2)
        
        # === [수정된 픽업 로직 끝] ===

        mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
        time.sleep(3)


        # === 쌓기 로직 ===
        if confirmed_is_defect:
            z_offset = STACK_HEIGHT * stack_count_bad
            print(f"🔴 불량품 쌓기 {stack_count_bad+1}단 (Z +{z_offset})")
            place_up = POSE_PLACE2_UP.copy()
            place_down = POSE_PLACE2.copy()
            place_up[2] += z_offset
            place_down[2] += z_offset
            stack_count_bad += 1
        else:
            z_offset = STACK_HEIGHT * stack_count_good
            print(f"🟢 양품 쌓기 {stack_count_good+1}단 (Z +{z_offset})")
            place_up = POSE_PLACE1_UP.copy()
            place_down = POSE_PLACE1.copy()
            place_up[2] += z_offset
            place_down[2] += z_offset
            stack_count_good += 1

        mc.send_coords(place_up, DEFAULT_SPEED, 0)
        time.sleep(3)
        mc.send_coords(place_down, DEFAULT_SPEED, 0)
        time.sleep(2)
        mc.set_gripper_value(50, 20, 1)
        time.sleep(1.5)
        mc.send_coords(place_up, DEFAULT_SPEED, 0)
        time.sleep(2)

        mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
        time.sleep(2)
        mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
        time.sleep(2)
        print(f"✅ 사이클 {i+1}/9 완료 → 홈 복귀")

    print("\n🎉 총 9회 쌓기 완료. 프로그램 종료.")
    mc.power_off()


if __name__ == "__main__":
    main()
