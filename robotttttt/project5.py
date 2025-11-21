# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 + YOLO 감지 + OpenCV 기반 각도 추정 + 색상 기반 불량 인식
[FINAL v2] 9회 반복 적재 + BGR2GRAY 오타 수정 + 각도 로직 개선
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
POSE_PLACE1_BASE = [-152.4,  181.4, 228.1, -170.3,   6.5,   38.8]   # 양품 (바닥)
POSE_PLACE2_BASE = [ -37.3,  318.6, 170.2,  162.81, -2.81, -29.21]  # 불량품 (바닥)
DEFAULT_SPEED = 20

# ===================== 적재 파라미터 =====================
CUBE_HEIGHT = 25.0   # 큐브 높이 (2.5cm = 25mm)
HOVER_OFFSET = 190.0 # 하강 전 상공 대기 높이 (mm)
PICKUP_Z = 282.0     # 픽업 Z 높이 (조정 필요)

# ===================== 보정 파라미터 (개잘잡히는 값) =====================
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
    dx = (cx - frame_w / 2) * SCALE_X
    dy = (cy - frame_h / 2) * SCALE_Y
    robot_x = POSE_HOME[0] + OFFSET_X + dx
    robot_y = POSE_HOME[1] + OFFSET_Y - dy
    robot_z = POSE_HOME[2] # Z는 나중에 덮어씀
    print(f"[DEBUG] pixel→robot: (cx={cx:.1f},cy={cy:.1f}) → ({robot_x:.1f},{robot_y:.1f})")
    return [robot_x, robot_y, robot_z, POSE_HOME[3], POSE_HOME[4], POSE_HOME[5]]

# ===================== OpenCV 기반 각도 계산 =====================
def get_angle_from_roi(frame, x1, y1, x2, y2):
    """[FIX 2] 개선된 각도 보정 로직 적용"""
    roi = frame[int(y1):int(y2), int(x1):int(x2)]
    if roi.size == 0:
        return 0.0
    
    # [FIX 1] cv2.COLOR_BGR2GRAY 오타 수정
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    
    (_, _), (w, h), raw_angle = rect

    # [FIX 2] 개선된 각도 계산 로직
    if w < h:
        angle = 90 + raw_angle
    else:
        angle = raw_angle
        if angle < -45:
            angle = 90 + raw_angle

    # 범위를 -90~90도로 제한
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
            
    return angle

# ===================== YOLO 감지 (불량은 노란점으로 판단) =====================
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
    
    # === 노란점 감지 (불량품) ===
    roi = frame[int(y1):int(y2), int(x1):int(x2)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 150])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = np.sum(mask_yellow > 0) / (roi.shape[0] * roi.shape[1])
    is_defect = yellow_ratio > 0.002
    
    status_text = "DEFECT" if is_defect else "OK"
    color_box = (0, 0, 255) if is_defect else (0, 255, 0)
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
    print("✅ YOLO 모델 로드 완료")
    
    mc = CobotClass(args.port, args.baud)
    mc.power_on()
    time.sleep(1)

    stack_count_good = 0
    stack_count_defect = 0

    for i in range(9):
        print(f"\n--- 🔄 사이클 {i+1} / 9 시작 ---")
        
        mc.send_angles([0,0,0,0,0,0],20)
        time.sleep(3)
        mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
        time.sleep(2)
        mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
        mc.set_gripper_mode(0)
        mc.set_electric_gripper(0)
        mc.set_gripper_value(50, 20, 1) # 그리퍼 열기
        time.sleep(2)
        print("🏠 홈 포즈 도달 및 초기화 완료")

        frame_container, stop_event = {"frame": None}, threading.Event()
        cam_thread = threading.Thread(target=camera_thread, args=(stop_event, frame_container), daemon=True)
        cam_thread.start()
        
        print("📷 감지 중 (3초 이상 유지 시 픽업 시작)")
        detect_start, detected, detected_angle, confirmed_is_defect = None, None, None, False
        
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
            print(f"❌ 감지 실패. 사이클 {i+1} 중단. 다음 사이클로 넘어갑니다.")
            time.sleep(3)
            continue

        x, y, z, r, p, yaw = detected
        print(f"🎯 이동 목표 좌표: {detected}")
        
        # [FIX 3] yaw 계산을 명확하게 수정
        base_yaw = POSE_HOME[5] 
        target_yaw = base_yaw + detected_angle
        print(f"🔧 최종 회전 각도 계산: (BaseYaw){base_yaw:.1f} + (ObjAngle){detected_angle:.1f} = {target_yaw:.1f}")

        # 2-1. 이동 (Z+70 지점) + 회전
        mc.send_coords([x, y, 325, r, p, target_yaw], 25, 1) # Z=325 (안전 높이)
        time.sleep(2)

        # 2-2. 하강 및 픽업
        mc.send_coords([x, y, PICKUP_Z, r, p, target_yaw], 20, 1)
        time.sleep(2)
        mc.set_gripper_value(10, 30, 1) # 집기
        time.sleep(2)

        # 2-3. 상승
        up = mc.get_coords()
        up[2] += 100
        mc.send_coords(up, 25, 0)
        time.sleep(2)

        # === 3. 분류 및 적재 (Stacking) ===
        mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
        time.sleep(2)

        if confirmed_is_defect:
            print(f"🔴 불량품 (스택 {stack_count_defect + 1}번째)")
            target_z = POSE_PLACE2_BASE[2] + (stack_count_defect * CUBE_HEIGHT)
            target_pose = POSE_PLACE2_BASE.copy()
            target_pose[2] = target_z
            hover_pose = target_pose.copy()
            hover_pose[2] += HOVER_OFFSET
            stack_count_defect += 1
            
        else: # 양품
            print(f"🟢 양품 (스택 {stack_count_good + 1}번째)")
            target_z = POSE_PLACE1_BASE[2] + (stack_count_good * CUBE_HEIGHT)
            target_pose = POSE_PLACE1_BASE.copy()
            target_pose[2] = target_z
            hover_pose = target_pose.copy()
            hover_pose[2] += HOVER_OFFSET
            stack_count_good += 1

        # [NEW] 적재 로직 (상공 -> 하강 -> 릴리즈 -> 상공)
        mc.send_coords(hover_pose, DEFAULT_SPEED, 0)
        time.sleep(2.5)
        mc.send_coords(target_pose, DEFAULT_SPEED, 0)
        time.sleep(2)
        mc.set_gripper_value(50, 20, 1) # 릴리즈
        time.sleep(1)
        mc.send_coords(hover_pose, DEFAULT_SPEED, 0)
        time.sleep(2)

        # === 4. 홈 복귀 (다음 사이클 준비) ===
        mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
        time.sleep(1)
        mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
        print(f"🏁 사이클 {i+1} / 9 완료 → 홈 복귀")
        time.sleep(3)

    print("🎉🎉 9개 사이클 모두 완료! 🎉🎉")
    mc.power_off()


if __name__ == "__main__":
    main()