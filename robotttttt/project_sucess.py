# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 + YOLO 감지 및 분류 + OpenCV 기반 각도 추정
(카메라 내부 파라미터 직접 내장 버전)
[MERGE] YOLO가 '양품'/'불량'를 직접 분류하는 로직으로 변경
[v2.2] 픽업 직전 각도 재보정 제거 + 감지 각도 실제 반영(6번 조인트)
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
DEFAULT_SPEED = 20

# ===================== 보정 파라미터 =====================
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
    """
    dx = (cx - frame_w / 2) * SCALE_X
    dy = (cy - frame_h / 2) * SCALE_Y

    robot_x = POSE_HOME[0] + OFFSET_X + dx
    robot_y = POSE_HOME[1] + OFFSET_Y - dy
    robot_z = POSE_HOME[2]

    print(f"[DEBUG] pixel(cx={cx:.1f}, cy={cy:.1f}) "
          f"→ Δ(dx={dx:.1f}, dy={dy:.1f}) "
          f"→ robot(x={robot_x:.1f}, y={robot_y:.1f})")

    # [중요] 반환되는 yaw(POSE_HOME[5])는 초기 감지 시 무시됨
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
    time.sleep(1) # 홈에서 잠시 대기

    # ==================== 9회 반복 루프 시작 ====================
    for i in range(9):
        print(f"\n--- :사이클: 사이클 {i + 1} / 9 시작 ---")

        # [중요] 매 사이클마다 스레드 관련 변수들을 새로 생성해야 합니다.
        frame_container, stop_event = {"frame": None}, threading.Event()
        cam_thread = threading.Thread(target=camera_thread, args=(stop_event, frame_container), daemon=True)
        cam_thread.start()
        print(":카메라: 감지 중 (3초 이상 유지 시 픽업 시작)")

        detect_start, detected, detected_angle = None, None, None
        confirmed_is_defect = False
        quit_pressed = False # 'q' 감지 플래그

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
                    detected_angle = angle # [중요] 초기 각도 저장
                    confirmed_is_defect = is_defect
                    stop_event.set() # 감지 완료, 카메라 스레드 종료 신호
                    break
            else:
                detect_start = None
                
            cv2.imshow("Camera", frame_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set() # 'q' 입력, 카메라 스레드 종료 신호
                quit_pressed = True # 전체 루프 탈출 플래그 설정
                break

        cam_thread.join() # 카메라 스레드가 완전히 종료될 때까지 대기
        cv2.destroyAllWindows()

        # 'q'를 눌러서 종료한 경우, 9회 반복 루프(for)를 탈출
        if quit_pressed:
            print(":손바닥: 'q'로 수동 중단됨. 전체 작업을 종료합니다.")
            break

        # 물체를 감지하지 못한 경우, 9회 반복 루프의 다음 사이클로 넘어감
        if not detected:
            print(":x: 감지 실패. 홈에서 2초 대기 후 다음 사이클을 시도합니다.")
            mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0) # 홈으로 다시 이동
            time.sleep(2)
            continue # for 루프의 다음 i로 넘어감

        # --- (여기부터는 물체가 정상 감지된 경우) ---
        x, y, z, r, p, yaw = detected
        print(f":다트: 이동 목표 좌표: {[x, y, z, r, p, yaw]}") 

        # === :일: 좌표 이동 (Z+70 지점)
        mc.send_coords([x, y, 325, r, p, yaw], 25, 1) # 수정된 yaw 값으로 이동
        time.sleep(3)

        # ===== ✅ 수정된 회전 적용 부분 (여기만 변경되었습니다) =====
        # detect에서 얻은 detected_angle을 실제 6번 조인트에 반영하여 회전시키기
        try:
            angles = mc.get_angles()
            if angles:
                # angles는 list 형태, 인덱스 5가 6번 조인트(그리퍼 회전축)
                angles[5] += detected_angle
                mc.send_angles(angles, 25)
                time.sleep(1.5)
                # 업데이트된 yaw 변수 (이후 이동 명령에서 사용할 수 있도록 갱신)
                yaw = angles[5]
                print(f"[INFO] 그리퍼 회전 보정 완료 ({detected_angle:.1f}°) -> new yaw={yaw:.2f}")
            else:
                print("[WARN] mc.get_angles() 반환값 없음 — 회전 보정 생략")
        except Exception as e:
            print(f"[ERROR] 회전 보정 중 예외 발생: {e} — 넘어갑니다")

        # === :셋: 픽업 (Z=325 중복 이동 제거) ===
        PICK_Z = 275
        
        # PICK_Z로 바로 하강 (보정된 yaw 값 사용)
        mc.send_coords([x, y, PICK_Z, r, p, yaw], 20, 1)
        time.sleep(3)
        mc.set_gripper_value(10, 30, 1) # 그리퍼 닫기
        time.sleep(3)
        
        # === :넷: 상승 (추가) ===
        up_coords = mc.get_coords()
        if up_coords:
            up_coords[2] = 325 # 안전 높이(Z=325)로 상승
            mc.send_coords(up_coords, 25, 0)
            time.sleep(2)

        mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
        time.sleep(3)

        # === :다섯: 분류 및 놓기 ===
        if confirmed_is_defect:
            print(":빨간색_원: 불량품 (YOLO가 'DEFECT'로 분류)")
            mc.send_coords(POSE_PLACE2_UP, DEFAULT_SPEED, 0)
            time.sleep(3)
            mc.send_coords(POSE_PLACE2, DEFAULT_SPEED, 0)
        else:
            print(":큰_초록색_원: 양품 (YOLO가 'OK'로 분류)")
            mc.send_coords(POSE_PLACE1_UP, DEFAULT_SPEED, 0)
            time.sleep(3)
            mc.send_coords(POSE_PLACE1, DEFAULT_SPEED, 0)

        time.sleep(2)
        mc.set_gripper_value(50, 20, 1) # 그리퍼 열기
        time.sleep(1.5)

        # === :여섯: 홈 복귀 전 준비 ===
        if confirmed_is_defect:
            mc.send_coords(POSE_PLACE2_UP, DEFAULT_SPEED, 0)
        else:
            mc.send_coords(POSE_PLACE1_UP, DEFAULT_SPEED, 0)
        time.sleep(3)

        mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
        time.sleep(2)
        mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
        print(f":체크무늬_깃발: 사이클 {i + 1} / 9 완료 → 홈 복귀")
        time.sleep(3) # 다음 사이클 전 홈에서 잠시 대기

    # ==================== 9회 반복 루프 종료 ====================
    print("\n:파티_폭죽: 총 9회 사이클(또는 수동 중단) 완료. 프로그램을 종료합니다.")
    mc.power_off() # (선택 사항) 로봇 전원 끄기


if __name__ == "__main__":
    main()
