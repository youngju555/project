# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 + YOLO 감지 + Stable Angle(EMA + Lock) + 정확도 기준 0.9
XY 이동이 안 되던 send_coords mode=1 문제 → 모두 mode=0으로 수정
"""
import time, argparse, threading, cv2, numpy as np
from collections import deque
from ultralytics import YOLO
try:
    from pymycobot.mycobot320 import MyCobot320 as CobotClass
except:
    from pymycobot.mycobot import MyCobot as CobotClass
# ===================== 포즈 설정 =====================
POSE_HOME   = [-264.3, 66.4, 325.0, -177.3, 7.78, 1.83]
POSE_SET1   = [-33.5, 208.1, 359.7, -147.75, -3.1, 4.54]
POSE_PLACE1 = [-13.7, 345.8, 200.3, 177.77, 2.49, 5.53]   # 양품
POSE_SET2   = [-253.0, 170.2, 366.4, -121.08, 0.6, 4.78]
POSE_PLACE2 = [-269.7, 244.4, 2.6, -152.37, -3.53, 7.72]  # 불량품
DEFAULT_SPEED = 25
# ===================== 픽셀→로봇 변환 보정값 =====================
SCALE_X = 0.33
SCALE_Y = 0.35
OFFSET_X = -5.0
OFFSET_Y = -85.0
# ===================== 카메라 보정 파라미터 =====================
K = np.array([
    [539.13729, 0.0, 329.02126],
    [0.0, 542.34217, 242.10995],
    [0.0, 0.0, 1.0]
])
D = np.array([[0.20528603, -0.76664068, -0.00096614, 0.00111891, 0.97630003]])
# ===================== 픽셀→로봇 좌표 변환 =====================
def pixel_to_robot(cx, cy, fw, fh):
    dx = (cx - fw/2) * SCALE_X
    dy = (cy - fh/2) * SCALE_Y
    robot_x = POSE_HOME[0] + OFFSET_X + dx
    robot_y = POSE_HOME[1] + OFFSET_Y - dy
    robot_z = POSE_HOME[2]
    print(f"[DEBUG] pixel→robot: {cx:.1f},{cy:.1f} → X={robot_x:.1f}, Y={robot_y:.1f}")
    return [robot_x, robot_y, robot_z, POSE_HOME[3], POSE_HOME[4], POSE_HOME[5]]
# ===================== 각도 필터 =====================
angle_buffer = deque(maxlen=10)
alpha = 0.2
angle_lock_threshold = 0.8
smooth_angle = None
locked_angle = None
lock_counter = 0
def reset_angle_filter():
    global angle_buffer, smooth_angle, locked_angle, lock_counter
    angle_buffer.clear()
    smooth_angle = None
    locked_angle = None
    lock_counter = 0
# ===================== 각도 계산 =====================
def get_angle_from_roi(frame, x1, y1, x2, y2):
    global angle_buffer, smooth_angle, locked_angle, lock_counter
    roi = frame[int(y1):int(y2), int(x1):int(x2)]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 300:
        return None
    rect = cv2.minAreaRect(c)
    ((_,_), (w,h), angle) = rect
    if angle < -45:
        angle += 90
    if w > h:
        angle += 90
    angle = angle % 90
    # ==== 안정화 ====
    angle_buffer.append(angle)
    avg = np.mean(angle_buffer)
    if smooth_angle is None:
        smooth_angle = avg
    else:
        smooth_angle = alpha * avg + (1 - alpha) * smooth_angle
    if abs(smooth_angle - avg) < angle_lock_threshold:
        lock_counter += 1
        if lock_counter > 10:
            locked_angle = smooth_angle
    else:
        locked_angle = None
        lock_counter = 0
    return locked_angle if locked_angle is not None else smooth_angle
# ===================== YOLO 감지 =====================
def detect_object(model, frame):
    results = model.predict(frame, imgsz=640, conf=0.55, verbose=False)
    r = results[0]
    boxes = r.boxes
    frame_vis = frame.copy()
    if len(boxes) == 0:
        return frame_vis, None, None, None, None, None, False
    # 최고 conf 박스 선택
    box = max(boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = box.xyxy[0]
    conf = float(box.conf[0])
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    angle = get_angle_from_roi(frame, x1, y1, x2, y2)
    # 정확도 0.9 기준
    is_defect = True if conf < 0.9 else False
    color = (0,0,255) if is_defect else (0,255,0)
    cv2.rectangle(frame_vis,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
    cv2.circle(frame_vis,(cx,cy),5,(0,255,255),-1)
    cv2.putText(frame_vis,f"Conf:{conf:.2f}",(int(x1),int(y1)-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
    if angle is not None:
        cv2.putText(frame_vis,f"Angle:{angle:.2f}",(int(x1),int(y2)+20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
    return frame_vis, (cx,cy), angle, conf, frame.shape[1], frame.shape[0], is_defect
# ===================== 메인 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--model", type=str, default="best.pt")
    args = parser.parse_args()
    model = YOLO(args.model)
    print(":흰색_확인_표시: YOLO 모델 로드 완료")
    mc = CobotClass(args.port, args.baud)
    mc.power_on()
    time.sleep(1)
    mc.send_angles([0,0,0,0,0,0],15)
    time.sleep(5)
    mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
    mc.set_gripper_value(50,20,1)
    time.sleep(2)
    # ---- 9회 반복 ----
    for cycle in range(1,10):
        print(f"\n====== :반복: 사이클 {cycle} / 9 ======\n")
        reset_angle_filter()
        frame_container = {"frame":None}
        stop_event = threading.Event()
        # 카메라 스레드
        def camera_thread():
            cap = cv2.VideoCapture(0)
            w,h = 640,480
            new_K,_ = cv2.getOptimalNewCameraMatrix(K,D,(w,h),1,(w,h))
            mx,my = cv2.initUndistortRectifyMap(K,D,None,new_K,(w,h),5)
            while not stop_event.is_set():
                ret,frame = cap.read()
                if ret:
                    frame = cv2.remap(frame,mx,my,cv2.INTER_LINEAR)
                    frame_container["frame"] = frame
                time.sleep(0.03)
            cap.release()
        cam = threading.Thread(target=camera_thread, daemon=True)
        cam.start()
        detected = None
        is_defect_flag = False
        detect_start = None
        # 감지 루프
        while True:
            frame = frame_container["frame"]
            if frame is None:
                continue
            vis, result, angle, conf, fw, fh, is_defect = detect_object(model, frame)
            if result:
                cx,cy = result
                if detect_start is None:
                    detect_start = time.time()
                elif time.time() - detect_start > 3:
                    print(":큰_초록색_원: 물체 확정!")
                    detected = pixel_to_robot(cx,cy,fw,fh)
                    is_defect_flag = is_defect
                    break
            else:
                detect_start = None
            cv2.imshow("Camera", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        stop_event.set()
        cam.join()
        cv2.destroyWindow("Camera")
        if detected is None:
            print(":x: 감지 실패, 스킵")
            continue
        x,y,z,r,p,yaw = detected
        # ---- 박스 위로 이동 (mode=0 !!!) ----
        mc.send_coords([x,y,z+110,r,p,yaw], DEFAULT_SPEED, 0)
        time.sleep(1.5)
        # ---- 각도 재측정 ----
        frame = frame_container["frame"]
        if frame is not None:
            _,_, new_angle, *_ = detect_object(model, frame)
            if new_angle is not None:
                print(f":삼각자: 재측정 각도={new_angle:.2f}")
                angles = mc.get_angles()
                if angles:
                    angles[5] += float(new_angle)
                    mc.send_angles(angles,25)
                    time.sleep(1)
        # ---- 박스 잡기 ----
        down = mc.get_coords()
        down[2] -= 40
        mc.send_coords(down, DEFAULT_SPEED, 0)
        time.sleep(1.5)
        mc.set_gripper_value(10,30,1)   # 세기 변경 없음
        time.sleep(1)
        up = mc.get_coords()
        up[2] += 120
        mc.send_coords(up,20,0)
        time.sleep(1.5)
        # ---- 분류 ----
        if is_defect_flag:
            mc.send_coords(POSE_SET2,25,0); time.sleep(2)
            mc.send_coords(POSE_PLACE2,25,0); time.sleep(2)
            mc.set_gripper_value(50,20,1); time.sleep(1)
            mc.send_coords(POSE_SET2,25,0); time.sleep(1)
        else:
            mc.send_coords(POSE_SET1,25,0); time.sleep(2)
            mc.send_coords(POSE_PLACE1,25,0); time.sleep(2)
            mc.set_gripper_value(50,20,1); time.sleep(1)
            mc.send_coords(POSE_SET1,25,0); time.sleep(1)
        mc.send_coords(POSE_HOME,25,0)
        time.sleep(1.5)
        print(f":흰색_확인_표시: 사이클 {cycle} 완료\n")
    print(":짠: 모든 9회 반복 완료!")
if __name__ == "__main__":
    main()