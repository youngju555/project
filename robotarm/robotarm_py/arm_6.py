# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 (pymycobot)
ROI 내 물체가 3초 이상 감지되면 각도 계산 후 그리퍼를 회전 정렬 → 열고 닫는 예제
(ROI 중앙점 표시 추가)
"""
import threading
import cv2
import time
import argparse
import numpy as np
# === MyCobot 클래스 불러오기 ===
try:
    from pymycobot.mycobot320 import MyCobot320 as CobotClass
except Exception:
    from pymycobot.mycobot import MyCobot as CobotClass
# -----------------------------
# 1) 포즈 정의
# -----------------------------
POSES = {
    "Home": [20.83, 24.08, 19.95, 37.88, -93.33, 20.56],
}
DEFAULT_SPEED = 20
# -----------------------------
# 2) ROI 내 색 감지 + 각도 계산 함수
# -----------------------------
def detect_color_and_angle(frame, target_color="blue"):
    """ROI 내 지정한 색 검출 및 회전각(angle) 계산"""
    h, w, _ = frame.shape
    # ROI 범위 (가운데 부분)
    roi_x1, roi_y1 = int(w * 0.6), int(h * 0.3)
    roi_x2, roi_y2 = int(w * 1.0), int(h * 0.8)
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    #  그리퍼 위치 시각화 + 중앙점 표시
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
    roi_cx, roi_cy = (roi_x1 + roi_x2)//2, (roi_y1 + roi_y2)//2
    cv2.drawMarker(frame, (roi_cx-165, roi_cy+210), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_ranges = {
        "red": [(0, 120, 70), (10, 255, 255)],
        "green": [(35, 80, 40), (85, 255, 255)],
        "blue": [(90, 80, 70), (130, 255, 255)],
        "yellow": [(20, 100, 100), (35, 255, 255)],
    }
    lower, upper = color_ranges.get(target_color, color_ranges["blue"])
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = False
    angle = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 400:
            rect = cv2.minAreaRect(c)
            (cx, cy), (w_box, h_box), angle = rect
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            # ROI 기준 → 전체 프레임 기준
            cx += roi_x1
            cy += roi_y1
            # 시각화
            cv2.drawContours(frame, [box + np.array([roi_x1, roi_y1])], 0, (255, 255, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{angle:.1f} deg", (int(cx) - 40, int(cy) + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            detected = True
    return frame, detected, angle
# -----------------------------
# 3) 그리퍼 제어 함수
# -----------------------------
def open_gripper(mc, value=50, speed=20):
    mc.set_gripper_mode(0)
    mc.init_electric_gripper()
    time.sleep(0.2)
    mc.set_gripper_value(value, speed, 1)
    print(":큰_초록색_원: Gripper Open")
    time.sleep(0.3)
def close_gripper(mc, value=10, speed=20):
    mc.set_gripper_mode(0)
    mc.init_electric_gripper()
    time.sleep(0.2)
    mc.set_gripper_value(value, speed, 1)
    print(":빨간색_원: Gripper Close")
    time.sleep(0.3)
# -----------------------------
# 4) 카메라 스레드 (감지 + 회전 보정 + 그리퍼 동작)
# -----------------------------
def camera_capture_thread(stop_event, frame_container, mc, target_color="blue"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(":경고: 카메라를 열 수 없습니다.")
        return
    print(":카메라: 카메라 스레드 시작")
    detect_start = None
    DETECT_TIME = 3.0
    action_done = False
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame, detected, angle = detect_color_and_angle(frame, target_color)
        frame_container["frame"] = frame
        if detected:
            if detect_start is None:
                detect_start = time.time()
            elif time.time() - detect_start >= DETECT_TIME and not action_done:
                print(f":큰_초록색_원: ROI 내 {target_color} 물체 3초 이상 감지됨 → 회전/그리퍼 동작")
                # --- 6축 회전 보정 ---
                if angle is not None:
                    current_angles = mc.get_angles()
                    rotation_correction = angle  # 방향은 테스트로 조정
                    current_angles[5] += rotation_correction
                    mc.send_angles(current_angles, 20)
                    print(f":다트: 6축 회전 보정 완료 ({rotation_correction:.1f}°)")
                # --- 그리퍼 동작 ---
                open_gripper(mc)
                close_gripper(mc)
                print(":로봇_얼굴: 그리퍼 동작 완료")
                action_done = True
        else:
            detect_start = None
            action_done = False
    cap.release()
    print(":카메라: 카메라 스레드 종료")
# -----------------------------
# 5) 메인 루틴
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--color", type=str, default="blue")
    args = parser.parse_args()
    mc = CobotClass(args.port, args.baud)
    time.sleep(0.5)
    mc.power_on()
    print(":전기_플러그: Power ON 완료")
    mc.send_angles(POSES["Home"], DEFAULT_SPEED)
    time.sleep(2)
    print(":집: Home 포즈 도달")
    frame_container = {"frame": None}
    stop_event = threading.Event()
    cam_thread = threading.Thread(
        target=camera_capture_thread,
        args=(stop_event, frame_container, mc, args.color),
        daemon=True
    )
    cam_thread.start()
    print(":흰색_확인_표시: 메인 루프 시작 (q 누르면 종료)")
    while not stop_event.is_set():
        frame = frame_container.get("frame")
        if frame is not None:
            cv2.imshow("Camera View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    stop_event.set()
    cam_thread.join()
    cv2.destroyAllWindows()
    print(":자물쇠: 종료 완료")
if __name__ == "__main__":
    main()