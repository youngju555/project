# -*- coding: utf-8 -*-
"""
axis_test_v3.py
- 축 매핑 테스트 (로봇 이동 기능 추가)
- 1. 로봇을 사용자가 지정한 'Home' 좌표로 이동
- 2. 카메라를 켜고 (cx, cy) 픽셀 좌표 출력
"""
import cv2
import numpy as np
import argparse
import time
try:
    from pymycobot.mycobot320 import MyCobot320 as CobotClass
except Exception:
    from pymycobot.mycobot import MyCobot as CobotClass
# ===============================================================
# [!!! 필수 수정 !!!]
# 메인 코드(v5.2)의 POSES["Home"]과 "완전히 동일하게" 설정해야 합니다.
# ===============================================================
SEARCH_POSE = [59.8, -215.9, 354.6, -175.33, 8.65, 86.68]
# ===============================================================
def find_object(frame, target_color="blue"):
    # (find_object 함수 내용은 이전 v2 테스트 코드와 동일합니다)
    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2
    color_ranges = {
        "red":    [(0, 120, 70),  (10, 255, 255)],
        "green":  [(35, 80, 40),  (85, 255, 255)],
        "blue":   [(100, 80, 40), (140, 255, 255)],
        "yellow": [(20, 100, 100), (35, 255, 255)],
    }
    lower, upper = color_ranges.get(target_color, ((0,0,0),(0,0,0)))
    if np.all(lower == 0) and np.all(upper == 0):
        return frame, None, None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found_cx, found_cy = None, None
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 300:
            x, y, w_box, h_box = cv2.boundingRect(c)
            found_cx = x + w_box // 2
            found_cy = y + h_box // 2
            cv2.drawMarker(frame, (center_x, center_y), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 255, 0), 2)
            cv2.drawMarker(frame, (found_cx, found_cy), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
            cv2.putText(frame, f"cx: {found_cx}, cy: {found_cy}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame, found_cx, found_cy
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--color", type=str, default="blue")
    parser.add_argument("--cam_id", type=int, default=0)
    args = parser.parse_args()
    # --- 1. 로봇 연결 및 'Home' 위치로 이동 ---
    mc = None
    try:
        mc = CobotClass(args.port, args.baud)
        time.sleep(0.5)
        mc.power_on()
        print(":전기_플러그: Power ON 완료")
        print(f":로봇_얼굴: 테스트를 위해 'Home 좌표' {SEARCH_POSE}로 이동합니다...")
        mc.send_coords(SEARCH_POSE, args.speed)
        time.sleep(3) # 이동 대기
        print(":흰색_확인_표시: Home 좌표 도착. 카메라 테스트를 시작합니다.")
    except Exception as e:
        print(f":경고: 로봇 연결 실패: {e}")
        mc = None
    # --- 2. 카메라 켜고 테스트 시작 ---
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        print(f":경고: 카메라 (ID: {args.cam_id})를 열 수 없습니다.")
        if mc: mc.power_off()
        return
    print("---")
    print("물체를 로봇의 물리적인 +X축, +Y축으로 움직여보세요.")
    print("콘솔의 (cx, cy) 값이 어떻게 변하는지 관찰하세요.")
    print("---")
    last_print_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            processed_frame, cx, cy = find_object(frame, args.color)
            current_time = time.time()
            if current_time - last_print_time > 0.1:
                if cx is not None:
                    print(f"\rObject Center: cx={cx:04d}, cy={cy:04d}", end="")
                else:
                    print("\r... 물체 찾는 중 ...               ", end="")
                last_print_time = current_time
            cv2.imshow("Axis Mapping Test (Press 'q' to quit)", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if mc:
            mc.power_off()
        print("\n:자물쇠: 테스트 종료")
if __name__ == "__main__":
    main()