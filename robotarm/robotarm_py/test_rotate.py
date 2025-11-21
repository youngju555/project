# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 (pymycobot)
카메라로 특정 색을 감지한 뒤,
- 감지한 물체의 화면 내 좌표와 거리(대략)를 계산하고
- 그 값을 로봇 좌표계로 변환해서 JSON 파일로 저장하기 위한 예제

※ 현재 버전에서는 "로봇이 실제로 정렬해서 움직이는 부분"이 주석 처리돼 있고,
   카메라에서 색을 찾고 좌표를 만드는 흐름이 남아 있음.
"""

# -----------------------------
# 기본 라이브러리 임포트
# -----------------------------
import threading       # 카메라를 별도 스레드로 돌리기 위해 사용
import cv2             # OpenCV: 카메라 캡처, 색 추출, 화면 표시
import time            # 대기(sleep) 처리
import argparse        # 실행 시 옵션(--port, --color 등) 받기
import numpy as np     # 영상 처리 시 배열 연산
import json, os        # 좌표를 JSON으로 저장 / 파일 존재 여부 확인

# 전역 플래그: 한 번 피킹 좌표를 저장하면 True로 바꿔서 중복 저장을 방지
picking_done = False

# -----------------------------
# 로봇 클래스 임포트
# -----------------------------
# 사용 환경에 따라 mycobot320이 있기도 하고, 일반 mycobot만 있을 수도 있어
# 두 경우를 모두 커버하기 위한 try-except
try:
    # MyCobot 320 전용 클래스
    from pymycobot.mycobot320 import MyCobot320 as CobotClass
except Exception:
    # 위 임포트가 실패하면 일반 MyCobot 클래스를 사용
    from pymycobot.mycobot import MyCobot as CobotClass

# -----------------------------
# 자주 쓰는 포즈(좌표) 정의
# 이 값은 사용자가 테스트하면서 미리 뽑아둔 좌표라고 보면 됨.
# send_coords 형태의 6자유도 포맷: [x, y, z, rx, ry, rz]
# -----------------------------
POSES = {
    "Home":  [59.8, -215.9, 354.6, -175.33, 8.65, 86.68],  # 시작/대기 위치
    "Place": [105.8, -65.0, 483.4, -116.46, 4.87, -78.69],  # 예: 내려둘 위치
    "Box1": [291.3, 210.0, 200, -172.57, -1.46, -87.15],  # 예: 내려둘 위치
    "Box2": [333.4, 11.7, 200, -175.19, -0.08, -89.53],  # 예: 내려둘 위치
    "Box3": [319.9, -169.5, 200, -172.32, -2.86, -87.15],  # 예: 내려둘 위치
    "Box1_up": [229.8, 132.6, 386.4, -147.34, 9.15, -74.66],  # 예: 내려둘 위치
    "Box2_up": [264.0, -1.3, 379.0, -153.71, 11.7, -106.33],  # 예: 내려둘 위치
    "Box3_up": [228.0, -203.0, 362.8, -146.13, 15.2, -149.53],  # 예: 내려둘 위치
}

# 로봇 이동 시 기본 속도
DEFAULT_SPEED = 20


# ======================================================================
# 1. 픽셀 좌표 → 로봇 좌표 대략 변환 함수
# ======================================================================
def pixel_to_robot(cx, cy, distance_cm, frame_w, frame_h):
    """
    화면(이미지) 상의 중심점(cx, cy)과 실제 거리 값(대략)을 받아
    로봇이 이해할 수 있는 x, y, z 좌표로 바꿔주는 함수.

    실제로는 카메라와 로봇의 상대 위치, 카메라 높이, 각도에 따라
    꽤 많은 보정이 필요하지만 여기서는 '대략 이렇게 변환한다'는 예시를 보여줌.
    """

    # 카메라 화면의 중심점(픽셀). 여기 기준으로 얼마나 벗어났는지 계산하려고 구해둠.
    center_x, center_y = frame_w / 2, frame_h / 2

    # 1픽셀이 실제 몇 mm인지에 대한 스케일값.
    # 실제 환경에서는 캘리브레이션으로 이 값을 맞춰야 함.
    scale = 0.4  # mm/pixel

    # ------------------------------------------------------------
    # 방향 보정
    # ------------------------------------------------------------
    # cx - center_x : 화면 중심에서 얼마나 오른쪽(+)으로 치우쳐 있는지
    # cy - center_y : 화면 중심에서 얼마나 아래쪽(+)으로 치우쳐 있는지
    #
    # 그런데 로봇 좌표계와 카메라 좌표계의 축 방향이 다를 수 있으므로
    # 여기서는 음수(-)를 붙여서 "카메라 오른쪽 → 로봇 왼쪽" 식으로 반대 변환
    dx = -(cx - center_x) * scale        # X축 보정량 (mm)
    dy = -(cy - center_y) * scale        # Y축 보정량 (mm)

    # z는 거리 기반으로 계산.
    # distance_cm는 카메라에서 물체까지의 거리를 "대략" 잰 값.
    # 여기서는 물체에 완전히 붙지 않고 20cm 정도 떨어져 멈추도록 (distance_cm - 20)
    # 그리고 로봇 좌표는 mm 단위로 쓴다고 가정해서 * 10
    dz = (distance_cm - 20) * 10

    # ------------------------------------------------------------
    # 로봇 기준 오프셋
    # ------------------------------------------------------------
    # 카메라가 로봇 툴 중앙에 딱 달려있지 않은 경우가 많음.
    # 예를 들어 카메라가 로봇 기준으로 x쪽으로 120mm 떨어져 있다면
    # 이만큼을 기본값으로 더해줘야 함.
    ROBOT_OFFSET_X = 120.0
    ROBOT_OFFSET_Y = 0.0
    ROBOT_OFFSET_Z = 30.0

    # 카메라 기준에서 로봇 기준으로 변환한 좌표
    robot_x = ROBOT_OFFSET_X + dx
    robot_y = ROBOT_OFFSET_Y + dy
    robot_z = ROBOT_OFFSET_Z + dz

    # 소수점 2자리까지 반올림해서 dict로 반환
    return {
        "x": round(robot_x, 2),
        "y": round(robot_y, 2),
        "z": round(robot_z, 2)
    }


# ======================================================================
# 2. 계산된 피킹 좌표를 JSON 파일로 저장하는 함수
# ======================================================================
def save_pick_coordinate(data, filename="picking_target.json"):
    """
    data: {"x": ..., "y": ..., "z": ...} 형태의 dict
    filename: 저장할 파일명
    """
    with open(filename, "w", encoding="utf-8") as f:
        # indent=4 로 예쁘게 들여쓰기, ensure_ascii=False로 한글도 그대로
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"💾 피킹 좌표 저장 완료 → {filename} / {data}")


# ======================================================================
# 3. 프레임(이미지)에서 특정 색을 찾고, 그 위치와 거리까지 계산하는 함수
# ======================================================================
def detect_color_and_distance(frame, target_color="blue"):
    """
    1) 입력받은 frame에서 ROI(가운데 영역)를 지정
    2) 해당 영역에서 HSV 색공간으로 변환
    3) 지정한 색 범위에 맞는 마스크를 만들고
    4) 가장 큰 컨투어(색 덩어리)를 찾아서
    5) 그 중심점, 바운딩 박스 크기 → 거리, 중심과의 오프셋을 계산해서 돌려줌
    """

    # 원본 프레임의 높이/너비
    h, w, _ = frame.shape

    # 화면 중앙 좌표 (전체 프레임 기준)
    center_x, center_y = w // 2, h // 2

    # -----------------------------
    # ROI(Region of Interest) 설정
    # 화면 전체에서 찾으면 노이즈도 많고 정확도 떨어질 수 있으니,
    # 화면 가운데 30%~70% 구간만 본다는 의미
    # -----------------------------
    roi_x1, roi_y1 = int(w * 0.3), int(h * 0.3)  # 좌상단
    roi_x2, roi_y2 = int(w * 0.7), int(h * 0.7)  # 우하단

    # 실제 ROI 이미지 잘라오기
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    # 디버깅을 위해 ROI 영역을 화면에 표시 (녹색 사각형)
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

    # 화면 중앙에도 십자 마커 그리기 (로봇이 맞출 기준점)
    cv2.drawMarker(frame, (center_x, center_y), (0, 255, 0),
                   cv2.MARKER_CROSS, 15, 2)

    # ROI를 HSV로 변환 (색 검출은 HSV가 더 안정적)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # -----------------------------
    # 색상 범위 사전
    # 필요에 따라 여기 추가(orange, purple 등)
    # -----------------------------
    color_ranges = {
        "red":    [(0, 120, 70),  (10, 255, 255)],
        "green":  [(35, 80, 40),  (85, 255, 255)],
        "blue":   [(100, 80, 40), (140, 255, 255)],
        "yellow": [(20, 100, 100), (35, 255, 255)],
    }

    # 만약 사용자가 지정한 색이 위에 없으면 그냥 빈 결과 반환
    if target_color not in color_ranges:
        return frame, []

    # 선택된 색 범위 가져오기
    lower, upper = color_ranges[target_color]

    # 색 범위에 해당하는 마스크 생성
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

    # 마스크에서 외곽선(컨투어) 찾기
    # RETR_EXTERNAL: 가장 바깥쪽 것만
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # 최종 결과를 담을 리스트
    # (색이름, (cx, cy), distance, offset_x, offset_y) 형태로 넣을 예정
    detected_info = []

    # 거리 계산용 상수
    # KNOWN_WIDTH: 실제 물체의 폭(cm)
    # FOCAL_LENGTH: 카메라 초점거리 (테스트값)
    KNOWN_WIDTH, FOCAL_LENGTH = 2.5, 620

    # 컨투어가 하나라도 있다면
    if contours:
        # 가장 큰 컨투어만 사용 (가장 가까이 있거나 가장 확실한 물체라고 가정)
        c = max(contours, key=cv2.contourArea)

        # 너무 작은 컨투어는 노이즈이므로 무시 (영역이 300px 이상일 때만 진행)
        if cv2.contourArea(c) > 300:
            # -----------------------------
            # 회전 사각형(물체 모양에 핏하게)
            # -----------------------------
            rect = cv2.minAreaRect(c)  # ((cx, cy), (w, h), angle)
            (cx, cy), (w_rect, h_rect), angle = rect

            # 회전된 박스 좌표 계산
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # ROI 오프셋 보정 (ROI는 프레임의 일부이므로 전체 좌표계로 변환)
            box[:, 0] += roi_x1
            box[:, 1] += roi_y1

            # 회전 사각형 그리기 (빨강)
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

            # 중심점 (전체 프레임 기준)
            cx = int(cx) + roi_x1
            cy = int(cy) + roi_y1

            # 중심점 표시
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

            # 각도 보정 (-90~90)
            if w_rect < h_rect:
                angle += 90
            angle = round(angle, 2)

            # 거리 계산
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w_rect if w_rect != 0 else 0

            # 텍스트 표시
            cv2.putText(frame, f"{target_color} {distance:.1f}cm  ang={angle:.1f}",
                        (cx - 70, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 결과 추가
            detected_info.append((target_color, (cx, cy), distance, angle))
            # 회전 각도 보정 (-90~0 범위)
            angle = rect[2]
            if rect[1][0] < rect[1][1]:
                angle += 90
            angle = round(angle, 2)

            # 중심 좌표 (ROI 기준 → 전체 프레임 기준)
            cx = int(rect[0][0]) + roi_x1
            cy = int(rect[0][1]) + roi_y1

            # 거리 추정
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / rect[1][0] if rect[1][0] != 0 else 0

            # 시각화 (텍스트)
            cv2.putText(
                frame,
                f"{target_color} {distance:.1f}cm angle={angle:.1f}",
                (cx - 80, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # 결과 저장
            detected_info.append((target_color, (cx, cy), distance, angle))


    # 처리된 프레임(시각화 포함), 검출 정보 반환
    return frame, detected_info


# ======================================================================
# 4. 카메라 스레드
#    - 메인 스레드와 별도로 카메라를 계속 읽으면서 색을 찾음
#    - 찾으면 좌표 변환하고 JSON 저장
# ======================================================================
def camera_capture_thread(stop_event, frame_container):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ 카메라를 열 수 없습니다.")
        return

    print("📷 카메라 스레드 시작 (프레임 송출 중...)")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_container["frame"] = frame  # 최신 프레임 공유

    cap.release()
    print("📷 카메라 스레드 종료")



def move_and_wait(mc, target, speed=20, mode=0, tol=30.0):
    """
    로봇이 목표 좌표에 도달할 때까지 대기
    tol: 허용 오차 (mm)
    """
    time.sleep(0.5)
    mc.send_coords(target, speed, mode)
    time.sleep(0.5)
    while True:
        cur = mc.get_coords()  # 현재 좌표 [x,y,z,rx,ry,rz]
        if cur and all(abs(c - t) < tol for c, t in zip(cur[:3], target[:3])):
            break
        time.sleep(0.2)
    print(f"✅ 이동 완료 → {target}")

# ======================================================================
# 5. 로봇을 미리 정의한 포즈로 이동시키는 간단한 함수
# ======================================================================
def move_to(mc, name, speed=DEFAULT_SPEED):
    """
    이름으로 정의된 POSES 좌표로 이동하고, 완료까지 대기
    """
    if name not in POSES:
        print(f"⚠️ Unknown pose: {name}")
        return

    target = POSES[name]
    print(f"➡️ Move: {name} → {target}")
    move_and_wait(mc, target, speed, mode=1)



# ======================================================================
# 6. 메인 루프
#    - 실행 옵션 파싱
#    - 로봇 연결 및 홈 포즈 이동
#    - 카메라 스레드 시작
#    - 화면 표시
# ======================================================================
def main():
    # ----------------------------------------
    # 1) 명령줄 인자 파싱
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--color", type=str, default="blue")
    args = parser.parse_args()

    # ----------------------------------------
    # 2) 이전 피킹 데이터 삭제
    # ----------------------------------------
    if os.path.exists("picking_target.json"):
        os.remove("picking_target.json")
        print("🧹 이전 picking_target.json 삭제 완료")

    # ----------------------------------------
    # 3) 카메라 스레드 준비
    # ----------------------------------------
    frame_container = {"frame": None}
    stop_event = threading.Event()

    # ----------------------------------------
    # 4) 로봇 연결
    # ----------------------------------------
    mc = CobotClass(args.port, args.baud)
    time.sleep(0.5)
    mc.power_on()
    print("🔌 Power ON 완료")

    # ----------------------------------------
    # 5) 홈 포즈로 이동 (픽엄위치)
    # ----------------------------------------
    print("🏠 홈 위치로 이동 중...")
    move_to(mc, "Home", args.speed)
    # 그리퍼 예시
    mc.set_gripper_mode(0)    
    mc.set_electric_gripper(0)
    mc.set_gripper_value(0, 20, 1)    # 100 = 완전 열림

    # ----------------------------------------
    # 6) 카메라 스레드 시작 (프레임만 송출)
    # ----------------------------------------
    cam_thread = threading.Thread(
        target=camera_capture_thread,
        args=(stop_event, frame_container),
        daemon=True
    )
    cam_thread.start()


    # ----------------------------------------
    # 7) 메인 루프 (ROI 내 물체 감지 후 자동 저장)
    # ----------------------------------------
    print("✅ 메인 루프 시작 (q: 종료, ROI 감지 후 3초 자동 실행)")

    roi_detect_start = None       # ROI 안에서 물체 감지를 시작한 시각
    DETECT_HOLD_TIME = 3.0        # 3초 연속 감지되면 실행
    PIXEL_TO_MM = 0.4             # 픽셀→mm 변환 비율 (실험 필요)

    while not stop_event.is_set():
        frame = frame_container.get("frame")
        if frame is None:
            continue

        # ROI 표시
        h, w, _ = frame.shape
        roi_x1, roi_y1 = int(w * 0.3), int(h * 0.3)
        roi_x2, roi_y2 = int(w * 0.7), int(h * 0.7)
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        cv2.drawMarker(frame, (w // 2, h // 2), (0, 255, 0),
                    cv2.MARKER_CROSS, 15, 2)

        # 색상 감지 수행
        processed_frame, detected = detect_color_and_distance(frame.copy(), args.color)

        # 감지된 물체가 ROI 안에 있는지 확인
        in_roi = False
        if detected:
            _, (cx, cy), _, angle = detected[0]
            if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                in_roi = True

        # ROI 내 감지 타이머 처리
        if in_roi:
            if roi_detect_start is None:
                roi_detect_start = time.time()
                print("🔵 ROI 감지 시작 (3초 유지 시 자동 실행)")
            else:
                elapsed = time.time() - roi_detect_start
                cv2.putText(frame, f"감지 중... {elapsed:.1f}s", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                if elapsed >= DETECT_HOLD_TIME:
                    print("🟢 3초 유지 확인 → 좌표 계산 및 저장 시작")
                    roi_detect_start = None  # 타이머 초기화

                    # -----------------------------
                    # 체커보드 기반 3D 좌표 변환
                    # -----------------------------
                    color_name, (cx, cy), dist, angle = detected[0]

                    fs = cv2.FileStorage("/home/vboxuser/robotarm/camera_info.yaml", cv2.FILE_STORAGE_READ)
                    if not fs.isOpened():
                        print("❌ camera_info.yaml 파일을 열 수 없습니다.")
                        continue

                    camera_matrix = fs.getNode("camera_matrix").mat()
                    dist_coeffs = fs.getNode("distortion_coefficients").mat()
                    fs.release()

                    uv_point = np.array([[cx, cy]], dtype=np.float32)
                    undistorted = cv2.undistortPoints(uv_point, camera_matrix, dist_coeffs, None, camera_matrix)
                    Xc, Yc, Zc = undistorted[0][0][0], undistorted[0][0][1], 0.0

                    # 픽셀 → mm 변환
                    Xc_mm = Xc * PIXEL_TO_MM
                    Yc_mm = Yc * PIXEL_TO_MM
                    Zc_mm = Zc * PIXEL_TO_MM

                    # 카메라 → 로봇 기준 오프셋
                    R_cam2robot = np.eye(3)
                    t_cam2robot = np.array([[120.0], [0.0], [30.0]])
                    cam_point = np.array([[Xc_mm], [Yc_mm], [Zc_mm]])
                    robot_point = R_cam2robot @ cam_point + t_cam2robot

                    # 좌표 계산
                    coord_data = {
                        "x": float(robot_point[0][0]),
                        "y": float(robot_point[1][0]),
                        "z": float(robot_point[2][0])
                    }

                    # 안전 범위 제한
                    safe_x = max(min(coord_data["x"], 350), -350)
                    safe_y = max(min(coord_data["y"], 350), -350)
                    safe_z = max(min(coord_data["z"], 350), -350)

                    # 기본 자세 + 회전 보정 (Rz)
                    base_coords = [safe_x, safe_y, safe_z, 180.0, 0.0, 90.0]
                    new_rz = base_coords[5] + angle
                    new_rz = max(min(new_rz, 180), -180)  # 안전 제한
                    target_coords = base_coords.copy()
                    target_coords[5] = new_rz

                    print(f"🎯 감지된 회전각 angle={angle:.2f}° → Rz={new_rz:.2f}° 적용")
                    print(f"🤖 이동 좌표: {target_coords}")

                    
                    # 로봇 이동
                    # mc.send_coords(target_coords, args.speed, mode=1)

                    # JSON으로 저장 (6축 전체)
                    coord_data = {
                        "x": target_coords[0],
                        "y": target_coords[1],
                        "z": target_coords[2],
                        "rx": target_coords[3],
                        "ry": target_coords[4],
                        "rz": target_coords[5]
                    }
                    #테스트
                    mc.send_coords([59.8, -215.9, 354.6, -175.33, 8.65, coord_data["rz"]], args.speed, mode=1)
                    time.sleep(3)
                    exit()
                    ##

                    save_pick_coordinate(coord_data)

                    print(f"✅ 감지 결과 저장 완료 → {coord_data}")
                    time.sleep(3)
                    print("✅ 로봇 이동 완료, 다시 ROI 감지 대기 중...\n")

        else:
            # ROI 밖으로 나가면 타이머 초기화
            roi_detect_start = None

        # 화면 출력
        cv2.imshow("Camera View", processed_frame)

        # q 키 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    # ----------------------------------------
    # 8) 종료 처리
    # ----------------------------------------
    stop_event.set()
    cam_thread.join()
    cv2.destroyAllWindows()
    print("🔒 종료")

    # ----------------------------------------
    # 9) place 기본 이동 (픽엄위치)
    # ----------------------------------------
    # 플레이스 기본 위치
    print("플레이스 기본 위치로 이동 중...")
    move_to(mc, "Place", args.speed)

    #박스1 상단
    print("플레이스 기본 위치로 이동 중...")
    move_to(mc, "Box3_up", args.speed)
    
    #박스1
    print("플레이스 기본 위치로 이동 중...")
    move_to(mc, "Box3", args.speed)

# ======================================================================
# 7. 파이썬 스크립트 엔트리포인트
# ======================================================================
if __name__ == "__main__":
    main()
