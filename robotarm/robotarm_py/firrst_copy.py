# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 (pymycobot)
[개선 버전 v5.8 + 주석 확장판]

📌 이번 버전의 핵심 변경점
----------------------------------------------------
1. (카메라 충돌 해결)
   - 카메라를 스레드 안에서 열면, 가상머신/리눅스 환경에 따라
     VideoCapture가 스레드와 충돌하는 경우가 있음.
   - 그래서 'main()'에서 카메라를 먼저 열고, 열린 cap 객체를
     카메라 스레드에 넘겨주는 구조로 수정.

2. (카메라 자원 정리 위치 이동)
   - cap.release()를 스레드 안이 아니라 main()의 finally에서 한 번만 호출.
   - 이렇게 해야 프로그램 종료 시점이 명확해지고, OS 자원도 깔끔히 반납됨.

3. (파란색 오검출 완화)
   - 바닥 전체가 파란색으로 인식되는 문제를 줄이기 위해
     HSV 범위의 S, V 최소값을 50 → 100으로 올림.
   - 실제 환경에 따라 (90, 90, 90) ~ (90, 120, 120) 사이에서 조정하면 됨.

4. (주석 보강)
   - 각 단계가 왜 필요한지, UiPath나 로봇 쪽에 연결할 때 어디를 손대야 하는지
     이해할 수 있도록 세부 주석을 추가.
"""

import threading
import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 0. 로봇 클래스 불러오기
#    - 실제 장비가 MyCobot 320 M5인 경우: pymycobot.mycobot320
#    - 일부 환경에서는 pymycobot.mycobot 만 있는 경우가 있어서 try/except
# ---------------------------------------------------------------------------
try:
    from pymycobot.mycobot320 import MyCobot320 as CobotClass
except Exception:
    from pymycobot.mycobot import MyCobot as CobotClass

# ---------------------------------------------------------------------------
# 1. 전역 변수와 Lock
#    - 카메라 스레드가 "목표 좌표를 찾았다"는 신호를 주면
#      메인 스레드가 그걸 읽어서 로봇을 움직여야 함
#    - 스레드 사이에서 같은 변수를 건드리므로 Lock 필요
# ---------------------------------------------------------------------------
picking_done = False           # 카메라가 목표를 찾으면 True로 바꿈
g_target_coordinate = None     # 카메라가 계산한 {x, y, z_debug} 저장
g_coord_lock = threading.Lock()  # 위 좌표를 안전하게 읽고 쓰기 위한 Lock
args = None                    # argparse 결과를 전역에서도 쓰기 위해

# ---------------------------------------------------------------------------
# 2. 로봇 기본 자세/캘리브레이션 값
#    - 이 값들은 사용자의 실제 로봇 기준으로 이미 한번 잡아둔 값이라고 가정
#    - X/Y 오프셋은 pixel_to_robot()에서 한 번 더 적용됨
# ---------------------------------------------------------------------------
POSES = {
    "Home":  [59.8, -215.9, 354.6, -175.33, 8.65, 86.68],   # 대기 자세
    "Clear_Air_A": [264.0, -1.0, 379.0, -153, 11, -106],    # 중간 이동용
    "Place_B": [333.0, 11.0, 170.0, -175, -0.08, -89.0],    # 내려놓는 자리
}
DEFAULT_SPEED = 20

# 카메라 내부 파라미터 (예시값)
# 실제 카메라 보정값으로 교체하는 걸 강력히 권장
CAMERA_MATRIX = np.array([
    [539.13729067, 0.0, 329.02126026],
    [0.0, 542.34217387, 242.10995541],
    [0.0, 0.0, 1.0]
])
# 왜곡 계수 (이것도 보정값으로 교체 가능)
DIST_COEFFS = np.array([[0.20528603, -0.76664068, -0.00096614, 0.00111892, 0.97630004]])

# ---------------------------------------------------------------------------
# 3. 픽셀 좌표 → 로봇 좌표 변환
#    - 카메라에서 찾은 (cx, cy) 는 화면상의 좌표일 뿐이라서
#      로봇이 이해할 수 있는 X, Y(mm)로 바꿔줘야 함
#    - 여기서는 Z는 고정 높이를 쓰도록 설계했으므로 디버그만 남김
# ---------------------------------------------------------------------------
def pixel_to_robot(cx, cy, distance_cm, frame_w, frame_h):
    """
    cx, cy         : 카메라에서 찾은 물체 중심 픽셀 좌표
    distance_cm    : 탐색 자세에서 물체까지의 대략적인 거리 (cm) - 스케일링용
    frame_w, frame_h: 화면 크기 (안 써도 되지만 남겨둠)
    """
    # 1) 왜곡 보정된 정규화 좌표로 변환
    pts = np.array([[[cx, cy]]], dtype=np.float32)
    undistorted_pts = cv2.undistortPoints(pts, CAMERA_MATRIX, DIST_COEFFS, P=None)
    norm_x, norm_y = undistorted_pts[0, 0]   # 왜곡 보정된 카메라 좌표계 (단위: 상대좌표)

    # 2) 거리(cm)를 mm로 바꿔서 실제 위치로 스케일링
    scale_z = distance_cm * 10.0            # cm → mm
    x_cam = norm_x * scale_z                # 카메라 좌표계 X
    y_cam = norm_y * scale_z                # 카메라 좌표계 Y

    # 3) 로봇 TCP(툴 중심) 기준 오프셋들
    #    이 값들은 "탐색할 때 로봇이 서 있던 좌표"와
    #    "카메라가 툴에서 얼마나 떨어져 있는지"를 합친 것
    TCP_BASE_OFFSET_X = 59.8
    TCP_BASE_OFFSET_Y = -215.9

    # ✨ 이 부분이 실제 설치환경 따라 가장 많이 바뀌는 부분
    # 카메라가 그리퍼보다 앞에 100mm 달려있다고 가정
    CAMERA_TO_TCP_OFFSET_X = 90.0
    CAMERA_TO_TCP_OFFSET_Y = 0.0

    # 4) 최종 로봇 좌표
    robot_x = TCP_BASE_OFFSET_X + CAMERA_TO_TCP_OFFSET_X + y_cam
    robot_y = TCP_BASE_OFFSET_Y + CAMERA_TO_TCP_OFFSET_Y + x_cam

    # 5) Z는 고정값을 쓸 거라 디버그용만 남김
    TCP_BASE_OFFSET_Z = 354.6
    robot_z_ignored = TCP_BASE_OFFSET_Z - scale_z

    return {
        "x": round(robot_x, 2),
        "y": round(robot_y, 2),
        "z_debug": round(robot_z_ignored, 2)
    }

# ---------------------------------------------------------------------------
# 4. 색상 검출 함수
#    - 프레임 하나 받아서 "파란색 물체가 어디 있냐"를 찾아줌
#    - 여기서 HSV 범위를 너무 느슨하게 잡으면 바닥까지 파란색으로 잡힘
#    - 그래서 S, V 최솟값을 100으로 살짝 올려서 "쨍한 파란색"만 잡게 함
# ---------------------------------------------------------------------------
def detect_color_and_distance(frame, target_color="blue"):
    h, w, _ = frame.shape

    # BGR → HSV 변환 (색 검출은 HSV가 훨씬 편함)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 색상 범위 사전
    # ▶ 여기 숫자만 조정해도 인식 품질이 많이 달라짐
    color_ranges = {
        # 기존: (90, 50, 50) 이라 바닥/그림자도 잡힘
        # 수정: S, V를 100으로 올려서 진짜 파란 큐브만 잡도록 함
        "blue": [(90, 100, 100), (140, 255, 255)],
    }

    # 선택한 색상의 범위 꺼내기
    lower, upper = color_ranges.get(target_color, ((0, 0, 0), (0, 0, 0)))
    lower = np.array(lower)
    upper = np.array(upper)

    # 색영역 마스크 생성
    mask = cv2.inRange(hsv, lower, upper)

    # 노이즈 제거 (작은 점, 빈틈 제거)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # 외곽선(컨투어) 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_info = []

    # 이 예제에서는 실제 거리 대신 "고정 거리"를 쓰므로 True
    USE_FIXED_DISTANCE = True
    FIXED_DISTANCE_CM = 19.0   # 실제로는 카메라→물체 거리를 자로 재서 바꿔주세요

    if contours:
        # 가장 큰 파란색 덩어리 하나만 선택
        c = max(contours, key=cv2.contourArea)

        # 너무 작은 건 노이즈로 간주
        if 400 < cv2.contourArea(c) < 20000:   # 윗값은 바닥이 잡히는 걸 막으려고 넣음
            x, y, w_box, h_box = cv2.boundingRect(c)
            cx, cy = x + w_box // 2, y + h_box // 2

            # 고정 거리 사용
            distance = FIXED_DISTANCE_CM

            detected_info.append((target_color, (cx, cy), distance))

            # 디버그 표시 (원본 프레임 위에 네모, 십자, 텍스트)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 255, 0), 2)
            cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
            cv2.putText(frame, f"{target_color} {distance:.1f}cm",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

    return frame, detected_info

# ---------------------------------------------------------------------------
# 5. 카메라 스레드
#    - main()에서 이미 열어둔 cap 객체를 받아서 계속 프레임만 읽어오는 역할
#    - "안정적으로 3프레임 연속 감지"되면 전역변수에 좌표 기록하고 picking_done = True
# ---------------------------------------------------------------------------
def camera_capture_thread(stop_event, frame_container, cap, model, mc=None, dry_run=False):
    global picking_done, g_target_coordinate, g_coord_lock

    print("📷 카메라 스레드 시작 (YOLO 감지 모드)")
    stable_frames = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # YOLO 감지 수행
        results = model.predict(frame, imgsz=640, conf=0.6, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        frame_container["frame"] = results[0].plot()  # 시각화된 결과

        # 물체가 감지된 경우
        if len(boxes) > 0 and not picking_done:
            stable_frames += 1
            if stable_frames >= 3:  # 3프레임 연속 감지 시 “확정”
                x1, y1, x2, y2 = boxes[0]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                distance_cm = 30.0  # 임시 고정거리 (실측 가능 시 수정)

                print(f"🎯 YOLO 객체 중심: ({cx}, {cy})")

                h, w, _ = frame.shape
                coord = pixel_to_robot(cx, cy, distance_cm, w, h)

                with g_coord_lock:
                    g_target_coordinate = coord
                picking_done = True
                stable_frames = 0
        else:
            stable_frames = 0

        time.sleep(0.05)

    print("📷 카메라 스레드 종료")
# ---------------------------------------------------------------------------
# 6. 메인 루프
#    - 인자 파싱 → 로봇 초기화 → 카메라 열기 → 스레드 시작 → GUI 루프
#    - 종료 시점에서만 cap.release(), mc.power_off() 실행
# ---------------------------------------------------------------------------
def main():
    global g_target_coordinate, g_coord_lock, picking_done, args

    # -----------------------------
    # 1) 실행 인자 파싱
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")  # 로봇 시리얼 포트
    parser.add_argument("--baud", type=int, default=115200)          # 로봇 보레이트
    parser.add_argument("--speed", type=int, default=20)             # 기본 이동 속도
    parser.add_argument("--color", type=str, default="blue")         # 찾을 색
    parser.add_argument("--camera", type=int, default=1)             # 사용할 카메라 번호
    parser.add_argument("--dry-run", action="store_true")            # 로봇 없이 테스트
    parser.add_argument("--model", type=str, default="best.pt", help="/home/vboxuser/robotarm/best.pt")
    args = parser.parse_args()


    # -----------------------------
    # 2) YOLO 모델 로드
    # -----------------------------
    print(f"🧠 YOLOv8 모델('{args.model}') 로드 중...")
    try:
        model = YOLO(args.model, task="detect")
        print("✅ YOLO 모델 로드 성공")
    except Exception as e:
        print(f"❌ YOLO 모델 로드 실패: {e}")
        return
    

    # 로봇 피킹 시 쓸 Z 관련 상수들
    GRIPPER_OFFSET_Z = 18.0
    FIXED_PICK_Z = 278.0
    APPROACH_HEIGHT = 40.0
    # 로봇 손목 각도 (물체 위에서 수직으로 내리꽂기용)
    PICK_RX, PICK_RY, PICK_RZ = -175.33, 8.65, 86.68

    # 카메라 스레드와 공유할 프레임 버퍼
    frame_container = {"frame": None}
    # 스레드 종료 신호
    stop_event = threading.Event()
    # 로봇, 카메라 객체
    mc = None
    cap = None

    try:
        # -----------------------------
        # 2) 로봇 초기화
        # -----------------------------
        if not args.dry_run:
            try:
                mc = CobotClass(args.port, args.baud)
                time.sleep(0.5)
                mc.power_on()
                print("🔌 로봇 Power ON 완료")
                mc.set_gripper_state(0, 80)  # 그리퍼 벌리기
                time.sleep(1)
            except Exception as e:
                print(f"❌ 로봇 연결 실패: {e}")
                # 로봇이 없으면 강제로 dry-run 모드로 전환
                mc = None
                args.dry_run = True
        else:
            print("🟡 dry-run 모드로 시작 (로봇 명령은 출력만 함)")

        # -----------------------------
        # 3) 카메라 초기화 (메인에서!)
        # -----------------------------
        print(f"📷 메인: 카메라 {args.camera}번 열기 시도...")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"⚠️ {args.camera}번 카메라 실패 → 0번으로 재시도")
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 카메라를 열 수 없습니다. 프로그램을 종료합니다.")
            raise Exception("camera open failed")
        print("✅ 메인: 카메라 연결 성공")

        # -----------------------------
        # 4) 카메라 스레드 시작
        # -----------------------------
        cam_thread = threading.Thread(
            target=camera_capture_thread,
            args=(stop_event, frame_container, cap, model, mc, args.dry_run),
            daemon=True
        )
        cam_thread.start()
# -----------------------------
        # 5) 로봇 기본 자세로 이동 -> (루프 안으로 이동!)
        # -----------------------------
        # if not args.dry_run and mc is not None:  <-- (이 부분을 주석 처리)
        #     mc.send_coords(POSES["Home"], args.speed)
        #     time.sleep(3)
        #     print("🏠 홈 위치 도달")

        print("✅ 메인 루프 시작 (창에서 q 누르면 종료)")

        # ✨ [추가] 로봇 상태 플래그
        robot_is_home = False

        # -----------------------------
        # 6) 메인 GUI + 로봇 제어 루프
        # -----------------------------
        while not stop_event.is_set():
            # 6-1) 최신 프레임이 있다면 보여주기
            frame = frame_container.get("frame")
            if frame is not None:
                cv2.imshow("Camera View", frame) # (성공) 이제 창이 바로 뜹니다.

            # 6-2) 사용자가 q를 누르면 전체 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

            # ✨ [추가] 로봇이 홈에 도착하지 않았다면, 먼저 홈으로 보냄 (딱 한 번)
            if not robot_is_home:
                if not args.dry_run and mc is not None:
                    print("🤖 로봇을 홈 위치로 이동합니다... (화면이 3초간 멈춥니다)")
                    mc.send_coords(POSES["Home"], args.speed)
                    time.sleep(3)
                    print("🏠 홈 위치 도달. 탐지를 시작합니다.")
                else:
                    print("🏠 [dry-run] 홈 위치 도달. 탐지를 시작합니다.")
                
                robot_is_home = True # 홈 도착 플래그
                picking_done = False # (매우 중요) 홈으로 가는 동안 감지된 것 초기화
                with g_coord_lock:
                    g_target_coordinate = None
                continue # 이번 루프는 여기까지만 실행

            # 6-3) 카메라가 좌표를 찾아놨다면 그걸 읽어서 로봇 움직이기
            # (중요) 'robot_is_home'이 True일 때만 이 코드가 실행됨
            current_coord = None
            if picking_done: # (robot_is_home이 True가 된 이후에 picking_done이 켜져야 함)
                with g_coord_lock:
                    if g_target_coordinate is not None:
                        current_coord = g_target_coordinate.copy()
                        g_target_coordinate = None

                if current_coord:
                    print(f"🤖 인식 성공 → 로봇 이동 시작: {current_coord}")
                    pick_x = current_coord["x"]
                    pick_y = current_coord["y"]

                    # 고정 Z 로직
                    z_approach = FIXED_PICK_Z + APPROACH_HEIGHT
                    z_grasp = FIXED_PICK_Z - GRIPPER_OFFSET_Z
                    print(f"   ↳ 접근Z={z_approach:.1f}, 잡기Z={z_grasp:.1f}")

                    if not args.dry_run and mc is not None:
                        # 1) 물체 위로 이동
                        mc.set_gripper_state(0, 80)
                        time.sleep(0.01)  # 🧠 GIL 해제 → 카메라 스레드도 돌 수 있게
                        time.sleep(1)
                        mc.send_coords(
                            [pick_x, pick_y, z_approach, PICK_RX, PICK_RY, PICK_RZ],
                            25, 1
                        )
                        time.sleep(0.01)
                        time.sleep(5)

                        # 2) 집기 높이까지 내리기
                        mc.send_coords(
                            [pick_x, pick_y, z_grasp, PICK_RX, PICK_RY, PICK_RZ],
                            15, 1
                        )
                        time.sleep(0.01)
                        time.sleep(1.5)

                        # 3) 집기
                        mc.set_gripper_state(1, 80)
                        print("집기", mc.get_coords)
                        time.sleep(1.5)
                        exit()

                        # 🆙 바로 위로 Z축만 상승 (회전 없이)
                        mc.send_coords(
                            [pick_x, pick_y, z_grasp + 80, PICK_RX, PICK_RY, PICK_RZ],  # 현재 z보다 +80mm 위로
                            25, 1
                        )
                        time.sleep(2)
                        # 4) 다시 위로
                        mc.send_coords(
                            [pick_x, pick_y, z_approach, PICK_RX, PICK_RY, PICK_RZ],
                            25, 1
                        )
                        time.sleep(2)

                        

                        # 5) 중간 지점 → 놓는 자리 → 다시 홈
                        mc.send_coords(POSES["Clear_Air_A"], args.speed, 1)
                        time.sleep(3)
                        mc.send_coords(POSES["Place_B"], args.speed, 1)
                        time.sleep(3)
                        mc.set_gripper_state(0, 80)
                        time.sleep(1.5)
                        mc.send_coords(POSES["Home"], args.speed)
                        time.sleep(3)
                        print("✅ 1회 피킹 완료")
                    else:
                        # 로봇이 없을 때는 흐름만 보여줌
                        print("   [dry-run] 로봇 없이 동작 흐름만 실행")
                        time.sleep(5)

                    # 다음 물체를 다시 찾을 수 있게 리셋
                    picking_done = False

    finally:
        # -----------------------------
        # 7) 종료 처리 (여기가 중요)
        # -----------------------------
        stop_event.set()              # 스레드에 종료 신호
        time.sleep(0.2)
        # 카메라 객체 정리
        if cap:
            cap.release()
        # OpenCV 윈도우 닫기
        cv2.destroyAllWindows()
        # 로봇 전원 끄기
        if mc:
            mc.power_off()
        print("🔒 프로그램 종료")


if __name__ == "__main__":
    main()
