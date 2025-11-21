# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 (pymycobot)
[YOLOv8-Seg + 카메라보정 + 스레드 + 자동종료 + 좌표/각도저장 v9.0]

📌 전체 순서
-------------------------------------------------
1️⃣ 카메라 스레드: 프레임 송출만 수행
2️⃣ 메인 루프: ROI 내 YOLO-Seg 감지 → 3초 유지 시
3️⃣ 좌표 계산(pixel_to_robot) + 각도 저장
4️⃣ 카메라 종료 → 로봇 이동 (Home→Pick(각도보정)→Place→Home)
"""

import threading
import cv2
import time
import argparse
import numpy as np
import json
import os
from ultralytics import YOLO

# ======================================================
# 0️⃣ 로봇 클래스 로드
# ======================================================
try:
    from pymycobot.mycobot320 import MyCobot320 as CobotClass
except Exception:
    from pymycobot.mycobot import MyCobot as CobotClass


# ======================================================
# 1️⃣ 포즈 정의
# ======================================================
POSES = {
    "Home":  [59.8, -215.9, 354.6, -175.33, 8.65, 86.68],
    "Clear": [264.0, -1.0, 379.0, -153, 11, -106],
    "Place": [333.0, 11.0, 170.0, -175, -0.08, -89.0],
}
DEFAULT_SPEED = 20



# ======================================================
# 2️⃣ 카메라 보정값 로드
# ======================================================
def load_camera_params(yaml_path="/home/vboxuser/robotarm/camera_info.yaml"):
    # 윈도우 경로 예시: r"C:\Users\peo00\camera_info.yaml"
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"❌ '{yaml_path}' 파일을 열 수 없습니다.")
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    fs.release()
    print("📷 카메라 보정 파라미터 로드 완료")
    return camera_matrix, dist_coeffs


# ======================================================
# 3️⃣ 픽셀 → 로봇 좌표 변환 (오프셋 포함)
# ======================================================
def pixel_to_robot(cx, cy, distance_cm, camera_matrix, dist_coeffs):
    pts = np.array([[[cx, cy]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=None)
    norm_x, norm_y = undistorted[0, 0]

    # 깊이 계산 (cm → mm)
    scale_z = distance_cm * 10.0
    x_cam = norm_x * scale_z
    y_cam = norm_y * scale_z

    # ----------------------------------------
    # 📏 오프셋 (테스트 기준)
    # ----------------------------------------
    TCP_BASE_OFFSET_X = 59.8
    TCP_BASE_OFFSET_Y = -215.9
    TCP_BASE_OFFSET_Z = 354.6
    CAMERA_TO_TCP_OFFSET_X = 75   # ← 카메라가 X방향으로 90mm 앞에 있음
    CAMERA_TO_TCP_OFFSET_Y = 0.0
    CAMERA_TO_TCP_OFFSET_Z = 170.0  # ← 실제 높이 차이 (현재는 사용 안 함)

    # ----------------------------------------
    # 로봇 좌표 계산
    # ----------------------------------------
    robot_x = TCP_BASE_OFFSET_X + CAMERA_TO_TCP_OFFSET_X + y_cam
    robot_y = TCP_BASE_OFFSET_Y + CAMERA_TO_TCP_OFFSET_Y + x_cam

    # Z는 현재 테스트에서는 고정 (움직이지 않음)
    robot_z = TCP_BASE_OFFSET_Z   # scale_z 적용 안 함

    return {"x": round(robot_x, 2), "y": round(robot_y, 2), "z": round(robot_z, 2)}



# ======================================================
# 4️⃣ YOLO 감지 함수 (Segmentation 기반)
# ======================================================
def detect_yolo(model, frame):
    """
    YOLOv8 Segmentation 모델을 사용하여 객체를 감지하고,
    마스크로부터 회전 각도(minAreaRect)를 계산합니다.
    """
    # stream=True 대신 predict 사용, device=0 (GPU) 설정
    results = model.predict(frame, imgsz=640, conf=0.6, verbose=False, device='cpu')
    
    frame_vis = frame.copy()
    detected_info = []
    FIXED_DISTANCE_CM = 30.0 # 고정 깊이 값 (필요시 수정)

    # 결과가 있고, 마스크 정보가 있는지 확인
    if len(results) > 0 and results[0].masks is not None:
        masks = results[0].masks
        confs = results[0].boxes.conf.cpu().numpy()

        # 감지된 모든 마스크에 대해 반복
        for i, mask in enumerate(masks.data):
            conf = confs[i]
            
            # 마스크 데이터를 (H, W) 형태로 변환
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            
            # 원본 프레임 크기로 리사이즈
            mask_np = cv2.resize(mask_np, (frame_vis.shape[1], frame_vis.shape[0]))

            # 🔹 마스크에서 외곽선(Contour) 추출
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue

            # 🔹 가장 큰 컨투어 선택
            cnt = max(contours, key=cv2.contourArea)
            
            # 🔹 최소 영역 사각형(회전된 사각형) 계산
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect

            # 🔹 회전 박스 시각화
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame_vis, [box], 0, (255, 255, 0), 2) # 청록색

            # 🔹 각도 표시
            cv2.putText(frame_vis, f"Angle: {angle:.1f}", (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 🔹 양품 / 불량품 판단 (신뢰도 기준)
            if conf >= 0.9:
                label = f"양품 ({conf:.2f})"
                color = (0, 255, 0) # 초록색
            else:
                label = f"불량품 ({conf:.2f})"
                color = (0, 0, 255) # 빨간색

            cv2.putText(frame_vis, label, (int(cx - 50), int(cy - 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 🔹 반환할 정보 추가 (cx, cy, 고정거리, 각도)
            # 메인 루프는 첫 번째 객체만 사용하므로, 모두 추가
            detected_info.append(("object", (int(cx), int(cy)), FIXED_DISTANCE_CM, angle))

    # 시각화된 프레임과 감지 정보 리스트 반환
    return frame_vis, detected_info


# ======================================================
# 5️⃣ 카메라 스레드 (프레임 송출만)
# ======================================================
def camera_capture_thread(stop_event, frame_container):
    # ⭐️ 사용자 요청에 따라 카메라 ID 1번으로 변경
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("⚠️ 카메라를 열 수 없습니다. (ID: 1)")
        return
    
    # ⭐️ 카메라 해상도 설정 (사용자 예제 코드 기준)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("📷 카메라 스레드 시작 (프레임 송출 중...)")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame_container["frame"] = frame
    cap.release()
    print("📷 카메라 스레드 종료")


# ======================================================
# 6️⃣ 로봇 이동 헬퍼
# ======================================================
def move_to(mc, name, speed=DEFAULT_SPEED):
    if name not in POSES:
        print(f"⚠️ Unknown pose: {name}")
        return
    target = POSES[name]
    mc.send_coords(target, speed, 0)
    time.sleep(2)
    print(f"✅ Move → {name}")


# ======================================================
# 7️⃣ 좌표 JSON 저장
# ======================================================
def save_pick_coordinate(coord, angle, filename="picking_target.json"):
    data_to_save = {
        "coordinates": coord,
        "angle": round(angle, 2)
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)
    print(f"💾 좌표/각도 저장 완료 → {filename} : {data_to_save}")


# ======================================================
# 8️⃣ 메인 루프
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--speed", type=int, default=20)
    # ⭐️ 사용자 요청에 따라 Seg 모델 경로로 기본값 변경
    parser.add_argument("--model", type=str, 
                        default="/home/young/Downloads/best.pt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    detected_angle = None # 최종 확정된 각도를 저장할 변수

    # YOLO 모델 로드
    print(f"🧠 YOLO 모델 로드 중: {args.model}")
    model = YOLO(args.model)
    model.to("cpu")
    print("✅ YOLO 모델 로드 완료")

    # 카메라 보정값 로드 (경로 확인 필수)
    try:
        camera_matrix, dist_coeffs = load_camera_params(r"/home/young/Downloads/camera_info.yaml") # 윈도우 경로 예시
    except FileNotFoundError as e:
        print(e)
        print("🔴 camera_info.yaml 경로를 확인하세요. 프로그램을 종료합니다.")
        return

    # 로봇 연결
    mc = None
    if not args.dry_run:
        try:
            mc = CobotClass(args.port, args.baud)
            time.sleep(0.5)
            mc.power_on()
            print("🔌 Power ON 완료")
            move_to(mc, "Home", args.speed)
            mc.set_gripper_mode(0)
            mc.set_electric_gripper(0)
            mc.set_gripper_value(0, 20, 1)  # 열림
        except Exception as e:
            print(f"🔴 로봇 연결 실패: {e}")
            print("🔴 포트({args.port})를 확인하거나 --dry-run 옵션을 사용하세요.")
            return
    else:
        print("🟡 dry-run 모드 (로봇 미연결)")

    # 카메라 스레드 시작
    frame_container = {"frame": None}
    stop_event = threading.Event()
    cam_thread = threading.Thread(
        target=camera_capture_thread, args=(stop_event, frame_container), daemon=True
    )
    cam_thread.start()

    print("✅ 메인 루프 시작 (ROI 감지 후 3초 유지 시 실행)")
    roi_detect_start = None
    DETECT_HOLD_TIME = 3.0
    detected_coord = None

    try:
        while not stop_event.is_set():
            frame = frame_container.get("frame")
            if frame is None:
                time.sleep(0.01) # 카메라가 켜지는 중
                continue

            # ROI 표시
            h, w, _ = frame.shape
            roi_x1, roi_y1 = int(w * 0.3), int(h * 0.3)
            roi_x2, roi_y2 = int(w * 0.7), int(h * 0.7)
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
            cv2.drawMarker(frame, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)

            # ⭐️ YOLO 감지 수행 (Segmentation 기반)
            processed_frame, detected = detect_yolo(model, frame)
            
            in_roi = False
            
            # 감지 결과 있을 때 (첫 번째 객체만 처리)
            if detected:
                # ⭐️ detect_yolo가 bbox 대신 angle을 반환
                # _, (cx, cy), dist, bbox = detected[0] (기존)
                _, (cx, cy), dist, angle_from_yolo = detected[0] # (변경)

                if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                    in_roi = True

                    # -------------------------------
                    # 🔸 YOLO Segmentation 기반 각도 사용
                    # -------------------------------
                    # (기존의 bbox 기반 각도 추정 로직 삭제)
                    # ⭐️ detect_yolo에서 계산된 각도를 바로 사용
                    if angle_from_yolo is not None:
                        detected_angle = angle_from_yolo
                else:
                    # ROI 밖에 있으면 각도 초기화
                    detected_angle = None
            else:
                 detected_angle = None


            # ROI 내부에 감지된 경우 (3초 유지 시 좌표 확정)
            if in_roi:
                if roi_detect_start is None:
                    roi_detect_start = time.time()
                    print("🔵 ROI 감지 시작 (3초 유지 시 좌표 확정)")
                else:
                    elapsed = time.time() - roi_detect_start
                    cv2.putText(processed_frame, f"감지 중... {elapsed:.1f}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    
                    if elapsed >= DETECT_HOLD_TIME:
                        print("🟢 감지 유지 3초 → 좌표 계산 시작")
                        detected_coord = pixel_to_robot(cx, cy, dist, camera_matrix, dist_coeffs)
                        
                        # ⭐️ 최종 좌표와 '각도'를 함께 출력
                        print(f"🎯 물체 좌표: {detected_coord}, 각도: {detected_angle}") 
                        
                        # (주석 처리된 그리퍼/회전보정 로직은 그대로 둠)
                        
                        # ... (기존 그리퍼 동작 코드) ...
                        if not args.dry_run and mc:
                            try:
                                mc.set_gripper_state(0, 80)   # 완전 열기
                                mc.set_gripper_state(1, 80)   # 완전 열기
                                print("🤖 그리퍼 동작 완료")
                            except Exception as e:
                                print(f"⚠️ 그리퍼 동작 중 오류 발생: {e}")

                        # -------------------------------
                        # ✅ 감지 완료 후 카메라 종료
                        # -------------------------------
                        stop_event.set()
                        print("📷 카메라 종료 요청...")
                        break # while 루프 탈출
            else:
                # ROI 밖으로 벗어나면 타이머 초기화
                roi_detect_start = None


            cv2.imshow("Camera View (YOLOv8-Seg)", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    finally:
        print("🔄 최종 정리 중...")
        stop_event.set()
        if cam_thread.is_alive():
            cam_thread.join()
        cv2.destroyAllWindows()
        print("✅ 모든 창 종료")

    # ==================================================
    # ✅ 감지된 좌표가 있으면 로봇 이동
    # ==================================================
    if detected_coord:
        # ⭐️ 좌표와 각도를 JSON 파일로 저장
        save_pick_coordinate(detected_coord, detected_angle if detected_angle is not None else 0.0)

        print("🤖 로봇 이동 시작...")
        if not args.dry_run and mc:
            mc.set_gripper_mode(0)
            mc.set_electric_gripper(0)
            
            base_r = -175.33
            base_p = 8.65
            base_y = 86.68 # Home 포즈의 6축(Yaw) 값

            # ⭐️ 감지된 각도(detected_angle)를 로봇 6축(Yaw)에 반영
            # (주의) angle 값의 범위(0~90, 0~-90)와 로봇 회전 방향(+, -)을 테스트하여
            # yaw_offset 보정값(예: 0.35)을 튜닝해야 합니다.
            # cv2.minAreaRect의 각도는 복잡하므로 테스트가 필수입니다.
            yaw_offset = (detected_angle if detected_angle is not None else 0.0)
            
            # 예: 각도가 0~90 사이 값만 나온다면, 90도(수직)일 때 0으로 보정
            if yaw_offset > 45: # 90도에 가까울수록
                 yaw_offset = yaw_offset - 90 
            
            # (보정 계수 튜닝 필요)
            yaw_correction = yaw_offset * 1.0 # 예시: 1.0으로 설정
            
            wrist_yaw = base_y + yaw_correction   # 📌 YOLO 각도 반영
            
            print(f"🧭 Wrist 회전 적용: base_y={base_y:.1f}, angle={detected_angle:.1f}, correction={yaw_correction:.1f} → 최종={wrist_yaw:.1f}")


            mc.set_gripper_value(50, 20, 1)  # 열림

            # 1. 위에서 접근 (Z = 300)
            mc.send_coords(
                [detected_coord["x"], detected_coord["y"], 300.0,
                base_r, base_p, wrist_yaw], # ⭐️ 각도(wrist_yaw) 적용
                25, 0
            )
            time.sleep(3)

            # 2. 내려가서 집기 (Z = 260+40) -> 300? (값 확인 필요)
            # (Z값을 실제 환경에 맞게 수정하세요)
            pick_z = 260.0 + 40.0 # 예시 Z
            mc.send_coords(
                [detected_coord["x"], detected_coord["y"], pick_z,
                base_r, base_p, wrist_yaw], # ⭐️ 각도(wrist_yaw) 적용
                15, 0
            )
            time.sleep(2)

            mc.set_gripper_value(8, 20, 1)  # 닫힘
            time.sleep(1.5)

            # 3. 위로 빼기 (Z = 260+100) -> 360?
            mc.send_coords(
                [detected_coord["x"], detected_coord["y"], pick_z + 60.0,
                base_r, base_p, wrist_yaw], # ⭐️ 각도(wrist_yaw) 적용
                15, 0
            )
            time.sleep(1.5)
            
            # 4. 이동 및 복귀
            move_to(mc, "Clear", args.speed)
            move_to(mc, "Place", args.speed)
            mc.set_gripper_state(0, 80) # 그리퍼 열기
            time.sleep(1)
            move_to(mc, "Home", args.speed)
        else:
            print(f"🟢 [dry-run] 좌표({detected_coord}) 및 각도({detected_angle:.1f}) 기반 시뮬레이션 완료")

    if mc:
        mc.power_off()
        print("🔌 Power OFF")
    print("🔒 종료 완료")


if __name__ == "__main__":
    main()