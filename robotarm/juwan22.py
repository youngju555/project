import cv2
import csv
import time
from ultralytics import YOLO
try:
    from pymycobot.mycobot320 import MyCobot320 as CobotClass
except:
    from pymycobot.mycobot import MyCobot as CobotClass

# ===================== 설정 =====================
MODEL_PATH = "/home/young/Downloads/best.pt"  # YOLO 모델 경로
ROBOT_PORT = "/dev/ttyACM0"  # 로봇 포트
ROBOT_BAUD = 115200  # 통신 속도
CSV_NAME = "calibration_pairs.csv"

# ===================== 모델/로봇 초기화 =====================
model = YOLO(MODEL_PATH)
mc = CobotClass(ROBOT_PORT, ROBOT_BAUD)
mc.power_on()
time.sleep(1)
cap = cv2.VideoCapture(0)

# ===================== CSV 파일 생성 =====================
csv_file = open(CSV_NAME, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow(["cx", "cy", "robot_x", "robot_y"])

# ===================== 상태 변수 =====================
step = 1  # 1=픽셀저장, 2=Power OFF 대기, 3=로봇좌표 저장
temp_cx, temp_cy = None, None
print("\n==============================================")
print(" :렌치: 픽셀 → 로봇 좌표 보정 캘리브레이션 모드 (v3)")
print("----------------------------------------------")
print("● ENTER 1 → YOLO 픽셀 좌표 저장 (cx,cy)")
print("● ENTER 2 → 로봇 POWER OFF (손으로 움직여 위치 맞추기)")
print("● ENTER 3 → 로봇 POWER ON + 실제 좌표 저장")
print("● ESC → 종료")
print("==============================================\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    # YOLO 감지
    results = model.predict(frame, conf=0.5, verbose=False)
    boxes = results[0].boxes
    cx, cy = None, None
    if len(boxes) > 0:
        box = max(boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = box.xyxy[0]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        # 화면 표시
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(frame, f"cx={cx}, cy={cy}",
                    (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"STEP: {step}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Calibration Camera", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    # ===================== STEP 1: 픽셀 좌표 저장 =====================
    if key == 13 and step == 1:
        if cx is not None:
            temp_cx = cx
            temp_cy = cy
            step = 2
            print(f"\n:압정: STEP1 완료 → 픽셀 저장됨: cx={cx}, cy={cy}")
            print(":오른쪽_화살표: 이제 로봇 POWER OFF 됨. 손으로 박스 중앙 위로 움직여주세요.\n")
            # 로봇 Power OFF
            mc.power_off()
        else:
            print(":x: YOLO 감지 실패! 박스를 카메라가 보게 해주세요.\n")
    # ===================== STEP 2: 로봇 수동 이동 (손으로 조작) =====================
    elif key == 13 and step == 2:
        print(":압정: STEP2 완료 → 로봇 손으로 이동 완료로 간주.")
        print(":오른쪽_화살표: 이제 로봇 POWER ON 후 좌표를 읽습니다.\n")
        # 로봇 Power ON
        mc.power_on()
        time.sleep(1.0)
        # 로봇 암의 힘을 상시 끄도록 설정
        mc.power_off()  # 상시 OFF 상태로 유지
        step = 3
    # ===================== STEP 3: 로봇 좌표 저장 =====================
    elif key == 13 and step == 3:
        coords = mc.get_coords()
        if coords:
            robot_x = coords[0]
            robot_y = coords[1]
            writer.writerow([temp_cx, temp_cy, robot_x, robot_y])
            print(f":흰색_확인_표시: 1세트 저장 완료: ({temp_cx}, {temp_cy}) → ({robot_x:.2f}, {robot_y:.2f})\n")
        else:
            print(":x: 로봇 좌표 읽기 실패!\n")
        # 다음 세트 준비
        step = 1
        print(":오른쪽_화살표: 다음 박스 위치로 이동 후 STEP1부터 다시 시작하세요.\n")

cap.release()
csv_file.close()
cv2.destroyAllWindows()
print("\n==============================================")
print(f" :파일_폴더: 데이터 저장 완료: {CSV_NAME}")
print(" 이 파일을 나에게 보내주면 → 보정 공식(픽셀→로봇 XY) 만들어줄게!")
print("==============================================\n")
