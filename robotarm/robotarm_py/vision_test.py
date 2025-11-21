import cv2
import numpy as np
import argparse
import time

def nothing(x):
    """트랙바 콜백 함수 (아무것도 안 함)"""
    pass

def main():
    parser = argparse.ArgumentParser(description="HSV Color Range Calibrator")
    parser.add_argument("--camera", type=int, default=1, 
                        help="Camera index (e.g., 0, 1, ...)")
    args = parser.parse_args()

    # --- 카메라 열기 ---
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"⚠️ {args.camera}번 카메라 실패. 0번으로 재시도.")
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다. 종료합니다.")
        return
    print(f"✅ {args.camera}번 카메라로 캘리브레이션 시작.")

    # --- 'HSV Calibrator' 컨트롤 창 생성 ---
    cv2.namedWindow("HSV Calibrator")
    cv2.resizeWindow("HSV Calibrator", 600, 300)

    # --- 6개의 트랙바(슬라이더) 생성 ---
    # Hue (색조): 0 ~ 179 (OpenCV의 H 범위)
    cv2.createTrackbar("H_min", "HSV Calibrator", 0, 179, nothing)
    cv2.createTrackbar("H_max", "HSV Calibrator", 179, 179, nothing)
    
    # Saturation (채도): 0 ~ 255
    cv2.createTrackbar("S_min", "HSV Calibrator", 0, 255, nothing)
    cv2.createTrackbar("S_max", "HSV Calibrator", 255, 255, nothing)
    
    # Value (명도): 0 ~ 255
    cv2.createTrackbar("V_min", "HSV Calibrator", 0, 255, nothing)
    cv2.createTrackbar("V_max", "HSV Calibrator", 255, 255, nothing)

    print("\n=======================================================")
    print(" 🚀 캘리브레이션 도구 사용법 (q를 눌러 종료):")
    print(" 1. 큐브(파랑/노랑/빨강)를 카메라 앞에 놓으세요.")
    print(" 2. 'HSV Calibrator' 창의 슬라이더 6개를 조절하세요.")
    print(" 3. 'Mask' 창에 '찾으려는 큐브만' 흰색으로 나오게 만드세요.")
    print(" 4. 값을 찾으면 6개의 숫자를 적어두세요.")
    print(" 5. 모든 색상에 대해 반복하세요.")
    print("=======================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            time.sleep(0.1)
            continue
            
        # 1. BGR -> HSV 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2. 트랙바에서 현재 값 읽어오기
        h_min = cv2.getTrackbarPos("H_min", "HSV Calibrator")
        h_max = cv2.getTrackbarPos("H_max", "HSV Calibrator")
        s_min = cv2.getTrackbarPos("S_min", "HSV Calibrator")
        s_max = cv2.getTrackbarPos("S_max", "HSV Calibrator")
        v_min = cv2.getTrackbarPos("V_min", "HSV Calibrator")
        v_max = cv2.getTrackbarPos("V_max", "HSV Calibrator")

        # 3. HSV 범위 정의
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        # 4. 마스크 생성 (범위 안의 픽셀은 흰색, 밖은 검은색)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # 5. (디버깅용) 원본 이미지에 마스크를 적용한 '결과'
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # 6. 창 표시하기
        cv2.imshow("Original Camera", frame)
        cv2.imshow("Mask (White=Detected)", mask)
        cv2.imshow("Result (Filtered)", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("캘리브레이션을 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()