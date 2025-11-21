# -*- coding: utf-8 -*-
"""
체커보드 기반 카메라 캘리브레이션 도우미
---------------------------------------------
✅ 사용법:
  1. 카메라 켜짐 → 체커보드를 여러 각도에서 비춰보세요.
  2. 스페이스바를 누르면 현재 프레임이 calib_images 폴더에 저장됩니다.
  3. q 키를 누르면 저장 종료 및 캘리브레이션 계산이 시작됩니다.
  4. 결과 camera_matrix.npy, dist_coeffs.npy 파일로 저장됩니다.
"""

import cv2
import numpy as np
import os

# ---------------------------------------------
# 🔧 체커보드 설정
# ---------------------------------------------
CHECKERBOARD = (6, 5)          # 내부 코너 수 (6x5 체커보드 → 5x4 코너)
square_size = (1.5, 1.0)       # 한 칸의 실제 크기 (mm)

# 저장 폴더 준비
save_dir = "calib_images"
os.makedirs(save_dir, exist_ok=True)

# 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("⚠️ 카메라를 열 수 없습니다.")
    exit()

print("📸 카메라 캘리브레이션 촬영 시작")
print("스페이스바: 사진 저장 / q: 종료 및 계산")

img_counter = 0

# ---------------------------------------------
# 🔹 스페이스바로 사진 촬영
# ---------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # 그레이 변환 후 체커보드 코너 감지
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_cb, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 감지되면 표시
    display = frame.copy()
    if ret_cb:
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners, ret_cb)
        cv2.putText(display, "Checkerboard detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Calibration Capture", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        # 스페이스바 누르면 사진 저장
        filename = os.path.join(save_dir, f"/home/robotarm/calib_{img_counter:02d}.jpg")
        cv2.imwrite(filename, frame)
        img_counter += 1
        print(f"💾 저장됨: {filename}")

    elif key == ord('q'):
        print("🔒 촬영 종료 → 캘리브레이션 계산 시작")
        break

cap.release()
cv2.destroyAllWindows()

# ---------------------------------------------
# 🔹 카메라 캘리브레이션 계산
# ---------------------------------------------
images = [os.path.join(save_dir, f) for f in os.listdir(save_dir)
          if f.lower().endswith(('.jpg', '.png'))]

if len(images) < 5:
    print("⚠️ 최소 5장 이상의 이미지가 필요합니다.")
    exit()

objpoints = []  # 3D 체커보드 점
imgpoints = []  # 2D 이미지 점

# 체커보드의 실제 3D 좌표 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp[:, 0] *= square_size[0]
objp[:, 1] *= square_size[1]

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret_cb, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret_cb:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        objpoints.append(objp)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret_cb)
        cv2.imshow("Detected", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# 내부 파라미터 계산
print("🧮 내부 파라미터 계산 중...")
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n=== ✅ 캘리브레이션 완료 ===")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# 결과 저장
np.save("camera_matrix.npy", camera_matrix)
np.save("dist_coeffs.npy", dist_coeffs)

print("\n💾 저장됨:")
print(" - camera_matrix.npy")
print(" - dist_coeffs.npy")

print("\n📏 총 이미지 수:", len(images))
print("평균 재투영 오차:", ret)
print("✅ 완료")
