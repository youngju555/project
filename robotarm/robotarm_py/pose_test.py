# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 - 좌표 이동 테스트
--------------------------------------------
지정된 좌표로 로봇을 이동시키는 간단한 코드
"""

import time
from pymycobot.mycobot320 import MyCobot320

# ---------------------------------------------
# 기본 설정
# ---------------------------------------------
PORT = "/dev/ttyACM0"   # 연결된 포트 확인 (예: /dev/ttyACM0, /dev/ttyUSB0 등)
BAUD = 115200

mc = MyCobot320(PORT, BAUD)
time.sleep(1)

print("🔌 로봇 연결 완료")

# 전원 ON (필요 시)
mc.power_off()
mc.power_on()
time.sleep(1)

# ---------------------------------------------
# 이동할 좌표 설정
# ---------------------------------------------
target_coords = [90,80,40,60,20,10]  # (X, Y, Z, Roll, Pitch, Yaw)
speed = 30   # 1~100 사이의 속도 (높을수록 빠름)
mode = 0     # 1 = 절대 좌표 이동, 0 = 상대 이동

print(f"🎯 지정 좌표로 이동 중: {target_coords}")
mc.send_angles(target_coords, speed, mode)

# 이동 완료 대기
time.sleep(5)

# ---------------------------------------------
# 현재 좌표 출력
# ---------------------------------------------
current_coords = mc.get_coords()
print(f"📍 현재 좌표: {current_coords}")

print("✅ 이동 테스트 완료")
