# [이 코드를 'find_home_angles.py'로 저장해서 1회만 실행하세요]
import time
try:
    from pymycobot.mycobot320 import MyCobot320 as CobotClass
except Exception:
    from pymycobot.mycobot import MyCobot as CobotClass

# --- 본인 환경에 맞게 수정 ---
PORT = "/dev/ttyACM0"
BAUD = 115200
POSE_HOME = [-264.3, 66.4, 325.0, -177.3, 7.78, 1.83] # v2.9의 POSE_HOME 값
# ---------------------------

print("🤖 로봇 연결...")
mc = CobotClass(PORT, BAUD)
mc.power_on()
time.sleep(3)

print(f"{POSE_HOME} 좌표로 이동합니다...")
mc.send_coords(POSE_HOME, 20, 0) # 20의 속도로 홈 이동
print("이동 중... 4초 대기...")
time.sleep(4) # wait_for_robot_stop() 대신 단순 대기

angles = mc.get_angles()

print("\n=========================================================")
print("✅ 'POSE_HOME'의 실제 앵글 값을 찾았습니다!")
print("   이 리스트를 복사해서 v2.10 코드에 붙여넣으세요:")
print(f"   {list(angles)}")
print("=========================================================")

mc.power_off()
print("🤖 로봇 연결 해제.")