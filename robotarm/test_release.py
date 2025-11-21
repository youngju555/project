from pymycobot import MyCobot320
import time

port = '/dev/ttyACM0'  # 본인 환경에 맞게 수정
baud = 115200

mc = MyCobot320(port, baud)
time.sleep(1)

print("✅ MyCobot 연결 완료")
servo = mc.is_all_servo_enable()
print("servo :",servo) 
print("error:", mc.get_error_information())
print(mc.is_power_on())
print(mc.is_controller_connected())

mc.release_all_servos()
exit()

# 1️⃣ 펌웨어 버전 확인
try:
    fw = mc.get_firmware_version()
    print("📦 펌웨어 버전:", fw)
except Exception as e:
    print("⚠️ get_firmware_version() 실패:", e)

# 2️⃣ 지원 함수 목록 확인 (디버깅용)
print("\n📋 지원 함수 중 'release' 관련 항목:")
for f in dir(mc):
    if "release" in f:
        print("  -", f)

# 3️⃣ release_all_servos 테스트
print("\n🔧 감쇠 모드 해제 시도...")
res = mc.release_all_servos()
print("반환값:", res)

time.sleep(1)

print("🔧 비감쇠 모드 해제 시도...")
res2 = mc.release_all_servos(1)
print("반환값:", res2)
