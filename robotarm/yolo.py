from ultralytics import YOLO
import numpy as np

# 1. 가지고 계신 모델 로드
# 🚨 모델 경로를 정확하게 입력하세요.
model_path = "/home/young/robotarm/best.pt"
model = YOLO(model_path) 

# 2. 아무 이미지나 빈 검은색 이미지를 생성 (테스트용)
# (이미지 파일이 없어도 테스트 가능)
dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)

# 3. predict 실행
results = model.predict(dummy_image)

# 4. 💥 masks 속성 확인!
if results[0].masks is None:
    print(f"\n❌ 이 모델({model_path})은 [BBox (Detection)] 모델입니다.")
    print(f"   -> masks 객체: {results[0].masks}")
else:
    print(f"\n✅ 이 모델({model_path})은 [Segmentation (-seg)] 모델입니다!")
    print(f"   -> masks 객체: {results[0].masks}") # <ultralytics.engine.results.Masks object ...>