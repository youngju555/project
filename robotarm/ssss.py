from ultralytics import YOLO
import numpy as np
import math
import matplotlib.pyplot as plt

# ✅ YOLO Segmentation 모델
model = YOLO(r"C:\Users\peo00\runs\segment\blue_box_seg_training3\weights\best.pt")

plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))

# 안정화 파라미터
smooth_angle = None
angle_velocity = 0
alpha = 0.15   # EMA (부드러움)
momentum = 0.85  # 회전 관성 계수
prev_angle = None

def stabilize_angle(new_angle, prev_angle, angle_velocity):
    """프레임 간 각도 튐 방지용 안정화 필터"""
    if prev_angle is None:
        return new_angle, 0
    
    diff = new_angle - prev_angle

    # 180도 뒤집힘 보정
    if diff > 90:
        new_angle -= 180
    elif diff < -90:
        new_angle += 180

    # EMA 필터로 각도 완화
    stabilized = (1 - alpha) * prev_angle + alpha * new_angle
    # 모멘텀(회전 관성)
    velocity = momentum * angle_velocity + (1 - momentum) * (new_angle - prev_angle)

    stabilized += velocity * 0.5  # 약간의 관성 반영
    return stabilized, velocity

for r in model.predict(source=1, device='0', stream=True, show=False):
    frame = r.orig_img
    masks = r.masks
    boxes = r.boxes
    confs = boxes.conf.cpu().numpy() if len(boxes) > 0 else []
    cls = boxes.cls.cpu().numpy() if len(boxes) > 0 else []
    names = model.names

    ax.clear()
    ax.imshow(frame[..., ::-1])

    if masks is not None and len(masks.data) > 0:
        for i, mask_tensor in enumerate(masks.data):
            mask = mask_tensor.cpu().numpy()
            ys, xs = np.nonzero(mask > 0.5)
            if len(xs) < 50:
                continue

            # 중심 및 PCA 기반 방향 계산
            coords = np.column_stack((xs, ys))
            mean = np.mean(coords, axis=0)
            centered = coords - mean
            cov = np.cov(centered, rowvar=False)
            eigvals, eigvecs = np.linalg.eig(cov)
            principal_axis = eigvecs[:, np.argmax(eigvals)]
            angle_rad = math.atan2(principal_axis[1], principal_axis[0])
            new_angle = math.degrees(angle_rad)
            if new_angle < 0:
                new_angle += 180

            # 안정화 필터 적용
            stabilized_angle, angle_velocity = stabilize_angle(new_angle, prev_angle, angle_velocity)
            prev_angle = stabilized_angle

            conf = confs[i] if len(confs) > i else 0.0
            label = names[int(cls[i])] if len(cls) > i else "object"

            ax.text(mean[0], mean[1] - 40,
                    f"{label}: {conf:.2f}, angle={stabilized_angle:.1f}°",
                    color='white', fontsize=12, weight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

            # 중심축 시각화
            length = 80
            x2 = mean[0] + length * math.cos(math.radians(stabilized_angle))
            y2 = mean[1] + length * math.sin(math.radians(stabilized_angle))
            ax.plot([mean[0], x2], [mean[1], y2], color='yellow', linewidth=3)

    ax.set_title("Stabilized Angle Detection (No OpenCV, Flip+Momentum Filter)", fontsize=14, weight='bold')
    ax.axis('off')
    plt.pause(0.01)

plt.ioff()
plt.show()
