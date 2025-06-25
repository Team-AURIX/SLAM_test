import numpy as np
import matplotlib.pyplot as plt

# 파라미터
num_particles = 100
landmark = np.array([5.0, 5.0])  # 고정된 랜드마크
sensor_noise_std = 0.5
motion_noise_std = 0.1

# 초기 입자 (x, y)
particles = np.random.randn(num_particles, 2) * 0.5 + np.array([2.0, 2.0])

# 실제 로봇 위치
true_pos = np.array([2.0, 2.0])

# 이동: 오른쪽으로 0.5m
control = np.array([0.5, 0.0])
true_pos += control

# 센서 측정: 랜드마크까지의 거리 (노이즈 추가)
true_distance = np.linalg.norm(landmark - true_pos)
measured_distance = true_distance + np.random.randn() * sensor_noise_std

# 1. 기존 제안 분포: 단순하게 운동모델로 입자 이동
motion_noise = np.random.randn(num_particles, 2) * motion_noise_std
particles += control + motion_noise

# 2. 개선된 제안 분포: 센서 정보를 고려하여 중요도 가중치 계산
weights = np.zeros(num_particles)
for i, p in enumerate(particles):
    expected_dist = np.linalg.norm(landmark - p)
    prob = np.exp(-0.5 * ((expected_dist - measured_distance) / sensor_noise_std) ** 2)
    weights[i] = prob
weights += 1e-300  # 제로 방지
weights /= np.sum(weights)

# 3. 리샘플링 (Systematic Resampling 등 가능)
indices = np.random.choice(range(num_particles), size=num_particles, p=weights)
particles = particles[indices]

# 4. 추정 위치: 입자 평균
estimated_pos = np.mean(particles, axis=0)

# 5. 시각화
plt.figure(figsize=(6, 6))
plt.scatter(particles[:, 0], particles[:, 1], color='gray', s=10, label='Particles')
plt.plot(true_pos[0], true_pos[1], 'go', label='True Position')
plt.plot(estimated_pos[0], estimated_pos[1], 'ro', label='Estimated Position')
plt.plot(landmark[0], landmark[1], 'b*', markersize=15, label='Landmark')
plt.legend()
plt.axis([0, 10, 0, 10])
plt.grid(True)
plt.title("Proposal Distribution Improved (Sensor-Aware Sampling)")
plt.show()
