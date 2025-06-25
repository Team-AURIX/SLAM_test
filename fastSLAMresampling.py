import numpy as np
import matplotlib.pyplot as plt

# 파라미터
num_particles = 200
landmark = np.array([5.0, 5.0])
sensor_noise_std = 0.5
motion_noise_std = 0.1
steps = 20

# 초기화
true_pos = np.array([2.0, 2.0])
particles = np.random.randn(num_particles, 2) * 0.2 + true_pos
particle_traces = [particles.copy()]  # 각 시간의 입자 위치 저장
trajectory = [true_pos.copy()]
estimates = []

# 리샘플링 함수
def systematic_resample(weights):
    n = len(weights)
    positions = (np.arange(n) + np.random.rand()) / n
    indexes = np.zeros(n, dtype=int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

# 필터 반복
for step in range(steps):
    # 로봇 이동
    control = np.array([0.5, 0.1 * np.sin(step / 2)])
    true_pos += control
    trajectory.append(true_pos.copy())

    # 센서 관측
    true_distance = np.linalg.norm(landmark - true_pos)
    measured_distance = true_distance + np.random.randn() * sensor_noise_std

    # 입자 이동
    motion_noise = np.random.randn(num_particles, 2) * motion_noise_std
    particles += control + motion_noise

    # 가중치 계산
    distances = np.linalg.norm(particles - landmark, axis=1)
    weights = np.exp(-0.5 * ((distances - measured_distance) / sensor_noise_std) ** 2)
    weights += 1e-300
    weights /= np.sum(weights)

    # 리샘플링
    indices = systematic_resample(weights)
    particles = particles[indices]

    # 추정 위치 저장
    estimates.append(np.mean(particles, axis=0))
    particle_traces.append(particles.copy())

# 시각화
trajectory = np.array(trajectory)
estimates = np.array(estimates)

plt.figure(figsize=(8, 8))

# 각 입자의 자취 그리기
particle_traces = np.array(particle_traces)  # shape: (steps+1, num_particles, 2)
for i in range(num_particles):
    trace = particle_traces[:, i, :]
    plt.plot(trace[:, 0], trace[:, 1], color='lightgray', linewidth=0.5)

# 마지막 입자들
plt.scatter(particles[:, 0], particles[:, 1], color='gray', s=10, label='Final Particles')

# 경로 표시
plt.plot(trajectory[:, 0], trajectory[:, 1], 'g-', label='True Path')
plt.plot(estimates[:, 0], estimates[:, 1], 'r--', label='Estimated Path')
plt.plot(landmark[0], landmark[1], 'b*', markersize=15, label='Landmark')

# 그래프 설정
plt.legend()
plt.title("Particle Filter with Particle Traces")
plt.grid(True)
plt.axis([0, 12, 0, 10])
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
