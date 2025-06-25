import numpy as np

# 상태 전이 함수 f(x): 간단한 이동 모델
def f(x):
    px, py, vx, vy = x
    dt = 1.0
    return np.array([
        px + vx * dt,
        py + vy * dt,
        vx,
        vy
    ])

# 관측 함수 h(x): 위치만 측정하는 센서 모델 (거리 측정)
def h(x):
    px, py, vx, vy = x
    return np.array([
        np.sqrt(px**2 + py**2)
    ])

# 수치적 야코비안 계산 함수
def numerical_jacobian(func, x, eps=1e-5):
    n = len(x)
    m = len(func(x))
    J = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        J[:, i] = (func(x + dx) - func(x - dx)) / (2 * eps)
    return J

# 예시 입력 상태 (위치 2,3 / 속도 1.0, 1.5)
x = np.array([2.0, 3.0, 1.0, 1.5])

# 상태 전이 함수의 야코비안 계산 (F)
F = numerical_jacobian(f, x)
print("야코비안 F (상태 전이 함수):")
print(F)

# 관측 함수의 야코비안 계산 (H)
H = numerical_jacobian(h, x)
print("\n야코비안 H (관측 함수):")
print(H)
