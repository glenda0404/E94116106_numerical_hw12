import numpy as np
import matplotlib.pyplot as plt

# 網格參數
dx = 0.1
dt = 0.1
x = np.arange(0, 1 + dx, dx)
t = np.arange(0, 1 + dt, dt)

Nx = len(x)
Nt = len(t)

# 波速 c = 1，lambda = (c*dt/dx)^2
c = 1
λ = (c * dt / dx)**2

# 初始化解陣列 p[i, j] 對應 p(x_i, t_j)
p = np.zeros((Nx, Nt))

# 初始條件：p(x,0) = cos(2πx)
p[:, 0] = np.cos(2 * np.pi * x)

# 初始速度：∂p/∂t(x,0) = 2π sin(2πx)
# 使用中央差分來計算第一層 (j=1)
for i in range(1, Nx - 1):
    p[i, 1] = (
        p[i, 0]
        + dt * 2 * np.pi * np.sin(2 * np.pi * x[i])
        + 0.5 * λ * (p[i + 1, 0] - 2 * p[i, 0] + p[i - 1, 0])
    )

# 套用邊界條件
p[0, :] = 1
p[-1, :] = 2

# 時間步進公式
for j in range(1, Nt - 1):
    for i in range(1, Nx - 1):
        p[i, j + 1] = (
            2 * p[i, j] - p[i, j - 1]
            + λ * (p[i + 1, j] - 2 * p[i, j] + p[i - 1, j])
        )

# ✅ 數值輸出
print("Numerical solution p(x, t):")
print("x \\ t | " + "  ".join([f"{tj:>6.2f}" for tj in t[:6]]))  # 只印前幾欄
print("-" * 60)
for i in range(Nx):
    print(f"{x[i]:5.2f} | " + "  ".join([f"{p[i, j]:6.3f}" for j in range(6)]))  # 前6欄

# 📈 繪圖顯示不同時間的波形
plt.figure(figsize=(8, 6))
for j in range(0, Nt, 2):  # 每隔兩個時間點畫一次
    plt.plot(x, p[:, j], label=f't = {t[j]:.1f}')
plt.title("Wave Equation Solution")
plt.xlabel("x")
plt.ylabel("p(x, t)")
plt.legend()
plt.grid(True)
plt.show()
