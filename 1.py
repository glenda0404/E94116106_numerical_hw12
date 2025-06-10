import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# 定義區域與網格
pi = np.pi
h = k = 0.1 * pi
x_vals = np.arange(0, pi + h, h)
y_vals = np.arange(0, pi/2 + k, k)
nx, ny = len(x_vals), len(y_vals)

# 建立解的陣列與初始值
u = np.zeros((nx, ny))

# 邊界條件
for j in range(ny):
    u[0, j] = np.cos(y_vals[j])        # u(0, y)
    u[-1, j] = -np.cos(y_vals[j])      # u(pi, y)

for i in range(nx):
    u[i, 0] = np.cos(x_vals[i])        # u(x, 0)
    u[i, -1] = 0                       # u(x, pi/2)

# 定義右邊項 f(x,y) = x*y
f = np.zeros((nx, ny))
for i in range(nx):
    for j in range(ny):
        f[i, j] = x_vals[i] * y_vals[j]

# Jacobi method 迭代
u_new = u.copy()
tolerance = 1e-6
max_iter = 10000

for iteration in range(max_iter):
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u_new[i, j] = 0.25 * (
                u[i+1, j] + u[i-1, j] +
                u[i, j+1] + u[i, j-1] -
                h**2 * f[i, j]
            )
    # 收斂檢查
    error = np.linalg.norm(u_new - u)
    if error < tolerance:
        break
    u = u_new.copy()

print(f"Converged in {iteration} iterations")

# 輸出數值結果
df = pd.DataFrame(u, index=np.round(x_vals, 3), columns=np.round(y_vals, 3))
df.index.name = "x\\y"
print("\nNumerical solution matrix u(x,y):")
print(df.round(6))  # 保留六位小數輸出
