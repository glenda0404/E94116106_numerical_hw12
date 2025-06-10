import numpy as np
import matplotlib.pyplot as plt

# 網格設定
Nr = 50     # r方向切分點數
Nθ = 50     # θ方向切分點數

r_min = 0.5
r_max = 1.0
θ_min = 0
θ_max = np.pi / 3

dr = (r_max - r_min) / (Nr - 1)
dθ = (θ_max - θ_min) / (Nθ - 1)

r = np.linspace(r_min, r_max, Nr)
θ = np.linspace(θ_min, θ_max, Nθ)

# 建立解的網格
T = np.zeros((Nr, Nθ))

# 邊界條件
T[0, :] = 50         # T(0.5, θ) = 50
T[-1, :] = 100       # T(1.0, θ) = 100
T[:, 0] = 0          # T(r, 0) = 0
T[:, -1] = 0         # T(r, π/3) = 0

# Laplace solver
def solve_laplace(T, r, dr, dθ, tol=1e-4, max_iter=10000):
    Nr, Nθ = T.shape
    for iteration in range(max_iter):
        T_old = T.copy()
        for i in range(1, Nr - 1):
            for j in range(1, Nθ - 1):
                ri = r[i]
                T[i, j] = (
                    (T[i+1, j] + T[i-1, j]) / dr**2 +
                    (1 / (ri**2)) * (T[i, j+1] + T[i, j-1]) / dθ**2 +
                    (1 / (ri * dr)) * (T[i+1, j] - T[i-1, j]) / (2 * dr)
                ) / (
                    2 / dr**2 + 2 / (ri**2 * dθ**2)
                )
        diff = np.max(np.abs(T - T_old))
        if diff < tol:
            print(f'✅ Converged after {iteration} iterations.\n')
            break
    else:
        print('⚠️ Did not converge within the max iterations.')
    return T

T = solve_laplace(T, r, dr, dθ)

# 數值結果輸出（前幾行）
print("Numerical result (Temperature field T[r, θ]):")
print("Partial output (first 10 r rows × first 10 θ columns):\n")
for i in range(min(10, Nr)):
    for j in range(min(10, Nθ)):
        print(f"{T[i,j]:8.2f}", end=' ')
    print()

# 可視化（極座標轉直角坐標）
R, Θ = np.meshgrid(r, θ, indexing='ij')
X = R * np.cos(Θ)
Y = R * np.sin(Θ)

plt.figure(figsize=(8, 6))
cp = plt.contourf(X, Y, T, levels=50, cmap='inferno')
plt.colorbar(cp, label='Temperature T')
plt.title('Solution to Laplace Equation in Polar Coordinates')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True)
plt.show()
