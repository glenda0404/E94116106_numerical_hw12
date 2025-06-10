import numpy as np
import pandas as pd

# 參數設定
K = 0.1
delta_t = 0.5
delta_r = 0.1
r = np.arange(0.5, 1.0 + delta_r, delta_r)
t = np.arange(0, 10 + delta_t, delta_t)
R, T = len(r), len(t)

# lambda 係數
lam = delta_t / (4 * K * delta_r**2)

# 建立解矩陣
T_sol = np.zeros((T, R))

# 初始條件 T(r,0) = 200(r - 0.5)
T_sol[0, :] = 200 * (r - 0.5)

# 時間向前迭代
for n in range(0, T - 1):
    for i in range(1, R - 1):
        ri = r[i]
        T_sol[n + 1, i] = T_sol[n, i] + lam * (
            T_sol[n, i + 1] - 2 * T_sol[n, i] + T_sol[n, i - 1]
            + (delta_r / (2 * ri)) * (T_sol[n, i + 1] - T_sol[n, i - 1])
        )

    # 邊界條件 r = 1（右端）：T = 100 + 40t
    T_sol[n + 1, -1] = 100 + 40 * t[n + 1]

    # 邊界條件 r = 0.5（左端）: ∂T/∂r + 3T = 0 → (T1 - T0)/dr + 3*T0 = 0
    T_sol[n + 1, 0] = T_sol[n + 1, 1] / (1 + 3 * delta_r)

# 顯示收斂結果
df = pd.DataFrame(T_sol, index=np.round(t, 2), columns=np.round(r, 2))
df.index.name = "t \\ r"
print(df.round(2))
