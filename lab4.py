import numpy as np
from numpy import sin, pi
import matplotlib.pyplot as plt
import configparser
from numba import njit, prange

# 设置图像的嵌入字体（本段代码使用 copilot 辅助生成）
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

# 使用 Numba 的 JIT 编译器加速迭代计算
@njit(fastmath=True, parallel=True)
def Iter_calc(n, c, tol, max_iter):
    # 初始化参数
    nu = 0.001
    h = 1.0 / n
    dt = c * h
    
    # 初始化 psi 和 omega 猜测解
    psi = np.zeros((n + 1, n + 1))
    omega = np.zeros((n + 1, n + 1))
    idx_arr = np.arange(1, n)
    omega[idx_arr, n] = - 2 * (sin(pi * idx_arr * h)) ** 2 / h
    
    max_delta = tol + 1
    psi_old = np.zeros_like(psi)
    
    for m in range(max_iter):
        # 减少拷贝次数以提升计算速度
        if m % 1000 == 0:
            psi_old[:] = psi[:]
        
        # 对流项
        conv_term = ((psi[2:, 1:-1] - psi[:-2, 1:-1]) * (omega[1:-1, 2:] - omega[1:-1, :-2]) \
                        - (psi[1:-1, 2:] - psi[1:-1, :-2]) * (omega[2:, 1:-1] - omega[:-2, 1:-1])) / (4 * h**2)
        
        # 黏性项（拉普拉斯项）
        laplace_term = nu * (omega[2:, 1:-1] + omega[0:-2, 1:-1] + omega[1:-1, 2:] + omega[1:-1, 0:-2] - 4 * omega[1:-1, 1:-1]) / h**2
        
        # 第一步：更新内点处的涡量 omega
        omega[1:-1, 1:-1] += (laplace_term + conv_term) * dt
        
        # 第二步：更新内点处的流函数 psi
        psi[1:-1, 1:-1] = (psi[2:, 1:-1] + psi[0:-2, 1:-1] + psi[1:-1, 2:] + psi[1:-1, 0:-2] + omega[1:-1, 1:-1] * (h ** 2)) * 0.25

        # 第三步：利用边界条件更新边界处的涡量 omega
        idx_arr = np.arange(1, n)
        omega[idx_arr, 0] = -2 * psi[idx_arr, 1] / h ** 2
        omega[idx_arr, n] = -2 * psi[idx_arr, n - 1] / h ** 2 - 2 * (sin(pi * idx_arr * h)) ** 2 / h
        omega[0, idx_arr] = -2 * psi[1, idx_arr] / h ** 2
        omega[n, idx_arr] = -2 * psi[n - 1, idx_arr] / h ** 2

        # 减少收敛判断次数以提升计算速度
        if m % 1000 == 0:
            max_delta = np.max(np.abs(psi - psi_old))
            print("Iteration", m, "max delta:", max_delta)                      # 显示当前迭代次数和最大误差
            if max_delta < tol:
                break

    return psi, omega, m, max_delta

#############################
# 第一部分：迭代计算稳态速度场 #
#############################

# 从 settings.txt 读取参数
config = configparser.ConfigParser()
config.read('settings.txt')

n        = config.getint('DEFAULT', 'n')                                        # 网格密度
c        = config.getfloat('DEFAULT', 'c')                                      # Courant 数
tol      = config.getfloat('DEFAULT', 'tol')                                    # 收敛阈值
max_iter = config.getint('DEFAULT', 'max_iter')                                 # 最大迭代次数
corner   = config.getint('DEFAULT', 'corner')                                   # 角部绘图的网格范围

print(f"Parameters: n={n}, c={c}, tol={tol}, max_iter={max_iter}")              # 打印读取的参数以便调试

# 调用迭代函数
psi, omega, m, max_delta = Iter_calc(n, c, tol, max_iter)
print(f"Iterations: {m}, Max Delta: {max_delta:.2e}")                           # 显示计算结束时的迭代次数和最大误差

# 计算速度场
h = 1.0 / n
u = np.zeros(psi.shape)
v = np.zeros(psi.shape)
u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, 0:-2]) / (2 * h)
v[1:-1, 1:-1] = - (psi[2:, 1:-1] - psi[0:-2, 1:-1]) / (2 * h)
speed = np.sqrt(u**2 + v**2)

# 保存结果到 CSV 文件
psi_filename   = f"psi_n={n}_c={c:.2f}.csv"
omega_filename = f"omega_n={n}_c={c:.2f}.csv"
speed_filename = f"speed_n={n}_c={c:.2f}.csv"
np.savetxt(psi_filename, psi, delimiter=",")
np.savetxt(omega_filename, omega, delimiter=",")
np.savetxt(speed_filename, speed, delimiter=",")

#################################
# 第二部分：绘制稳态速度场的流线图 #
#################################

x = np.linspace(0, 1, n + 1)
y = np.linspace(0, 1, n + 1)
X, Y = np.meshgrid(x, y)

# 设置图像格式（本段代码使用 copilot 辅助生成）
fig, ax = plt.subplots(figsize=(14, 9))
strm = ax.streamplot(
    X, Y, u.T, v.T,
    density=5,
    color=speed.T,
    cmap='jet'
)

cbar = fig.colorbar(strm.lines, ax=ax)
cbar.set_label(r'Velocity $|\vec{u}_{i,j}|$', fontsize=24)
cbar.ax.tick_params(labelsize=20)

ax.set_title(
    r'Velocity field in the 2-D square cavity',
    fontsize=30,
    pad=10
)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
strm_filename   = f"streamplot_n={n}_c={c:.2f}.png"
plt.savefig(strm_filename, dpi=300, bbox_inches='tight')

#######################################
# 第三部分：绘制计算区域角部的局部流线图 #
#######################################

# 定义四个角部在 psi 中的索引
corners = {
    "Top--Left":     (slice(0, corner),             slice(n + 1 -corner, n + 1)),
    "Top--Right":    (slice(n + 1 - corner, n + 1), slice(n + 1 - corner, n + 1)),
    "Bottom--Left":  (slice(0, corner),             slice(0, corner)),
    "Bottom--Right": (slice(n + 1 - corner, n + 1), slice(0, corner))
}

# 设置图像格式（本段代码使用 copilot 辅助生成）
fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=False, sharey=False)

for ax, (name, (si, sj)) in zip(axes.flatten(), corners.items()):
    u_sub = u[si, sj]
    v_sub = v[si, sj]

    # 构造各子图的实际坐标
    x = (np.arange(corner + 1) + sj.start) * h
    y = (np.arange(corner + 1) + si.start) * h
    X_sub, Y_sub = X[si, sj], Y[si, sj]

    # 绘制流线图
    strm = ax.streamplot(
        X_sub, Y_sub, u_sub.T, v_sub.T,
        density=1,
        color=np.sqrt(u_sub**2 + v_sub**2).T,
        cmap='jet'
        )
    ax.set_title(name, fontsize=24)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.tick_params(axis='both', labelsize=20)

# 为图像添加统一的 colorbar
cbar = fig.colorbar(
    strm.lines,
    ax=axes,
    orientation='vertical',
    fraction=0.02,                                                              # colorbar 宽度占比
    pad=0.04                                                                    # 子图与 colorbar 之间的间距
    )
cbar.set_label(r'Velocity $|\vec{u}|$', fontsize=20)
cbar.ax.tick_params(labelsize=20)
corner_filename   = f"corner_n={n}_c={c:.2f}.png"
plt.savefig(corner_filename, dpi=300, bbox_inches='tight')

###################################
# 第四部分：绘制中心线上的速度剖面图 #
###################################

# 手动输入文献 GGS82 和 ECG05 提供的原始数据
GGS82_y = np.array([1.00000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
GGS82_x = np.array([1.00000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
GGS82_u = np.array([1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702, -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.0000])
GGS82_v = np.array([0.0000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012, 0.27485, 0.0000])

ECG05_y = np.array([1.000, 0.990, 0.980, 0.970, 0.960, 0.950, 0.940, 0.930, 0.920, 0.910, 0.900, 0.500, 0.200, 0.180, 0.160, 0.140, 0.120, 0.100, 0.080, 0.060, 0.040, 0.020, 0.000])
ECG05_x = np.array([1.000, 0.985, 0.970, 0.955, 0.940, 0.925, 0.910, 0.895, 0.880, 0.865, 0.850, 0.500, 0.150, 0.135, 0.120, 0.105, 0.090, 0.075, 0.060, 0.045, 0.030, 0.015, 0.000])
ECG05_u = np.array([1.000, 0.8486, 0.7065, 0.5917, 0.5102, 0.4582, 0.4276, 0.4101, 0.3993, 0.3913, 0.3838, -0.0620, -0.3756, -0.3869, -0.3854, -0.3690, -0.3381, -0.2960, -0.2472, -0.1951, -0.1392, -0.0757, 0.000])
ECG05_v = np.array([0.000, -0.0973, -0.2173, -0.3400, -0.4417, -0.5052, -0.5263, -0.5132, -0.4803, -0.4407, -0.4028, 0.0258, 0.3756, 0.3705, 0.3605, 0.3460, 0.3273, 0.3041, 0.2746, 0.2349, 0.1792, 0.1019, 0.000])

# 提取当前计算结果的中心线速度剖面（注意之前未计算边界点的速度，仍然为默认值零）
pre_x = np.linspace(0, 1, n + 1)
pre_y = np.linspace(0, 1, n + 1)
pre_u = u[n // 2, :]                                                            # 此处假设 n 为偶数，在本次实验的全过程中是成立的
pre_v = v[:, n // 2]                                                            # 对于 n 为奇数的情形，事实上会填入 i = (n-1)/2 或 j = (n-1)/2 的速度值

# 绘制速度剖面图（本段代码使用 copilot 辅助生成）
plt.figure(figsize=(7, 4.5))
plt.plot(pre_x[1:-1], pre_u[1:-1], label='Present', color='black', linewidth=2) # 去除 pre_x 和 pre_u 的边界点
plt.plot(GGS82_y, GGS82_u, 'o', label='GGS82', markersize=6, color='blue', markerfacecolor='none')
plt.plot(ECG05_y, ECG05_u, '--x', label='ECG05', markersize=6, color='red')
plt.title(r'Velocity $u_{i,j}$ profiles at $y=0.5$', fontsize=20)
plt.xlabel(r'Coordinate $x$', fontsize=16)
plt.ylabel(r'Velocity $u_{i,j}$', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.grid()
plt.tight_layout()

profile_u_filename= f"profile_u_n={n}_c={c:.2f}.png"
plt.savefig(profile_u_filename, dpi=300, bbox_inches='tight')

plt.figure(figsize=(7, 4.5))
plt.plot(pre_x[1:-1], pre_v[1:-1], label='Present', color='black', linewidth=2)
plt.plot(GGS82_x, GGS82_v, 'o', label='GGS82', markersize=6, color='blue', markerfacecolor='none')
plt.plot(ECG05_x, ECG05_v, '--x', label='ECG05', markersize=6, color='red')
plt.title(r'Velocity $v_{i,j}$ profiles at $x=0.5$', fontsize=20)
plt.xlabel(r'Coordinate $y$', fontsize=16)
plt.ylabel(r'Velocity $v_{i,j}$', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.grid()
plt.tight_layout()

profile_v_filename= f"profile_v_n={n}_c={c:.2f}.png"
plt.savefig(profile_v_filename, dpi=300, bbox_inches='tight')