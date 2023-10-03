# %%
import numpy as np
from scipy import optimize

def convert_mm_to_angle(r_target):
    # 引数 r が 60 ~ 100の間でないと精度が低い
    r0 = 95 # 付け根部分の関係
    l = 180 # 指の長さ

    # 解きたい関数をリストで戻す
    def func(x, r):
        t = x[0]
        equations = [
            (r - r0) * t + l * np.sin(np.radians(70)) - l*np.sin(t + np.radians(70))
        ]
        return equations

    # 制約を設定
    def constraint_func(x):
        return x[0] - 0.6  # t >= 0.6

    # 初期値
    cons = (
        {'type': 'ineq', 'fun': constraint_func}
    )

    # 最適化を実行
    initial_guess = [0.0]
    result = optimize.minimize(lambda x: np.sum(np.array(func(x, r_target))**2), initial_guess, constraints=cons, method="SLSQP")
    return result.x[0]

def convert_angle_to_pressure(angle):
    # degree
    a = 53.77
    b = 21.79
    c = 10.84
    print(angle)
    return (- b + np.sqrt(b * b - 4 * a  *(c - angle)) ) / (2 * a)

def convert_mm_to_pascal(r):
    angle = np.rad2deg(convert_mm_to_angle(r))
    print(angle)
    pressure = convert_angle_to_pressure(angle)
    return pressure

target_r = 95

convert_mm_to_pascal(target_r)


# %%


r0 = 95
l = 180

# 解きたい関数をリストで戻す
def func(x, r):
    t = x[0]
    equations = [
        (r - r0) * t + l * np.sin(np.radians(70)) - l*np.sin(t + np.radians(70))
    ]
    return equations

# 制約を設定
def constraint_func(x):
    return x[0] - 0.6  # t >= 0.6

# 初期値
cons = (
    {'type': 'ineq', 'fun': constraint_func}
)

rr = 60

# 最適化を実行
initial_guess = [0.0]
result = optimize.minimize(lambda x: np.sum(np.array(func(x, rr))**2), initial_guess, constraints=cons, method="SLSQP")
print(np.rad2deg(result.x[0]))

(rr - r0) * result.x[0] + l * np.sin(np.radians(70)) - l*np.sin(result.x[0] + np.radians(70))

# ref : https://qiita.com/imaizume/items/44896c8e1dd0bcbacdd5

# %%
# ref : https://lib-arts.hatenablog.com/entry/scipy_tutorial7

# 以下２つは失敗

from scipy.optimize import minimize_scalar
# r = 100
# r0 = 95
# l = 180
# f = lambda t: (r - r0) * t + l * np.sin(np.radians(70)) - l*np.sin(t + np.radians(70))
# res = minimize_scalar(f, method='brent')
# print(res)
# print(np.rad2deg(res.x))
# res_t = res.x
# (r - r0) * res_t + l * np.sin(np.radians(70)) - l*np.sin(res_t + np.radians(70))


from scipy.optimize import root
r = 65
r0 = 95
l = 180

def func(t):
    return 5 * t + 180 * np.sin(np.radians(70)) - 180*np.sin(t + np.radians(70))
res = root(func, [1])
print(res.x)

print(np.rad2deg(res.x[0]))
res_t = res.x[0]
(r - r0) * res_t + l * np.sin(np.radians(70)) - l*np.sin(res_t + np.radians(70))
