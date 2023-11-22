#%%
import numpy as np
from scipy import optimize

def compute_cabbage_angle(ratio):
    # 引数 r が 60 ~ 100の間でないと精度が低い
    a = 0.6
    return np.arccos((2 * ratio * ratio - 1 - a * a) / (1 - a * a)) / 2



ratio = 1 
print(compute_cabbage_angle(ratio))
ratio = 0.8 
print(compute_cabbage_angle(ratio))
ratio = 0.6 
print(compute_cabbage_angle(ratio))


#%% 
cabbage_size_mm = 300

def compute_cabbage_angle(ratio, a):
    return np.arccos((2 * ratio * ratio - 1 - a * a) / (1 - a * a)) / 2

def compute_cabbage_angle_reverse(angle, a):
    return np.sqrt(((1 - a*a) * np.cos(2 * angle) + 1 + a*a) / 2)


def get_major_minor_ratio(contours):


    return 0.9

def get_access_distance(contours):
    ratio  = get_major_minor_ratio(contours)
    a = 0.6
    angle = compute_cabbage_angle(ratio, a)
    print(angle)
    print((1 - compute_cabbage_angle_reverse(np.pi / 2 - angle, a)))
    return (1 - compute_cabbage_angle_reverse(np.pi / 2 - angle, a)) * cabbage_size_mm / 2


print(get_access_distance(None))