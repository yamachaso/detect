# %%
import numpy as np
from scipy.spatial.transform import Rotation as R

# %%
# xyz型のオイラー角を定義
phi = 0  # X軸周りの角度
theta = 0  # Y軸周りの角度
psi = 30  # Z軸周りの角度

# xyz型のオイラー角を回転行列に変換
r = R.from_euler('xyz', [phi, theta, psi], degrees=True)
r_xyz = r.as_matrix()  # xyz型の回転行列

# 転置を取り、逆行列を計算し、zyx型の回転行列を得る
r_zyx = np.linalg.inv(r_xyz.T)

# zyx型の回転行列をzyx型のオイラー角に変換
r_zyx_euler = R.from_matrix(r_zyx)
zyx_euler = r_zyx_euler.as_euler('zyx', degrees=True)

print("zyx型のオイラー角:", zyx_euler)


# %%
# だめっぽい？
def xyz_to_zyx(phi, theta, psi):
  # xyz型のオイラー角を回転行列に変換
  r = R.from_euler('xyz', [phi, theta, psi], degrees=True)
  r_xyz = r.as_matrix()  # xyz型の回転行列

  # 転置を取り、逆行列を計算し、zyx型の回転行列を得る
  r_zyx = np.linalg.inv(r_xyz.T)

  # zyx型の回転行列をzyx型のオイラー角に変換
  r_zyx_euler = R.from_matrix(r_zyx)
  zyx_euler = r_zyx_euler.as_euler('zyx', degrees=True)

  return zyx_euler[2], zyx_euler[1], zyx_euler[0]

print(xyz_to_zyx(30, 0, 0)) # 30, 0, 0
print(xyz_to_zyx(0, 30, 0)) # 0 30, 0
print(xyz_to_zyx(0, 0, 30)) # 0, 0, 30

print(xyz_to_zyx(-90, 0, 90)) # 0, 90, 90 ← 合ってない気がする

# %%
# NumPyライブラリをインポート
import numpy as np

# xyz型のオイラー角（φ, θ, ψ）を定義（度単位）
phi = 30
theta = 45
psi = 60

# 各軸周りの回転行列を計算（ラジアン単位）
Rx = np.array([[1, 0, 0],
               [0, np.cos(np.radians(phi)), -np.sin(np.radians(phi))],
               [0, np.sin(np.radians(phi)), np.cos(np.radians(phi))]])
Ry = np.array([[np.cos(np.radians(theta)), 0, np.sin(np.radians(theta))],
               [0, 1, 0],
               [-np.sin(np.radians(theta)), 0, np.cos(np.radians(theta))]])
Rz = np.array([[np.cos(np.radians(psi)), -np.sin(np.radians(psi)), 0],
               [np.sin(np.radians(psi)), np.cos(np.radians(psi)), 0],
               [0, 0, 1]])

# xyz型の回転行列を作成
R_xyz = np.dot(np.dot(Rz, Ry), Rx)

# 逆行列を取得し、zyx型の回転行列を得る
R_zyx = np.linalg.inv(R_xyz.T)

# zyx型のオイラー角を得る（度単位）
phi_new = np.degrees(np.arctan2(R_zyx[1, 2], R_zyx[2, 2]))
theta_new = np.degrees(np.arctan2(-R_zyx[0, 2], np.sqrt(R_zyx[0, 0]**2 + R_zyx[0, 1]**2)))
psi_new = np.degrees(np.arctan2(R_zyx[0, 1], R_zyx[0, 0]))

# 結果を表示
print("xyz型のオイラー角（φ, θ, ψ）は", phi, theta, psi, "です。")
print("zyx型のオイラー角（φ, θ, ψ）は", phi_new, theta_new, psi_new, "です。")


# %%
# 回転行列
t = np.radians(5)


P = np.array([[1, 0, 0],
              [0, np.cos(t), -np.sin(t)],
              [0, np.sin(t), np.cos(t)]])
Q = np.array([[np.cos(t), 0, np.sin(t)],
              [0, 1, 0],
              [-np.sin(t), 0, np.cos(t)]])

z = np.array([0, 0, 1])
np.dot(P, np.dot(Q, z))

# %%
import math

class ContactOrientationController:
  def __init__(self):
    self.angle = 10
    self.contact_angles = {
      0 : [np.radians(0), np.radians(0), math.pi],
      1 : [np.radians(-30), np.radians(0), math.pi + np.radians(-self.angle)], 
      2 : [np.radians(0), np.radians(self.angle), math.pi],
      3 : [np.radians(-15), np.radians(self.angle), math.pi + np.radians(-self.angle)],
      4 : [np.radians(30), np.radians(0), math.pi + np.radians(self.angle)],
      6 : [np.radians(15), np.radians(self.angle), math.pi + np.radians(self.angle)],
      8 : [np.radians(60), np.radians(-self.angle), math.pi],
      9 : [np.radians(-45), np.radians(-self.angle), math.pi + np.radians(-self.angle)],
      12 : [np.radians(45), np.radians(-self.angle), math.pi + np.radians(self.angle)]
    }
    self.z_direction = {}
    for k, v in self.contact_angles.items():
      self.z_direction[k] = self.compute_z_direction(v[2], v[1])

  def compute_z_direction(self, xt, yt):
    P = np.array([[1, 0, 0],
                  [0, np.cos(xt), -np.sin(xt)],
                  [0, np.sin(xt), np.cos(xt)]])
    Q = np.array([[np.cos(yt), 0, np.sin(yt)],
                  [0, 1, 0],
                  [-np.sin(yt), 0, np.cos(yt)]])

    z = np.array([0, 0, 1])
    return np.dot(P, np.dot(Q, z))

coc = ContactOrientationController()
print(coc.contact_angles)
print(coc.z_direction)

# 単位ベクトルか確認
for i in coc.z_direction.values():
  x, y, z = i
  print(x*x + y*y + z*z)