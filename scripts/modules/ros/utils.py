import struct
from typing import Tuple

import numpy as np
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import PointCloud2
from image_geometry import PinholeCameraModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from std_msgs.msg import MultiArrayDimension
from tf.transformations import quaternion_from_matrix

def get_xyz_from_pc2(msg: PointCloud2, uv: Tuple[int, int]) -> Tuple[float, float, float]:
    fields = msg.fields
    point_step = msg.point_step
    row_step = msg.row_step
    data = msg.data

    offset = (uv[1] * row_step) + (uv[0] * point_step)
    target_data = data[offset:offset+point_step]
    xyz = tuple([struct.unpack("<f", target_data[fields[i].offset:fields[i].offset+4])[0] for i in range(3)])

    return xyz

class PointProjector:
    def __init__(self, cam_info):
        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(cam_info)
        self.cu = cam_info.P[2]
        self.cv = cam_info.P[6]
    
    def screen_to_camera_2(self, msg: PointCloud2, uv: Tuple[int, int]) -> Point:
        """
        スクリーン座標系上のピクセル(と対応したdepth)をカメラ座標系へ３次元投影
        ---
        uv: ピクセル位置 (スクリーン座標系)
        d: u,vにおける深度 (この値自体は元々カメラ座標系)
        """
        xyz = get_xyz_from_pc2(msg, uv)
        object_point = Point(*xyz)

        return object_point

    def screen_to_camera(self, uv, d) -> Point:
        """
        Deprecated: スクリーン座標系上のピクセル(と対応したdepth)をカメラ座標系へ３次元投影
        ---
        uv: ピクセル位置 (スクリーン座標系)
        d: u,vにおける深度 (この値自体は元々カメラ座標系)
        """
        # ref: https://news.mynavi.jp/techplus/photo/article/computer_vision-62/images/011l.jpg
        distance = d / 1000  # mm to m

        unit_c = np.array([0,0,1]) # 光軸の単位ベクトル
        unit_x = self._get_direction((uv[0], self.cv))
        unit_y = self._get_direction((self.cu, uv[1]))

        x = np.tan(self._get_angle_between_vectors(unit_c, unit_x)) * distance
        y = np.tan(self._get_angle_between_vectors(unit_c, unit_y)) * distance
        z = distance

        if unit_x[0] < 0:
            x = -x
        if unit_y[1] < 0:
            y = -y

        object_point = Point(x, y, z)
        return object_point
    
    def _get_angle_between_vectors(self, u: np.ndarray, v: np.ndarray):
        """ベクトル間の角度を求める"""
        i = np.inner(u, v) # 内積
        n = np.linalg.norm(u) * np.linalg.norm(v) # ノルムの積 (正)
        c = i / n
        # return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
        return np.arccos(np.clip(c, -1.0, 1.0))

    def _get_direction(self, uv):
        """カメラ座標系原点から対象点までの3次元単位方向ベクトルを算出"""
        vector = np.array(self.cam_model.projectPixelTo3dRay(uv))
        return vector

    def get_length_between_2d_points(self, pt1_2d: Tuple[int, int], pt2_2d: Tuple[int, int], depth: np.ndarray):
        """スクリーン座標系の二点をカメラ座標系に投影し、二点間の長さ[mm]を算出"""
        pt1_3d_c = self.screen_to_camera(pt1_2d, depth[pt1_2d[1], pt1_2d[0]])
        pt2_3d_c = self.screen_to_camera(pt2_2d, depth[pt2_2d[1], pt1_2d[0]])
        distance = self.get_length_between_3d_points(pt1_3d_c, pt2_3d_c)

        return distance

    def get_flat_length_between_2d_points(self, pt1_2d: Tuple[int, int], pt2_2d: Tuple[int, int], depth: np.ndarray):
        """スクリーン座標系の二点をカメラ座標系に投影し、二点間の長さ[mm]を算出"""
        pt1_3d_c = self.screen_to_camera(pt1_2d, depth[pt1_2d[1], pt1_2d[0]])
        pt2_3d_c = self.screen_to_camera(pt2_2d, depth[pt2_2d[1], pt1_2d[0]])
        distance = self.get_length_between_3d_points(pt1_3d_c, pt2_3d_c)

        return distance

    def get_length_between_3d_points(self, pt1_3d: Point, pt2_3d: Point):
        """カメラ座標系の二点間の長さ[mm]を算出"""
        pt1_arr = np.array([pt1_3d.x, pt1_3d.y, pt1_3d.z])
        pt2_arr = np.array([pt2_3d.x, pt2_3d.y, pt2_3d.z])
        distance = np.linalg.norm(pt1_arr - pt2_arr)

        return distance


class PoseEstimator:
    def __init__(self):
        self.pca = PCA(n_components=3)
        self.ss = StandardScaler()

    def get_orientation(self, depth, mask) -> Quaternion:
        """マスクに重なったデプスからインスタンスの姿勢を算出"""
        # ここの値あってるか要検証...
        pts = [(x, y, depth[y, x]) for y, x in zip(*np.where(mask > 0))]
        self.pca.fit(self.ss.fit_transform(pts))
        n, t, b = self.pca.components_
        rmat_44 = np.eye(4)
        rmat_33 = np.dstack([n, t, b])[0]
        rmat_44[:3, :3] = rmat_33
        # 4x4回転行列しか受け入れない罠
        q = quaternion_from_matrix(rmat_44)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


# ref: https://qiita.com/kotarouetake/items/3c467e3c8aee0c51a50f
def numpy2multiarray(multiarray_type, np_array):
    """Convert numpy.ndarray to multiarray"""
    multiarray = multiarray_type()
    multiarray.layout.dim = [MultiArrayDimension(
        "dim%d" % i, np_array.shape[i], np_array.shape[i] * np_array.dtype.itemsize)
        for i in range(np_array.ndim)]
    multiarray.data = np_array.reshape(1, -1)[0].tolist()
    return multiarray


def multiarray2numpy(pytype, dtype, multiarray):
    """Convert multiarray to numpy.ndarray"""
    dims = [x.size for x in multiarray.layout.dim]
    res = np.array(multiarray.data, dtype=pytype).reshape(dims).astype(dtype)
    return res
