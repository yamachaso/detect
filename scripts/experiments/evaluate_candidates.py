# %%
from glob import glob
from multiprocessing import Manager, Pool
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from modules.const import SAMPLES_PATH
from modules.grasp import GraspCandidate as NewGraspCandidate
from modules.grasp import (compute_bw_depth_profile, compute_contact_point,
                           compute_depth_profile_in_finger_area,
                           compute_intersection_between_contour_and_line,
                           evaluate_single_insertion_point)
from modules.visualize import get_color_by_score
from scipy.ndimage import map_coordinates
from utils import imshow, load_py2_pickle

# %%
path_list = sorted(glob(f"{SAMPLES_PATH}/saved_data/*"))
path = path_list[0]

data = load_py2_pickle(path)

img = data["img"]
depth = data["depth"]
objects = data["objects"]
objects_num = len(objects)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(img)
axes[1].imshow(depth, cmap="binary")

# %%


def crop(img, center, width):
    half_width = width // 2
    return img[center[1] - half_width:center[1] + half_width, center[0] - half_width:center[0] + half_width]

# %%


def compute_min_max_depth(depth, mask=None):
    if mask is None:
        return depth.min(), depth.max()  # global min max
    else:
        values = depth[mask > 0]
        return values.min(), values.max()  # objects min max


finger_radius = 10  # for index 0
merged_mask = np.where(np.sum([obj["mask"] for obj in objects], axis=0) > 0, 255, 0).astype("uint8")
objects_min_depth, objects_max_depth = compute_min_max_depth(depth, merged_mask)
print("objects range:", objects_min_depth, objects_max_depth)
# %%
target_index = 3
candidate_img = img.copy()
insertion_points_info = []
for i, obj in enumerate(objects):
    if i != target_index:
        continue
    candidates = obj["candidates"]
    contour = obj["contour"]
    mask = obj["mask"]
    center = obj["center"]

    instance_min_depth = depth[mask > 0].min()

    for j, points in enumerate(candidates):
        for edge in points:
            score = evaluate_single_insertion_point(depth, edge, finger_radius, instance_min_depth, objects_max_depth)
            insertion_points_info.append({"edge": edge, "score": score, "candidate_idx": j})
            cv2.line(candidate_img, center, np.int0(edge), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(candidate_img, edge, finger_radius, (255, 0, 0), 1, cv2.LINE_AA)

print(len(insertion_points_info))
fig, axes = plt.subplots(1, 2)
axes[0].imshow(crop(candidate_img, center, 160))
axes[1].hist([obj["score"] for obj in insertion_points_info])

# %%
# ip_scoreによるスレッショルド
ip_score_thresh = 0.5
finger_num = 4
candidate_img_2 = img.copy()
valid_candidates = []
for i in range(0, len(insertion_points_info) + 1, finger_num):
    ip_infos = insertion_points_info[i:i + finger_num]
    invalid_flg = False
    valid_edges = []
    for ip_info in ip_infos:
        edge = ip_info["edge"]
        score = ip_info["score"]
        if score < ip_score_thresh:
            invalid_flg = True
        else:
            valid_edges.append(edge)
    for edge in valid_edges:
        color = (50, 50, 100) if invalid_flg else (255, 100, 0)
        cv2.line(candidate_img_2, center, np.int0(edge), color, 1, cv2.LINE_AA)
        cv2.circle(candidate_img_2, edge, finger_radius, (255, 0, 0), 1, cv2.LINE_AA)
    if not invalid_flg and len(valid_edges):
        valid_candidates.append(valid_edges)

print(valid_candidates)
imshow(crop(candidate_img_2, center, 160))

# %%
# 生き残ったinsertion pointからcontact pointを計算
candidate_img_3 = img.copy()

candidate_ip_scores = []
candidate_cp_scores = []
for points in valid_candidates:
    tmp_ip_scores = []
    tmp_cp_scores = []
    for edge in points:
        contact_point = compute_contact_point(contour, center, edge, finger_radius)
        tmp_ip_scores.append(evaluate_single_insertion_point(depth, edge, finger_radius, instance_min_depth, objects_max_depth))
        tmp_cp_scores.append(evaluate_single_insertion_point(depth, contact_point, finger_radius, instance_min_depth, objects_max_depth))
        cv2.line(candidate_img_3, center, np.int0(edge), (255, 100, 0), 1, cv2.LINE_AA)
        cv2.circle(candidate_img_3, edge, finger_radius, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(candidate_img_3, contact_point, finger_radius, (0, 255, 0), 1, cv2.LINE_AA)
    candidate_ip_scores.append(tmp_ip_scores)
    candidate_cp_scores.append(tmp_cp_scores)

print("ip scores", candidate_ip_scores)
print("cp scores", candidate_cp_scores)
candidate_edge_scores = [list(a * b) for a, b in zip(np.array(candidate_ip_scores), np.array(candidate_cp_scores))]
print("ip x cp scores", candidate_edge_scores)
# TODO: contact pointの中心をもとめて、マスクの重心との差をcandidateのスコアに盛り込む
candidate_scores = [np.prod(scores) for scores in candidate_edge_scores]
print("candidate scores", candidate_scores)

imshow(crop(candidate_img_3, center, 160))
# %%
candidate_img_4 = img.copy()
best_index = np.argmax(candidate_scores)
best_candidate = valid_candidates[best_index]
final_contact_points = []
for edge in best_candidate:
    contact_point = compute_contact_point(contour, center, edge, finger_radius)
    final_contact_points.append(contact_point)
    cv2.line(candidate_img_4, center, np.int0(edge), (255, 100, 0), 2, cv2.LINE_AA)
    cv2.circle(candidate_img_4, edge, finger_radius, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(candidate_img_4, contact_point, finger_radius, (0, 255, 0), 1, cv2.LINE_AA)
hand_radius = np.linalg.norm(np.array(center) - np.array(edge), ord=2)  # const
print(hand_radius)
new_center = np.int0(np.round(np.mean(final_contact_points, axis=0)))
center_diff_score = 1 - (np.linalg.norm(np.array(center) - np.array(new_center), ord=2) / hand_radius)
print(center, new_center, center_diff_score)
cv2.circle(candidate_img_4, center, 3, (0, 0, 255), -1, cv2.LINE_AA)
cv2.circle(candidate_img_4, new_center, 3, (0, 255, 0), -1, cv2.LINE_AA)
imshow(crop(candidate_img_4, center, 160))

# %%


class GraspCandidateElement:
    def __init__(self, depth, contour, center, edge, finger_radius, insertion_score_thresh=0.2, contact_score_thresh=0.2, bw_depth_score_thresh=0.2):
        self.center = center
        self.insertion_point = edge
        self.intersection_point = self.compute_intersection_point(contour)

        self.contact_point = None
        self.contact_score = 0
        self.bw_depth_score = 0
        self.total_score = 0
        self.is_valid = False

        # TODO: ハンドの開き幅調整可能な場合 insertion point = contact pointとなるので、insertionのスコアはいらない
        # 挿入点の評価
        _, intersection_max_depth, intersection_mean_depth = compute_depth_profile_in_finger_area(depth, self.intersection_point, finger_radius)
        min_depth = intersection_mean_depth
        max_depth = max(intersection_max_depth, depth[edge[1], edge[0]])
        # self.insertion_score = self.compute_point_score(self.insertion_point, depth, min_depth, max_depth, finger_radius)
        self.insertion_score = self.compute_point_score(self.insertion_point, depth, min_depth, max_depth, finger_radius)
        if self.insertion_score < insertion_score_thresh:
            return
        # 接触点の計算と評価
        self.intersection_point = self.compute_intersection_point(contour)
        self.contact_point = self.compute_contact_point(finger_radius)
        # self.contact_score = self.compute_point_score(self.contact_point, depth, min_depth, max_depth, finger_radius)
        self.contact_score = self.compute_point_score(self.contact_point, depth, min_depth, max_depth, finger_radius)
        if self.contact_score < contact_score_thresh:
            return
        # 挿入点と接触点の間の障害物の評価
        self.bw_depth_score = self.compute_bw_depth_score(depth)
        if self.bw_depth_score < bw_depth_score_thresh:
            return
        self.total_score = self.compute_total_score()
        # すべてのスコアが基準を満たしたときのみvalid判定
        self.is_valid = True

    def compute_intersection_point(self, contour):
        # TODO: 個々のcropはまだ上位に引き上げられそう
        x, y, w, h = cv2.boundingRect(contour)
        upper_left_point = np.array((x, y))
        shifted_contour = contour - upper_left_point
        shifted_center, shifted_edge = [
            tuple(pt - upper_left_point) for pt in (self.center, self.insertion_point)]
        shifted_intersection = compute_intersection_between_contour_and_line(
            (h, w), shifted_contour, shifted_center, shifted_edge)
        intersection = tuple(shifted_intersection + upper_left_point)
        return intersection

    def compute_contact_point(self, finger_radius):
        # contact_point = tuple(compute_contact_point(contour, self.center, self.insertion_point, finger_radius))
        direction_v = np.array(self.insertion_point) - np.array(self.center)
        unit_direction_v = direction_v / np.linalg.norm(direction_v, ord=2)
        # 移動後座標 = 移動元座標 + 方向ベクトル x 移動量(指半径[pixel])
        contact_point = np.int0(
            np.round(self.intersection_point + unit_direction_v * finger_radius))
        return contact_point

    def compute_point_score(self, point, depth, min_depth, max_depth, finger_radius):
        # score =  evaluate_single_insertion_point(depth, point[::-1], finger_radius, min_depth, max_depth)
        _, _, mean_depth = compute_depth_profile_in_finger_area(depth, point, finger_radius)
        score = max(0, (mean_depth - min_depth)) / (max_depth - min_depth + 1e-6)
        return score

    def compute_bw_depth_score(self, depth):
        # score = compute_bw_depth_score(depth, self.contact_point, self.insertion_point, min_depth)
        local_min_depth, local_max_depth, local_mean_depth = compute_bw_depth_profile(depth, self.contact_point, self.insertion_point)
        # score = max(0, (mean_depth - min_depth)) / (max_depth - min_depth + 1e-6)
        # ２点間の線分内の深度差が小さいとスコアが大きくなる
        score = 1 - ((local_mean_depth - local_min_depth) / (local_max_depth - local_min_depth + 1e-6))
        # インスタンスにおける深度に対してどれだけの深さがあるかも考慮
        # coef = min(local_mean_depth, mean_intersection_depth) / max(local_mean_depth, mean_intersection_depth)
        # _, _, mean_intersection_depth = compute_depth_profile_in_finger_area(depth, self.intersection_point, finger_radius)
        # min_mean_depth = min(local_min_depth, mean_intersection_depth)
        # max_mean_depth = max(local_max_depth, mean_intersection_depth)
        # coef = ((local_mean_depth - min_mean_depth) / (max_mean_depth - min_mean_depth + 10e-6)) ** 2
        # score = 1 - (((local_mean_depth - local_min_depth) / (local_max_depth - local_min_depth + 1e-6)) * coef)
        return score

    def compute_total_score(self):
        # TODO: ip, cp間のdepthの評価 & 各項の重み付け
        return self.insertion_score * self.contact_score * self.bw_depth_score

    def check_invalidness(self, thresh):
        return self.total_score < thresh

    def get_points(self):
        return {"center": self.center, "contact": self.contact_point, "insertion": self.insertion_point}

    def get_scores(self):
        return {"insertion": self.insertion_score, "contact": self.contact_score, "bw_depth": self.bw_depth_score}

    def draw(self, img, line_color=(0, 0, 0), line_thickness=1, circle_thickness=1, show_circle=True):
        cv2.line(img, self.center, self.insertion_point, line_color, line_thickness, cv2.LINE_AA)
        if show_circle:
            cv2.circle(img, self.insertion_point, finger_radius, (255, 0, 0), circle_thickness, cv2.LINE_AA)
            cv2.circle(img, self.contact_point, finger_radius, (0, 255, 0), circle_thickness, cv2.LINE_AA)

        return img


gce = GraspCandidateElement(depth, contour, center, edge, finger_radius)
print(gce.get_points())
print(gce.get_scores())
print("total score:", gce.total_score)

test_img = img.copy()
gce.draw(test_img, (255, 100, 0))

imshow(crop(test_img, center, 160))
# %%


class GraspCandidate:
    def __init__(self, depth, contour, center, edges, finger_radius, hand_radius,
                 elements_score_thresh=0, center_diff_score_thresh=0, el_insertion_score_thresh=0.2, el_contact_score_thresh=0.2, el_bw_depth_score_thresh=0.2):
        self.center = center
        self.elements = [
            GraspCandidateElement(depth, contour, center, edge,
                                  finger_radius, el_insertion_score_thresh,
                                  el_contact_score_thresh, el_bw_depth_score_thresh)
            for edge in edges]
        self.elements_is_valid = self.merge_elements_validness()

        self.shifted_center = None
        self.elements_score = 0
        self.center_diff_score = 0
        self.total_score = 0
        self.is_valid = False

        if self.elements_is_valid:
            # elementの組み合わせ評価
            self.elements_score = self.compute_elements_score()
            if self.elements_score < elements_score_thresh:
                return
            # contact pointsの中心とマスクの中心のズレの評価
            self.shifted_center = self.compute_contact_points_center()
            self.center_diff_score = self.compute_center_diff_score(hand_radius)
            if self.center_diff_score < center_diff_score_thresh:
                return
            # 各スコアの合算
            self.total_score = self.compute_total_score()
            # すべてのスコアが基準を満たしたときのみvalid判定
            self.is_valid = True

    def merge_elements_validness(self):
        return np.all([el.is_valid for el in self.elements])

    def compute_contact_points_center(self):
        contact_points = self.get_contact_points()
        return np.int0(np.round(np.mean(contact_points, axis=0)))

    def compute_elements_score(self):
        element_scores = self.get_element_scores()
        # return np.prod(element_scores)
        # return np.mean(element_scores) * (np.min(element_scores) / np.max(element_scores))
        return (np.mean(element_scores) - np.min(element_scores)) / (np.max(element_scores) - np.min(element_scores))

    def compute_center_diff_score(self, hand_radius):
        return 1. - (np.linalg.norm(np.array(self.center) - np.array(self.shifted_center), ord=2) / hand_radius)

    def compute_total_score(self):
        return self.elements_score * self.center_diff_score

    def get_insertion_points(self):
        return tuple([el.insertion_point for el in self.elements])

    def get_contact_points(self):
        return tuple([el.contact_point for el in self.elements])

    def get_element_scores(self):
        return tuple([el.total_score for el in self.elements])

    def get_scores(self):
        return {"elements": self.elements_score, "center_diff": self.center_diff_score}

    def draw(self, img, line_color=(0, 0, 0), line_thickness=1, circle_thickness=1, show_circle=True):
        for el in self.elements:
            el.draw(img, line_color, line_thickness, circle_thickness, show_circle)
        return img


gc = GraspCandidate(depth, contour, center, points, finger_radius, hand_radius, 0, 0.5)
print(gc.get_insertion_points())
print(gc.get_contact_points())
print(gc.get_scores())

test_img = img.copy()
gc.draw(test_img, (255, 100, 0), 2, 0)
imshow(crop(test_img, gc.center, 160))

# %%
candidate_img = img.copy()
total_spent_time = 0
for i, obj in enumerate(objects):
    candidates = obj["candidates"]
    contour = obj["contour"]
    center = obj["center"]

    gc_list = []
    best_score = 0
    best_index = 0
    for j, points in enumerate(candidates):
        start = time()
        gc = GraspCandidate(depth, contour, center, points, finger_radius, hand_radius, 0, 0, 0.2, 0.2, 0.2)
        spent_time = time() - start
        total_spent_time += spent_time
        if gc.is_valid:
            print(i + j, gc.total_score, spent_time)
            print([[x for x in el.get_scores().values()] for el in gc.elements])
            # validなcandidateの多さはインスタンスの優先順位決定に使えそう
            gc_list.append(gc)
            if gc.total_score > best_score:
                best_score = gc.total_score
                best_index = j
            color = get_color_by_score(gc.total_score)
            gc.draw(candidate_img, line_color=color, line_thickness=2, show_circle=False)

    cv2.circle(candidate_img, center, 3, (0, 0, 255), -1, cv2.LINE_AA)

print("total instance:", len(objects))
print("total time:", total_spent_time)
print("mean time:", total_spent_time / len(objects))
imshow(candidate_img)

# %%
# multiprocessingでは無名関数はつかえないらしい


def func(x):
    return GraspCandidate(depth, contour, center, candidates[x], finger_radius, hand_radius)


# candidates単位で並列化するだけでもだいたい
pool_obj = Pool()
start = time()
answer = pool_obj.map(func, range(len(candidates)))
end = time()
print(end - start)
print(answer)

# %%
# 並列化による高速化ver1

candidate_img = img.copy()
total_spent_time = 0
gc_list_list = []
for i, obj in enumerate(objects):
    candidates = obj["candidates"]
    mask = obj["mask"]
    contour = obj["contour"]
    center = obj["center"]
    instance_min_depth = depth[mask > 0].min()

    pool_obj = Pool()
    start = time()
    gc_list_list.append(pool_obj.map(func, range(len(candidates))))
    spent_time = time() - start
    total_spent_time += spent_time

for i, gc_list in enumerate(gc_list_list):
    for j, gc in enumerate(gc_list):
        if gc.is_valid:
            color = get_color_by_score(gc.total_score)
            gc.draw(candidate_img, line_color=color, line_thickness=2, show_circle=False)

        cv2.circle(candidate_img, gc.center, 3, (0, 0, 255), -1, cv2.LINE_AA)

print("total instance:", objects_num)
print("total time:", total_spent_time)
print("mean time:", total_spent_time / objects_num)
imshow(candidate_img)

# %%
# ref: https://superfastpython.com/parallel-nested-for-loops-in-python/
# 並列化による高速化ver2 (共有プロセスプールの使用)


def sub_task(contour, center, candidate):
    global finger_radius, hand_radius
    return GraspCandidate(depth, contour, center, candidate, finger_radius, hand_radius)


def task(pool, obj):
    global depth
    contour = obj["contour"]
    center = obj["center"]
    candidates = obj["candidates"]

    args = [(contour, center, candidate) for candidate in candidates]
    gc_list = pool.starmap(sub_task, args)
    return gc_list


with Manager() as manager:
    with manager.Pool(100) as pool:
        args = [(pool, obj) for obj in objects]
        start = time()
        gc_list_list = pool.starmap(task, args)
        total_spent_time = time() - start

print("total instance:", len(objects))
print("total time:", total_spent_time)
print("mean time:", total_spent_time / len(objects))

candidate_img = img.copy()
for i, gc_list in enumerate(gc_list_list):
    for j, gc in enumerate(gc_list):
        if gc.is_valid:
            color = get_color_by_score(gc.total_score)
            gc.draw(candidate_img, line_color=color, line_thickness=2, show_circle=False)

        cv2.circle(candidate_img, gc.center, 3, (0, 0, 255), -1, cv2.LINE_AA)

imshow(candidate_img)


# %%
column = 5
row = objects_num // column

fig, axes = plt.subplots(row, column)
for i, gc_list in enumerate(gc_list_list):
    candidate_img = img.copy()
    for gc in gc_list:
        color = get_color_by_score(gc.total_score)
        gc.draw(candidate_img, line_color=color, line_thickness=2, show_circle=False)
        cv2.circle(candidate_img, gc.center, 3, (0, 0, 255), -1, cv2.LINE_AA)

    r = i // (row + 1)
    c = i % column
    axes[r, c].imshow(crop(candidate_img, gc.center, 160))
    axes[r, c].axis("off")

# %%
# p1, p2 = center, edge
# p1, p2 = (526, 387), (549, 421)
# p1, p2 = (485, 177), (493, 146)
p1_invalid, p2_invalid = (519, 295), (527, 263)  # invalid
p1_valid, p2_valid = (472, 176), (501, 168)  # valid
# p1_valid, p2_valid = (0, 176), (501, 176)  # valid

fig, axes = plt.subplots(2, 4)
for i, (p1, p2) in enumerate(((p1_valid, p2_valid), (p1_invalid, p2_invalid))):
    n = np.int0(np.round(np.linalg.norm(np.array(p1) - np.array(p2), ord=2)))
    h, w = np.linspace(p1[0], p2[0], n), np.linspace(p1[1], p2[1], n)
    res_constant = map_coordinates(depth, np.vstack((h, w)), cval=0, order=1)
    res_gconstant = map_coordinates(depth, np.vstack((h, w)), mode="grid-constant", order=1)
    res_nearest = map_coordinates(depth, np.vstack((h, w)), mode="nearest", order=1)

    tmp_img = img.copy()
    cv2.circle(tmp_img, p1, 10, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(tmp_img, p2, 10, (255, 0, 0), -1, cv2.LINE_AA)

    axes[i, 0].imshow(tmp_img)
    axes[i, 1].plot(res_constant)
    axes[i, 2].plot(res_gconstant)
    axes[i, 3].plot(res_nearest)
# %%
tmp_img = img.copy()
p1, p2 = (428, 313), (401, 297)
print(depth[p1[::-1]], depth[p2[::-1]])
cv2.circle(tmp_img, p1, 3, (0, 255, 0), -1, cv2.LINE_AA)
cv2.circle(tmp_img, p2, 3, (255, 0, 0), -1, cv2.LINE_AA)

n = np.int0(np.round(np.linalg.norm(np.array(p1) - np.array(p2), ord=2)))
h, w = np.linspace(p1[1], p2[1], n), np.linspace(p1[0], p2[0], n)
res = map_coordinates(depth, np.vstack((h, w)), mode="mirror", order=1)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(tmp_img)
axes[1].plot(res)

# %%
# 整理したクラスのテスト


def new_sub_task(contour, center, candidate):
    global finger_radius, hand_radius, depth
    return NewGraspCandidate(
        hand_radius_px=hand_radius, finger_radius_px=finger_radius,
        angle=0, depth=depth, contour=contour,
        center=center, insertion_points=candidate,
        elements_th=0, center_diff_th=0,
        el_insertion_th=0.2, el_contact_th=0.2,
        el_bw_depth_th=0.2)


def new_task(pool, obj):
    global depth
    contour = obj["contour"]
    center = obj["center"]
    candidates = obj["candidates"]

    args = [(contour, center, candidate) for candidate in candidates]
    gc_list = pool.starmap(new_sub_task, args)
    return gc_list


with Manager() as manager:
    with manager.Pool(100) as pool:
        args = [(pool, obj) for obj in objects]
        start = time()
        gc_list_list = pool.starmap(new_task, args)
        total_spent_time = time() - start

print("total instance:", len(objects))
print("total time:", total_spent_time)
print("mean time:", total_spent_time / len(objects))

candidate_img = img.copy()
for i, gc_list in enumerate(gc_list_list):
    for j, gc in enumerate(gc_list):
        if gc.is_valid:
            color = get_color_by_score(gc.total_score)
            gc.draw(candidate_img, line_color=color, line_thickness=2, show_circle=False)
        else:
            print("---")
            print(gc.debug_infos)
            for k, el in enumerate(gc.elements):
                if len(el.debug_infos):
                    print(k, el.debug_infos)

        cv2.circle(candidate_img, gc.center, 3, (0, 0, 255), -1, cv2.LINE_AA)

imshow(candidate_img)

# %%
