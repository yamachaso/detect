# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
from modules.grasp import BaseGraspDetector

# %%
colors = [(255, 0, 0),
          (0, 255, 0),
          (0, 0, 255),
          (255, 255, 0),
          (0, 255, 255),
          (255, 0, 255),
          (0, 150, 50),
          (0, 0, 0),
          (0, 0, 0),
          (0, 0, 0),
          (0, 0, 0),
          (0, 0, 0),
          (0, 0, 0),
          (0, 0, 0),
          ]
center = (320, 240)

# %%
paralell_detector = BaseGraspDetector((480, 640), finger_num=2)

img = np.zeros((480, 640, 3))
img[:] = 255

cv2.drawMarker(img, center, (255, 0, 0))

candidates = paralell_detector.detect(center, 100)

for i, cnd in enumerate(candidates):
    p1, p2 = cnd.edges
    cv2.line(img, center, p1.get_edge_on_rgb(), colors[i], thickness=2)
    cv2.line(img, center, p2.get_edge_on_rgb(), colors[i], thickness=2)

plt.imshow(img)
# %%
triangle_detector = BaseGraspDetector((480, 640), finger_num=3)

img = np.zeros((480, 640, 3))
img[:] = 255

cv2.drawMarker(img, center, (255, 0, 0))

candidates = triangle_detector.detect(center, 100)

for i, cnd in enumerate(candidates):
    p1, p2, p3 = cnd.edges
    cv2.line(img, center, p1.get_edge_on_rgb(), colors[i], thickness=2)
    cv2.line(img, center, p2.get_edge_on_rgb(), colors[i], thickness=2)
    cv2.line(img, center, p3.get_edge_on_rgb(), colors[i], thickness=2)

plt.imshow(img)

# %%
