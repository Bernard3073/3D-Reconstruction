import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import cv2

# 1. Load the two temple images and the points from data/some_corresp.npz
data = np.load("./data/some_corresp.npz")
img_1 = cv2.imread("./data/im1.png")
img_2 = cv2.imread("./data/im2.png")
pts1 = data['pts1']
pts2 = data['pts2']
# M is the maximum of the image's width and height
M = max(img_1.shape[0], img_1.shape[1])
# 2. Run eight_point to compute F
F = sub.eight_point(pts1, pts2, M)
# 3. Load points in image 1 from data/temple_coords.npz
temp_coord = np.load("./data/temple_coords.npz")
# 4. Run epipolar_correspondences to get points in image 2
pts2_new = sub.epipolar_correspondences(img_1, img_2, F, pts1)
hlp.epipolarMatchGUI(img_1, img_2, F)
# 5. Compute the camera projection matrix P1

# 6. Use camera2 to get 4 camera projection matrices P2

# 7. Run triangulate using the projection matrices

# 8. Figure out the correct P2

# 9. Scatter plot the correct 3D points

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
