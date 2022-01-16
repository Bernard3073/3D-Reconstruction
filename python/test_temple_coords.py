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
pts1_start = temp_coord['pts1']
# 4. Run epipolar_correspondences to get points in image 2
# hlp.epipolarMatchGUI(img_1, img_2, F)
pts2_array = []
for p1 in pts1_start:
    pts2_new = sub.epipolar_correspondences(img_1, img_2, F, p1)
    pts2_array.append(pts2_new)
pts2_array = np.array(pts2_array)
# 5. Compute the camera projection matrix P1
intrinsics = np.load('./data/intrinsics.npz')
K1, K2 = intrinsics['K1'], intrinsics['K2']
M1 = np.zeros((3,4))
M1[:3,:3] = np.eye(3)
P1 = K1 @ M1
# 6. Use camera2 to get 4 camera projection matrices P2
E = sub.essential_matrix(F, K1, K2)
M2s = hlp.camera2(E)
# 7. Run triangulate using the projection matrices
bestP2 = None
for i in range(M2s.shape[2]):
    P2 = K2 @ M2s[:, :, i]
    P = sub.triangulate(P1, pts1, P2, pts2)    
    # 8. Figure out the correct P2
    # ensure all z values are positive
    if np.all(P[:, -1] > 0):
        P2 = M2s[:, :, i]
        bestP2 = P2
P2 = bestP2
# 9. Scatter plot the correct 3D points
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=3)
plt.show()
# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
