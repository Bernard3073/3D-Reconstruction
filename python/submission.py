"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import cv2
import helper

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    # scale the data by dividing each coordinate by M
    pts1 = pts1/M
    pts2 = pts2/M
    A = []
    for p1, p2 in zip(pts1, pts2):
        A.append([p2[0]*p1[0], p2[0]*p1[1], p2[0], p2[1]*p1[0], p2[1]*p1[1], p2[1], p1[0], p1[1], 1])
    A = np.array(A)
    A.reshape((len(pts1), 9))
    # Solve A using SVD
    U, S, V = np.linalg.svd(A)
    V = V.T
    # last col = solution
    sol = V[:,-1]
    F = sol.reshape((3, 3))
    U_F, S_F, V_F = np.linalg.svd(F)
    # Rank 2 constraint(set the smallest singular value to 0)
    S_F[-1] = 0
    S_new = np.diag(S_F)
    # Recompute normalized F
    F_new = U_F @ S_new @ V_F
    # Refine F by using local minimization
    F = helper.refineF(F, pts1, pts2)
    T = np.diag([1/M, 1/M, 1])
    F_norm = T.T @ F_new @ T
    F_norm = F_norm / F_norm[-1, -1]
    return F_norm


'''
Returns a 2D Gaussian kernel
Input:  size, the kernel size (will be square)
        sigma, the sigma Gaussian parameter
Output: kernel, (size, size) array with the centered gaussian kernel
'''
def gaussianWindow(size, sigma=3):
    x = np.linspace(-(size//2), size//2, size)
    x /= np.sqrt(2)*sigma
    x2 = x**2
    kernel = np.exp(- x2[:, None] - x2[None, :])
    return kernel / kernel.sum()

"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    # Convert to grayscale for better correspondence
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # x1, y1 = pts1[0,0], pts1[0,1]
    x1, y1 = pts1
    # Find epipolar line
    l = F @ [x1, y1, 1]
    s = np.sqrt(l[0]**2 + l[1]**2)
    l = l/s
    # search over the set of pixels that lie along the epipolar line
    # Find start and end points for epipolar line
    sy, sx = im2.shape
    # epipolar line eq: [a, b, c].T @ [x, y, z] = 0 (l = [a, b, c])
    if l[0] != 0:
        ye, ys = sy-1, 0
        xe, xs = round(-(l[1] * ye + l[2])/l[0]), round(-(l[1] * ys + l[2])/l[0])
    else:
        xe, xs = sx-1, 0
        ye, ys = round(-(l[0] * xe + l[2])/l[1]), round(-(l[0] * xs + l[2])/l[1])

    # Generate (x, y) test points
    N = max(abs(ye - ys), abs(xe - xs)) + 1
    xp = np.round(np.linspace(xs, xe, N)).astype('int')
    yp = np.round(np.linspace(ys, ye, N)).astype('int')
    # Correspondence parameters
    limit = 40
    win_size = 9
    x2, y2 = None, None
    best_score = np.finfo('float').max

    for x, y in zip(xp, yp):
        # Check if test point is close within limit
        if (abs(x-x1) > limit) or (abs(y-y1) > limit): continue

        # Check if it's possible to create a window
        if not ((y-win_size//2 >= 0) and (y+1+win_size//2 < sy) \
            and (x-win_size//2 >= 0) and (x+1+win_size//2 < sx)): continue

        # Create windows
        win1 = im1[y1-win_size//2:y1+1+win_size//2, x1-win_size//2:x1+1+win_size//2]
        win2 = im2[y-win_size//2:y+1+win_size//2, x-win_size//2:x+1+win_size//2]
        
        # Apply gaussian kernel and compute SSD error
        gaussian_kernel = gaussianWindow(win_size)
        score = np.sum((gaussian_kernel * (win1 - win2))**2)

        # Save best matching points
        if score < best_score:
            best_score = score
            x2, y2 = x, y

    return np.array([x2, y2])


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    E = K2.T @ F @ K1
    U, S, V = np.linalg.svd(E)
    # Due to the noise in K, the singular values of E are not necessarily (1, 1, 0)
    # This can be corrected by reconstructing it with (1, 1, 0) singular values
    S[0] = 1
    S[1] = 1
    S[2] = 0
    S_new = np.diag(S)
    E_new = U @ S_new @ V
    return E_new


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    # replace pass by your implementation
    pass


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    pass


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
