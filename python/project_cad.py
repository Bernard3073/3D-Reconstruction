import numpy as np
import cv2
import submission as sub
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# write your implementation here
def main():
    # load data
    pnp = np.load("./data/pnp.npz", allow_pickle=True)
    cad = pnp['cad']
    image = pnp['image']
    x = pnp['x']
    X = pnp['X']
    # estimate camera matrix, intrinsic matrix, rotation matrix, and translation vector
    P = sub.estimate_pose(x, X)
    K, R, t = sub.estimate_params(P)
    # plot the given 2D point and the projected 3D points on screen
    X_homo = np.hstack([X, np.ones([X.shape[0], 1])])
    project_pts3d = P @ X_homo.T
    project_pts3d = np.round((project_pts3d / project_pts3d[-1]).T).astype('int')
    project_pts3d = project_pts3d[:,:-1]
    plt.imshow(image)
    plt.scatter(x=x[:,0], y=x[:,1], facecolors='none', edgecolors='b', s=100)
    plt.scatter(x=project_pts3d[:,0], y=project_pts3d[:,1], c='r', s=5)
    plt.show()
    plt.close()
    # draw the CAD model rotated by the estimated rotation on screen
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vertices = cad['vertices'][0][0]
    faces = cad['faces'][0][0]

    x = cad['vertices'][0][0][:,0]
    y = cad['vertices'][0][0][:,1]
    z = cad['vertices'][0][0][:,2]
    CAD_vertices = np.concatenate([x,y,z]).reshape([3,-1]).T
    CAD_vertices_rot = R @ CAD_vertices.T
    CAD_vertices_rot = CAD_vertices_rot.T
    ax.set_xlim3d(-0.35, -1)
    ax.set_ylim3d(0.1, 0.45)
    ax.set_zlim3d(0.2, 0.65)
    ax.add_collection3d(Poly3DCollection(CAD_vertices_rot[faces-1],alpha = 0.2,color = 'black'))
    plt.show()
    plt.close() 
    # project the CAD's all vertices onto the image and draw the projected CAD model overlapping with the 2D image
    vertices_homo = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    project_vertices = P @ vertices_homo.T
    project_vertices = np.round((project_vertices / project_vertices[-1]).T).astype('int')
    project_vertices = project_vertices[:, :-1]
    plt.imshow(image)
    plt.scatter(x=project_vertices[:,0], y=project_vertices[:,1], c='g', marker='o',s=5)
    plt.show()
if __name__ == '__main__':
    main()