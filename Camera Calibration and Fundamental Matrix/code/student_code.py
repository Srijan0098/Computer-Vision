import numpy as np


def calculate_projection_matrix(points_2d, points_3d):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_2d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """

    # Placeholder M matrix. It leads to a high residual. Your total residual
    # should be less than 1.
    M = np.asarray([[0.1768, 0.7018, 0.7948, 0.4613],
                    [0.6750, 0.3152, 0.1136, 0.0480],
                    [0.1020, 0.1725, 0.7244, 0.9932]])

    ###########################################################################
    # TODO: YOUR PROJECTION MATRIX CALCULATION CODE HERE
    ###########################################################################

    #raise NotImplementedError('`calculate_projection_matrix` function in ' +
        #'`student_code.py` needs to be implemented')
    aone = np.ones(points_3d.shape[0])
    arr1 = np.column_stack((points_3d,aone))
    b = np.zeros((arr1.shape))
    A1 = np.concatenate((arr1,b),axis=1)
    A1 = A1.reshape(-1,4)
    A2 = np.concatenate((b,arr1),axis=1)
    A2 = A2.reshape(-1,4)
    x1 = points_2d.flatten()
    x1 = x1.reshape(-1,1).repeat(3,axis=1)
    y1 = points_3d.repeat(2,axis=0)
    A3 = -np.multiply(x1,y1)
    A = np.concatenate((A1,A2,A3),axis=1)
    b = points_2d.reshape(-1,1)
    a1,b1,c1,d1 = np.linalg.lstsq(A,b)
    M = np.append(a1,[1]).reshape(3,4)
    residuals = b1
    return M
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################


def calculate_camera_center(M):
    """
    Returns the camera center matrix for a given projection matrix.

    The center of the camera C can be found by:

        C = -Q^(-1)m4

    where your project matrix M = (Q | m4).

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    # Placeholder camera center. In the visualization, you will see this camera
    # location is clearly incorrect, placing it in the center of the room where
    # it would not see all of the points.
    

    ###########################################################################
    # TODO: YOUR CAMERA CENTER CALCULATION CODE HERE
    ###########################################################################
    Q1=np.zeros((3,3))

    m=np.zeros((3,1))
    
    for i in range(3):
        for j in range(3):
            Q1[i][j]=M[i][j]
        
 
    for i in range(3):
        m[i][0]=M[i][3]
            
    mat = np.linalg.solve(Q1,-m).reshape(3,)
    return mat
    
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

   

def estimate_fundamental_matrix(points_a, points_b):
    """
    Calculates the fundamental matrix. Try to implement this function as
    efficiently as possible. It will be called repeatedly in part 3.

    You must normalize your coordinates through linear transformations as
    described on the project webpage before you compute the fundamental
    matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B

    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """

    # Placeholder fundamental matrix
    F = np.asarray([[0, 0, -0.0004],
                    [0, 0, 0.0032],
                    [0, -0.0044, 0.1034]])

    ###########################################################################
    # TODO: YOUR FUNDAMENTAL MATRIX ESTIMATION CODE HERE
    ###########################################################################

    #raise NotImplementedError('`estimate_fundamental_matrix` function in ' +
        #'`student_code.py` needs to be implemented')
    
    assert points_a.shape[0]==points_b.shape[0]
    a1 = np.ones((points_a.shape[0]))
    p1 = np.column_stack((points_a,a1))
    p2 = np.column_stack((points_b,a1))
    p2 = np.repeat(p2,3,axis=1)
    p1 = p1.repeat(3,axis=0).reshape(-1,9)
    A = np.multiply(p1,p2)
    U,sigma,V = np.linalg.svd(A)
    f_matrix = V[-1].reshape((3,3))
    U1,D1,V1 = np.linalg.svd(f_matrix)
    D1[-1] = 0
    D2 = np.zeros((3,3))
    for i in range(3):
        D2[i][i] = D1[i]
    F = U1@D2@V1
    
    
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return F

def ransac_fundamental_matrix(matches_a, matches_b):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Your RANSAC loop should contain a call to
    estimate_fundamental_matrix() which you wrote in part 2.

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 100 points for either left or
    right images.

    Args:
    -   matches_a: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image A
    -   matches_b: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_a: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image A that are inliers with
                   respect to best_F
    -   inliers_b: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image B that are inliers with
                   respect to best_F
    """

    # Placeholder values
    best_F = estimate_fundamental_matrix(matches_a[:10, :], matches_b[:10, :])
    inliers_a = matches_a[:100, :]
    inliers_b = matches_b[:100, :]

    ###########################################################################
    # TODO: YOUR RANSAC CODE HERE
    ###########################################################################

    #raise NotImplementedError('`ransac_fundamental_matrix` function in ' +
     #   '`student_code.py` needs to be implemented')
    
    num_iterator = 20000
    threshold = 0.001
    best_F = np.zeros((3, 3))
    max_inlier = 0
    num_rand_samples = 8
    
    a1=np.ones((matches_a.shape[0]))
    b1=np.ones((matches_b.shape[0]))
    p1=np.column_stack((matches_a,a1))
    p2=np.column_stack((matches_b,b1))
    p2=np.repeat(p2,3,axis=1)
    p1=p1.repeat(3,axis=0).reshape(-1,9)
    A=np.multiply(p1,p2)

    

    for i in range(num_iterator):
        index_rand = np.random.randint(matches_a.shape[0], size=num_rand_samples)
        F_matrix = estimate_fundamental_matrix(matches_a[index_rand, :], matches_b[index_rand, :])
        err = np.abs(np.matmul(A, F_matrix.reshape((-1))))
        current_inlier = np.sum(err <= threshold)
        if current_inlier > max_inlier:
            best_F = F_matrix.copy()
            max_inlier = current_inlier

    err = np.abs(np.matmul(A, best_F.reshape((-1))))
    index = np.argsort(err)

    inliers_a = matches_a[index[:max_inlier]]
    inliers_b = matches_b[index[:max_inlier]]
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return best_F, inliers_a, inliers_b