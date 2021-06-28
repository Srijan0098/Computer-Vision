import numpy as np
import cv2

"""
  Returns the harris corners,  image derivative in X direction,  and 
  image derivative in Y direction.
  Args
  - image: numpy nd-array of dim (m, n, c)
  - window_size: The shaps of the windows for harris corner is (window_size, window size)
  - alpha: used in calculating corner response function R
  - threshold: For accepting any point as a corner, the R value must be 
   greater then threshold * maximum R value. 
  - nms_size = non maximum suppression window size is (nms_size, nms_size) 
    around the corner
  Returns 
  - corners: the list of detected corners
  - Ix: image derivative in X direction
  - Iy: image derivative in Y direction

"""
def harris_corners(image, window_size=5, alpha=0.04, threshold=1e-2, nms_size=10):

    image = cv2.GaussianBlur(image,(5,5),1)
    
    der_x = np.array([[0,0,0],[0,0,1],[0,0,0]])
    der_y = np.array([[0,1,0],[0,0,0],[0,0,0]])
    
    ix = cv2.filter2D(image,-1,der_x) - image
    iy = cv2.filter2D(image,-1,der_y) - image
    
    ixx = cv2.GaussianBlur(ix * ix,(window_size,window_size),8)
    iyy = cv2.GaussianBlur(iy * iy,(window_size,window_size),8)
    ixy = cv2.GaussianBlur(ix * iy,(window_size,window_size),8)
    
    det_M = ixx * iyy - ixy**2
    trace_M = ixx + iyy
    R = det_M - alpha*(trace_M)**2
    
    R[R<threshold*np.max(R)]=0
    
    
    for i in range(R.shape[0]-nms_size):
        for j in range(R.shape[1]-nms_size):
            nms_window = R[i:i+nms_size,j:j+nms_size]
            nms_window[nms_window!=np.max(nms_window)]=0
            R[i:i+nms_size,j:j+nms_size]=nms_window
    
    corners = R
    Ix = ix
    Iy = iy

    return corners, Ix, Iy

"""
  Creates key points form harris corners and returns the list of keypoints. 
  You must use cv2.KeyPoint() method. 
  Args
  - corners:  list of Normalized corners.  
  - Ix: image derivative in X direction
  - Iy: image derivative in Y direction
  - threshold: only select corners whose R value is greater than threshold
  
  Returns 
  - keypoints: list of cv2.KeyPoint
  
"""

def get_keypoints(corners, Ix, Iy, threshold):
    
    keypoints=[]
    
    for i in range(corners.shape[0]):
        for j in range(corners.shape[1]):
            if (corners[i,j]>threshold):
                keypoints.append(cv2.KeyPoint(j,i,1,np.degrees(np.arctan(Iy[i,j]/Ix[i,j]))+90,corners[i,j]))

        
    return keypoints


def get_features(image, keypoints, feature_width, scales=None):
   
    """
    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    
    assert image.ndim == 2, 'Image must be grayscale'
    assert len(x) == len(y)

    image=np.pad(image,((8,8),(8,8)))
   
    Ix = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    Iy = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
   
    fv=[]
    for kp in keypoints:
        descriptor=np.zeros(128)
        kp_x=int(kp.pt[1])
        kp_y=int(kp.pt[0])
        for x in range(-8,8):
            for y in range(-8,8):
                i=x+8
                j=y+8
                grad_mag=np.sqrt(Ix[kp_x+x+8,kp_y+y+8]**2+Iy[kp_x+x+8,kp_y+y+8]**2)
                grad_angle=np.rad2deg(np.arctan2(Iy[kp_x+x+8,kp_y+y+8], Ix[kp_x+x+8,kp_y+y+8]))%360
                descriptor[int(binary_2_decimal(np.floor(i/8),np.floor(j/8))*32+
                          binary_2_decimal(np.floor((i-(np.floor(i/8)*8))/4),np.floor((j-(np.floor(j/8)*8))/4))*8+
                          np.floor(grad_angle/45))] += grad_mag
        fv.append(descriptor)
    
    return fv