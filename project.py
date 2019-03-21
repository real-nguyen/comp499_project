import numpy as np
import cv2
import math
from os.path import dirname
from os.path import join

DIRNAME = dirname(__file__)
DIR_IMG = join(DIRNAME, 'project_images')
R_THRESHOLD = 0.000255
boxes = cv2.imread(join(DIR_IMG, 'Boxes.png'))
boxes_gray = cv2.cvtColor(boxes,cv2.COLOR_BGR2GRAY)
boxes_gray = np.float32(boxes_gray)
# rainier1_orig = cv2.imread(join(DIR_IMG, 'Rainier1.png'))
# rainier2_orig = cv2.imread(join(DIR_IMG, 'Rainier2.png'))
# rainier3_orig = cv2.imread(join(DIR_IMG, 'Rainier3.png'))
# rainier4_orig = cv2.imread(join(DIR_IMG, 'Rainier4.png'))
# rainier5_orig = cv2.imread(join(DIR_IMG, 'Rainier5.png'))
# rainier6_orig = cv2.imread(join(DIR_IMG, 'Rainier6.png'))

def detect_features(img, Ix, Iy):
    Ix2 = np.matrix(Ix * Ix, dtype=np.float32)
    Iy2 = np.matrix(Iy * Iy, dtype=np.float32)
    IxIy = np.matrix(Ix * Iy, dtype=np.float32)
    keypoints = img.copy()
    keypoints[:] = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            det = Ix2[i, j] * Iy2[i, j] - (IxIy[i, j])**2
            trace = Ix2[i, j] + Iy2[i, j]
            R = det / trace if trace != 0 else 0
            if R > R_THRESHOLD:
                keypoints[i, j] = img[i, j]
    local_max = get_local_max(keypoints)
    return local_max

def get_local_max(img):
    # Returns a black image with dots representing the features
    local_max = img.copy()
    local_max[:] = 0
    for i in range(img.shape[0]):        
        for j in range(img.shape[1]):
            # Construct 3x3 window around pixel (i, j)
            # If pixel is in a corner or at an edge, pad with 0's
            up_left = img[i-1, j-1] if (i > 0 and j > 0) else 0
            up = img[i-1, j] if i > 0 else 0
            up_right = img[i-1, j+1] if (i > 0 and j < img.shape[1]-1) else 0
            left = img[i, j-1] if j > 0 else 0
            centre = img[i, j]
            right = img[i, j+1] if j < img.shape[1]-1 else 0
            bottom_left = img[i+1, j-1] if (i < img.shape[0]-1 and j > 0) else 0
            bottom = img[i+1, j] if i < img.shape[0]-1 else 0
            bottom_right = img[i+1, j+1] if (i < img.shape[0]-1 and j < img.shape[1]-1) else 0
            padded_img = np.matrix([[up_left, up, up_right], [left, centre, right], [bottom_left, bottom, bottom_right]])
            # Keep centre pixel if it's the max in its window, otherwise suppress it
            local_max[i,j] = img[i,j] if img[i,j] == padded_img.max() else 0
    return local_max

def get_keypoints(features):
    keypoints = []
    for y in range(features.shape[0]):
        for x in range(features.shape[1]):
            if features[y, x] > 0.01*features.max():
                keypoints.append(cv2.KeyPoint(x, y, 3))
    return keypoints

features = cv2.cornerHarris(boxes_gray,3,3,0.04)
# features = cv2.dilate(features, None)
features = get_local_max(features)
kp = get_keypoints(features)
sift = cv2.xfeatures2d.SIFT_create()
descriptors = sift.compute(boxes,  None)
boxes_kp = boxes.copy()
img = cv2.drawKeypoints(boxes, kp, boxes_kp)
cv2.imshow('detected keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()