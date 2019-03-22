import numpy as np
import cv2
import math
from os.path import dirname
from os.path import join

DIRNAME = dirname(__file__)
DIR_IMG = join(DIRNAME, 'project_images')
RATIO_THRESHOLD = 0.5
boxes = cv2.imread(join(DIR_IMG, 'Boxes.png'))
boxes_gray = cv2.cvtColor(boxes,cv2.COLOR_BGR2GRAY)
boxes_gray = np.float32(boxes_gray)
rainier_lst = []
rainier_gray_lst = []
rainier_Ix = []
rainier_Iy = []
for i in range(6):
    rainier_lst.append(cv2.imread(join(DIR_IMG, f'Rainier{i + 1}.png')))
    rainier_gray_lst.append(np.float32(cv2.cvtColor(rainier_lst[i], cv2.COLOR_BGR2GRAY)))
    rainier_Ix.append(cv2.Sobel(rainier_gray_lst[i], cv2.CV_32F, 1, 0, ksize=3))
    rainier_Iy.append(cv2.Sobel(rainier_gray_lst[i], cv2.CV_32F, 0, 1, ksize=3))

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
                keypoints.append(cv2.KeyPoint(x, y, 1))
    return keypoints

def get_out_img(features):
    out = np.zeros(features.shape)
    for y in range(features.shape[0]):
        for x in range(features.shape[1]):
            if features[y, x] > 0.01*features.max():
                # Map feature intensity to 0-255 values, so that the features can be visualised
                out[y, x] = math.floor((features[y,x]*255) / features.max())
    return out

# =========================PART 1=========================
# Get image gradients
boxes_Ix = cv2.Sobel(boxes_gray, cv2.CV_32F, 1, 0, ksize=3)
boxes_Iy = cv2.Sobel(boxes_gray, cv2.CV_32F, 0, 1, ksize=3)

boxes_features = cv2.cornerHarris(boxes_gray, 3, 3, 0.04)
boxes_features = get_local_max(boxes_features)
boxes_kp = get_keypoints(boxes_features)
boxes_out = get_out_img(boxes_features)
cv2.imwrite(join(DIR_IMG, '1a.png'), boxes_out)

r1 = rainier_lst[0]
r1_features = cv2.cornerHarris(rainier_gray_lst[0], 3, 3, 0.04)
r1_features = get_local_max(r1_features)
r1_kp = get_keypoints(r1_features)
r1_out = get_out_img(r1_features)
cv2.imwrite(join(DIR_IMG, '1b.png'), r1_out)

r2 = rainier_lst[1]
r2_features = cv2.cornerHarris(rainier_gray_lst[1], 3, 3, 0.04)
r2_features = get_local_max(r2_features)
r2_kp = get_keypoints(r2_features)
r2_out = get_out_img(r2_features)
cv2.imwrite(join(DIR_IMG, '1c.png'), r2_out)

sift = cv2.xfeatures2d.SIFT_create()
_, r1_desc = sift.compute(r1, r1_kp)
_, r2_desc = sift.compute(r2, r2_kp)
bf = cv2.BFMatcher()
matches = bf.knnMatch(r1_desc, r2_desc, 2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append([m])
matched_img = cv2.drawMatchesKnn(r1, r1_kp, r2, r2_kp, good, None, flags=2)
cv2.imwrite(join(DIR_IMG, '2.png'), matched_img)
# =========================PART 1 END=========================


# =========================PART 2=========================

# =========================PART 2 END=========================