import numpy as np
import cv2
import math
from os.path import dirname
from os.path import join
from random import randint

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

def project(x, y, H):
    # Step 3A
    coord = np.array([x, y, 1])
    projection = np.matmul(H, coord)
    x2 = projection[0] / projection[2]
    y2 = projection[1] / projection[2]
    return (x2, y2)

def computeInlierCount(H, matches, img1_pts, img2_pts, inlierThreshold):
    # Step 3B
    nb_inliers = 0
    projections = []
    for x, y in img1_pts:
        projections.append(project(x, y, H))

    match_idx = 0
    for x1, y1 in projections:
        x2, y2 = img2_pts[match_idx]
        # Calculate Euclidian distance
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if distance < inlierThreshold:
            nb_inliers += 1
        match_idx += 1
    return nb_inliers

def RANSAC(matches, kp1, kp2, numIterations, inlierThreshold, img1, img2):
    # Step 3C
    best_H = None
    max_inliers = 0
    nb_matches = len(matches)
    img1_pts = np.array([kp1[m.queryIdx].pt for m in matches])
    img2_pts = np.array([kp2[m.trainIdx].pt for m in matches]) 
    for i in range(numIterations):
        rand_matches = []
        for j in range(4):
            idx = randint(0, nb_matches-1)
            # Regenerate match if dupe
            while matches[idx] in rand_matches:
                idx = randint(0, nb_matches-1)
            rand_matches.append(matches[idx])
        rand_img1_pts = np.array([kp1[m.queryIdx].pt for m in rand_matches])
        rand_img2_pts = np.array([kp2[m.trainIdx].pt for m in rand_matches])
        # findHomography returns a 2-tuple, the first element is what we need
        H, _ = cv2.findHomography(rand_img1_pts, rand_img2_pts, 0)
        nb_inliers = computeInlierCount(H, matches, img1_pts, img2_pts, inlierThreshold)
        if nb_inliers > max_inliers:
            max_inliers = nb_inliers
            best_H = H
    # Compute refined homography using inliers
    inliers = get_inliers(best_H, matches, img1_pts, img2_pts, inlierThreshold)
    img1_inlier_pts = np.array([kp1[m.queryIdx].pt for m in inliers])
    img2_inlier_pts = np.array([kp2[m.trainIdx].pt for m in inliers])
    refined_H, _ = cv2.findHomography(img1_inlier_pts, img2_inlier_pts, 0)
    H_inv = np.linalg.inv(refined_H)
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, inliers, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    return (refined_H, H_inv, matched_img)

def get_inliers(H, matches, img1_pts, img2_pts, inlierThreshold):
    inliers = []
    projections = []
    for x, y in img1_pts:
        projections.append(project(x, y, H))

    match_idx = 0
    for x1, y1 in projections:
        x2, y2 = img2_pts[match_idx]
        # Calculate Euclidian distance
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if distance < inlierThreshold:
            inliers.append(matches[match_idx])
        match_idx += 1
    return inliers

def stitch(img1, img2, H, H_inv):
    # Step 4
    # Compute size of stitched_img
    # Projecting top right and bottom right corners of img2 give us all the info we need
    bottom_edge = img2.shape[0] - 1
    right_edge = img2.shape[1] - 1
    top_right_x, top_right_y = project(right_edge, 0, H_inv)
    bottom_right_x, bottom_right_y = project(right_edge, bottom_edge, H_inv)
    
    img1_height, img1_width = img1.shape[0], img1.shape[1]
    top_height_diff = abs(top_right_y) if top_right_y < 0 else 0
    bottom_height_diff = bottom_right_y - img1_height if bottom_right_y > img1_height else 0
    width_diff = max(top_right_x, bottom_right_x) - img1_width

    # First, create blank image with new dimensions (assume 3 channels)
    stitched_img_height = math.ceil(img1_height + top_height_diff + bottom_height_diff)
    stitched_img_width = math.ceil(img1_width + width_diff)
    stitched_img = np.zeros((stitched_img_height, stitched_img_width, 3), np.uint8)
    start_idx = math.floor(top_height_diff)
    # Then, dump the contents of img1 at the right place in stitched_img
    for row in range(img1_height):
        for col in range(img1_width):
            stitched_img[row + start_idx, col] = img1[row, col]
    # For every pixel in stitched_img, project onto img2
    for x in range(stitched_img_width):
        for y in range(-start_idx, stitched_img_height):
            proj_x, proj_y = project(x, y, H)
            # If projection is within img2's boundaries, add/blend img2's pixel onto stitched_img
            if (0 <= proj_x <= right_edge) and (0 <= proj_y <= bottom_edge):
                # Use bilinear interpolation to get img2's pixel
                patch = cv2.getRectSubPix(img2, (1,1), (proj_x, proj_y))[0]
                stitched_img[y + start_idx, x] = patch
    return stitched_img

# =========================PART 1=========================
# Most methods used for part 1 are built-in OpenCV methods
# Much of the code using these methods are adapted from OpenCV documentation/tutorials
# Get image gradients
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
bf = cv2.BFMatcher()

_, r1_desc = sift.compute(r1, r1_kp)
_, r2_desc = sift.compute(r2, r2_kp)
# Returns 2 best matches, to apply ratio test
matches = bf.knnMatch(r1_desc, r2_desc, 2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < RATIO_THRESHOLD*n.distance:
        good.append(m)
matched_img = cv2.drawMatches(r1, r1_kp, r2, r2_kp, good, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite(join(DIR_IMG, '2.png'), matched_img)
# =========================PART 1 END=========================


# =========================PART 2=========================
# Test with built-in OpenCV method; uncomment as needed
# perspectiveTransform expects a 3D array
# r1_pts_temp = np.array([r1_pts])
# projections_test should be the same as the return value of project()
# projections_test = cv2.perspectiveTransform(r1_pts_temp, H)

H, H_inv, matched_img = RANSAC(good, r1_kp, r2_kp, 200, 0.5, r1, r2)
cv2.imwrite(join(DIR_IMG, '3.png'), matched_img)

stitched_img = stitch(r1, r2, H, H_inv)
cv2.imwrite(join(DIR_IMG, '4.png'), stitched_img)
# =========================PART 2 END=========================