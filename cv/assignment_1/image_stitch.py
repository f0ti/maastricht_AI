#!/usr/bin/env python3

import skimage.feature as skfeature
import matplotlib.pyplot as plt
import numpy as np
import cv2

left_img = cv2.imread("sample_images/leftImage.png", cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread("sample_images/rightImage.png", cv2.IMREAD_GRAYSCALE)

cv2.imshow("left image gray", left_img)
cv2.imshow("right image gray", right_img)

left_harris_corners = cv2.cornerHarris(left_img, 3, 5, 0.04)
right_harris_corners = cv2.cornerHarris(right_img, 3, 5, 0.04)

cv2.imshow("left image", left_harris_corners)
cv2.imshow("right image", right_harris_corners)

cv2.waitKey(0)

'''
print(gray_left.shape)
print(gray_right.shape)

# left_harris_corners = np.where(left_harris_corners > .01 * left_harris_corners.max(), 255.0, 0.0)
# right_harris_corners = np.where(right_harris_corners > .01 * right_harris_corners.max(), 255.0, 0.0)

print(f'Left Harris corners\n {left_harris_corners}')
print(f'Right Harris corners\n {right_harris_corners}')

# left_key_points = [left_harris_features > .01 * left_harris_features.max()]

left_img[left_harris_corners > .01 * left_harris_corners.max()] = [0, 0, 255]x
right_img[right_harris_corners > .01 * right_harris_corners.max()] = [0, 0, 255]

cv2.imshow("left image", left_img)
cv2.imshow("right image", right_img)

# Normalizing Harris corners and display

# l_dst_norm = np.empty(left_harris_corners.shape, dtype=np.float32)
# left_corners = cv2.normalize(left_harris_corners, l_dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
# cv2.imshow("left corners", left_corners)

# r_dst_norm = np.empty(right_harris_corners.shape, dtype=np.float32)
# right_corners = cv2.normalize(right_harris_corners, r_dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
# cv2.imshow("right corners", right_corners)

lefty = skfeature.corner_harris(gray_left)
print(f'Lefty: {lefty}')
'''
left_harris_features = skfeature.corner_peaks(skfeature.corner_harris(gray_left), min_distance=5)
right_harris_features = skfeature.corner_peaks(skfeature.corner_harris(gray_right), min_distance=5)

plt.ylabel

print(np.float32(np.array([tuple(x) for x in left_harris_features])))
print(np.float32(np.array([tuple(x) for x in right_harris_features])))

# for l_feat, r_feat in zip(left_harris_features, right_harris_features):
#     cv2.circle(gray_left, l_feat, 3, (0, 0, 0))
#     cv2.circle(gray_right, r_feat, 3, (0, 0, 0))

# Keypoint descriptors using SIFT

sift = cv2.SIFT_create()

# left_kp = sift.detect(gray_left)
# left_kp = sift.detect(gray_left)

left_kp, left_desc = sift.compute(gray_left, (left_harris_features, 13))
right_kp, right_desc = sift.compute(gray_right, (right_harris_features, 13))

print(left_desc.shape)
print(right_desc.shape)
print(len(left_kp))
print(len(right_kp))

# for x in left_kp:
#     print("LEFT", (x.pt, x.angle))

left_kp, left_desc = sift.detectAndCompute(gray_left, None)
print(f"kp: {len(left_kp)}, desc {len(left_desc)}")

print(len(left_desc))

kp = cv2.KeyPoint()
print(kp)

# l_eye_corner_coordinates = cv2.connectedComponentsWithStats(left_harris_corners)[3][1:,:]
# l_eye_corner_keypoints = [cv2.KeyPoint(crd[0], crd[1], 13) for crd in l_eye_corner_coordinates]

# r_eye_corner_coordinates = cv2.connectedComponentsWithStats(right_harris_corners)[3][1:,:]
# r_eye_corner_keypoints = [cv2.KeyPoint(crd[0], crd[1], 13) for crd in r_eye_corner_coordinates]

# l_eye_corner_descriptors = [sift.compute(gray_left,[kp])[1] for kp in l_eye_corner_keypoints]
# print(l_eye_corner_descriptors)

left_kp, left_desc = sift.detectAndCompute(gray_left, None)
right_kp, right_desc = sift.detectAndCompute(gray_right, None)

brute_force_matcher = cv2.BFMatcher(crossCheck=True)
matches = brute_force_matcher.match(left_desc, right_desc)
# sort matches by way of matcher default distance function (l2 norm)
matches = sorted(matches, key = lambda x:x.distance)
print(f"Found {len(matches)} matches!")

matches_img = cv2.drawMatches(gray_left, left_kp, gray_right, right_kp, matches, None, flags=2)
cv2.imshow("matches", matches_img)

cv2.waitKey(0)

'''
