from xxlimited import new
import numpy as np
from matplotlib.pyplot import imshow
import cv2

def resize(gray_img, *target):

    '''
        Nearest neighbor interpolation
        https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
    '''

    new_img = np.zeros(target, dtype=np.uint8)

    h_ratio, w_ratio = gray_img.shape[0] / target[0], gray_img.shape[1] / target[1]  # height ratio, width ratio

    for i in range(target[0]):
        for j in range(target[1]):
            new_img[i, j] = int(gray_img[int(h_ratio * i), int(w_ratio * j)])

    return new_img

if __name__ == "__main__":

    # read original image and turn image in grayscale value
    img = cv2.imread("test.jpg") # (194, 259)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    src_h, src_w = gray.shape    
    # set ratio or hard-coded target dimensions
    RATIO = 0.5
    trg_h, trg_w = int(src_h * RATIO), int(src_w * RATIO)
    resized = resize(gray, trg_h, trg_w)

    # show result images
    cv2.imshow("original image", gray)
    cv2.imshow("resized image", resized)

    cv2.waitKey(0)
