import cv2
import numpy as np

'''
    Filters implemented:
        - median
        - mean
'''

'''
    TODO:
        - apply blur in the border
        - implement gaussian blur
'''


def gaussian(kernel, mask, sigma=1.0):

    # print(np.random.uniform(2))

    # # calculate mask gaussian
    # mean = np.divide(np.dot(kernel, window), np.sum(window))

    return mean

def median(kernel, mask):

    # calculate mask median
    median = np.sort(kernel)[np.int8(np.divide((np.multiply(mask, mask)), 2) + 1)]

    return median

def mean(kernel, mask):

    window = np.ravel(np.ones((mask, mask)))
    # calculate mask mean
    mean = np.divide(np.dot(kernel, window), np.sum(window))

    return mean

def apply(filter, gray_img, mask=3):

    """
        :param gray_img: gray image
        :param mask: mask size
        :return: image with median filter
    """

    # set image borders
    bd = int(mask / 2)
    # copy image size
    median_img = np.zeros_like(gray_img)

    for i in range(gray_img.shape[0] - bd):
        for j in range(bd, gray_img.shape[1] - bd):
            
            # get kernel according to mask
            kernel = np.ravel(gray_img[i - bd : i + bd + 1, j - bd : j + bd + 1])
            median_img[i, j] = filter(kernel, mask)

    return median_img


if __name__ == "__main__":

    # read original image and turn image in gray scale value
    img = cv2.imread("test.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get values with two different mask size
    median3x3 = apply(median, gray, 3)
    mean3x3 = apply(mean, gray, 3)

    # show result images
    cv2.imshow("original image", gray)
    cv2.imshow("median filter with 3x3 mask", median3x3)
    cv2.imshow("mean filter with 3x3 mask", mean3x3)

    cv2.waitKey(0)
