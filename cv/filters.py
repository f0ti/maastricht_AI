import cv2
import math
import numpy as np

'''
    Filters implemented:
        - gaussian
        - median
        - mean
'''

def gen_gaussian_kernel(sigma=1.0, size=3):

    return np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))

def gaussian(kernel, mask, sigma=2.0):

    # generate gaussian
    # gaussian_kernel = np.ravel(gen_gaussian_kernel(sigma, mask))

    gaussian_kernel = np.ravel(gen_gaussian_kernel(sigma, mask))

    gaussian = np.dot(kernel, gaussian_kernel / np.sum(gaussian_kernel))

    return gaussian

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

    assert mask % 2 != 0, "Please provide odd number kernel size"

    new_img = np.zeros_like(gray_img)

    # add padding, handle edge convolution by mirroring (replicating pixels inside image as border)

    bd = int(mask / 2)

    gray_img = np.pad(gray_img, [(bd, bd), (bd, bd)], 'edge')
    
    for i in range(bd, gray_img.shape[0] - bd):
        for j in range(bd, gray_img.shape[1] - bd):
            
            # get kernel according to mask
            kernel = np.ravel(gray_img[i - bd : i + bd + 1, j - bd : j + bd + 1])
            new_img[i-bd, j-bd] = filter(kernel, mask)

    # print(f"Padded image: {gray_img.shape}")
    # print(f"New image: {new_img.shape}")

    return new_img


if __name__ == "__main__":

    # read original image and turn image in grayscale value
    img = cv2.imread("test.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # print(f"Original image shape: {gray.shape}")

    # get values with two different mask size
    median3x3 = apply(median, gray, 3)
    mean3x3 = apply(mean, gray, 3)
    gaussian3x3 = apply(gaussian, gray, 3)

    # show result images
    cv2.imshow("original image", gray)
    cv2.imshow("median filter with 3x3 mask", median3x3)
    cv2.imshow("mean filter with 3x3 mask", mean3x3)
    cv2.imshow("gaussian filter with 3x3 mask", gaussian3x3)

    cv2.waitKey(0)
