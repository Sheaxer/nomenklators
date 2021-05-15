import cv2
import numpy as np

"""OTSU binarization"""

def binarize_image(img, inv =True):
    if inv:
        threshold, img_binarized = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        threshold, img_binarized = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_binarized


"""Apply CLAHE to equalize image"""


def equalize_image(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


"""Gabor filter kernel"""


def _get_kernel(theta) -> float:
    ksize = 31
    return cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)


"""Apply Gabor filter in the desired angle"""


def filter_image(img, theta=np.pi):
    kernel = _get_kernel(theta)
    return cv2.filter2D(img, -1, kernel)


"""Invert values of image"""

def invert(img):
    return cv2.bitwise_not(img)


"""Preprocessing of image - equalization using CLAHE"""


def preprocess_img(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # noise_removal = cv2.GaussianBlur(gray, (5,5),3)
    # noise_removal = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
    equalized = equalize_image(gray)

    return equalized, gray


def get_binary_after_gabor(gray):
    img_filtered_vertical = filter_image(gray.copy(), theta=np.pi)
    img_vertical_binarized = binarize_image(img_filtered_vertical)
    #cv2.imwrite("export/bin_vert.jpg", img_vertical_binarized)
    img_filtered_horizontal = filter_image(gray.copy(), theta=np.pi / 2)
    img_horizontal_binarized = binarize_image(img_filtered_horizontal)
    #cv2.imwrite("export/bin_horizont.jpg", img_horizontal_binarized)
    img = cv2.bitwise_or(img_vertical_binarized, img_horizontal_binarized)
    return img


def get_vertical_contours_gabor(equalized_image):
    img_filtered = filter_image(equalized_image, theta=np.pi)

    # try applying opening to filter out remainders?
    _, img_vertical_binarized = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours_vertical, hierarchy_horizontal = cv2.findContours(img_vertical_binarized, cv2.RETR_LIST,
                                                               cv2.CHAIN_APPROX_NONE)
    return contours_vertical


def get_horizontal_contours_gabor(equalized_image):
    img_filtered = filter_image(equalized_image, theta=np.pi / 2)
    _, img_horizontal_binarized = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours_horizontal, hierarchy_horizontal = cv2.findContours(img_horizontal_binarized, cv2.RETR_LIST,
                                                                 cv2.CHAIN_APPROX_NONE)
    return contours_horizontal
