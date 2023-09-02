import cv2
import numpy as np
import random
from PST import PST
from scipy import ndimage

STRENGTHS = [5, 10, 15, 20, 25]
CANNY_T_LOWER = 100
CANNY_T_UPPER = 200

def load_image(file_name : str):
    """
    Reads and returns in grayscale image, prepares data-output folder.
    :param file_name: Original image name.
    :return: OpenCV grayscale image.
    """
    img = cv2.imread('data-input/'+file_name, cv2.IMREAD_GRAYSCALE)
    # img = cv2.GaussianBlur(img_gray, (3, 3), 0)
    return img

def detect_edges(algorithm : str, image):
    """
    Detects edges with algorithm.
    :param algorithm: Algorithm name. Options: "Laplacian", "Canny", "Prewitt", "Sobel", "Scharr"
    :param image: OpenCV grayscale image.
    :return: detected edges (OpenCV image).
    """
    match algorithm:
        case 'Laplacian':
            dst = cv2.Laplacian(image, cv2.CV_16S, ksize=3)
            abs_dst = cv2.convertScaleAbs(dst)
            return abs_dst
        case 'Canny':
            res = cv2.Canny(image, CANNY_T_LOWER, CANNY_T_UPPER)
            return res
        case 'Prewitt':
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            img_prewittx = cv2.filter2D(image, -1, kernelx)
            img_prewitty = cv2.filter2D(image, -1, kernely)
            return img_prewittx + img_prewitty
        case 'Sobel':
            img_sobelx = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=5)
            img_sobely = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize=5)
            return img_sobelx + img_sobely
        case 'Scharr':
            scharr_X = cv2.Scharr(image, cv2.CV_64F, 1, 0)
            scharr_Y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
            return cv2.convertScaleAbs(scharr_X+scharr_Y)
        case 'Robert':
            roberts_cross_v = np.array([[1, 0],
                                        [0, -1]])

            roberts_cross_h = np.array([[0, 1],
                                        [-1, 0]])
            img = image.copy().astype('float64')
            img /= 255.0
            vertical = ndimage.convolve(img, roberts_cross_v)
            horizontal = ndimage.convolve(img, roberts_cross_h)

            res= np.sqrt(np.square(horizontal) + np.square(vertical))
            res *= 255
            return res
        case 'PST':
            return PST(image)*255
        case _:
            raise Exception('No such algorithm!')

def blur_images(img):
    """
    Blures images with averaging.
    :param img: image
    :return: array of images blurred with different kernels (5, 10, 15, 20, 25).
    """
    arr = []
    for i in STRENGTHS:
        arr.append(cv2.blur(img, (i, i)))
    return arr

def scatter_images(img):
    """
    Scatters edges with different strengths.
    :param img: original image
    :return: array of scattered images with different kernels (5, 10, 15, 20, 25)
    """
    edges = cv2.Canny(img, 100, 200)
    edge_pixels = np.where(edges != 0)
    arr = []
    for i in STRENGTHS:
        edge_pixels = np.column_stack(np.where(edges != 0))
        scattered_pixels = edge_pixels + np.random.normal(0, i, size=edge_pixels.shape).astype(int)
        scattered_pixels[:, 0] = np.clip(scattered_pixels[:, 0], 0, img.shape[0] - 1)
        scattered_pixels[:, 1] = np.clip(scattered_pixels[:, 1], 0, img.shape[1] - 1)
        scattered_edges = img.copy()
        scattered_edges[edge_pixels[:, 0], edge_pixels[:, 1]] = scattered_edges[
            scattered_pixels[:, 0], scattered_pixels[:, 1]]
        arr.append(scattered_edges)

    return arr

def noise_images(img):
    arr = []
    row, col = img.shape
    for i in STRENGTHS:
        or_image = img.copy()
        # Add salt
        for j in range(i*400):
            x_coord = random.randint(0, col-1)
            y_coord = random.randint(0, row-1)
            or_image[y_coord][x_coord] = 255

        # Add pepper
        for j in range(i*400):
            x_coord = random.randint(0, col-1)
            y_coord = random.randint(0, row-1)
            or_image[y_coord][x_coord] = 0

        arr.append(or_image)
    return arr
def gamma_d_images(img):
    arr = []
    for i in STRENGTHS:
        gamma = i * 0.25
        table = np.array([((s / 255.0) ** gamma) * 255
                          for s in np.arange(0, 256)]).astype("uint8")
        arr.append(cv2.LUT(img, table))
    return arr
def gamma_w_images(img):
    arr = []
    for i in STRENGTHS:
        gamma = 1/(i * 0.25)
        table = np.array([((s / 255.0) ** gamma) * 255
                          for s in np.arange(0, 256)]).astype("uint8")
        arr.append(cv2.LUT(img, table))
    return arr
def motion_blur_h_images(img):
    arr = []
    for i in STRENGTHS:
        kernel_h = np.zeros((i, i))
        kernel_h[int((i - 1) / 2), :] = np.ones(i)
        kernel_h /= i
        arr.append(cv2.filter2D(img, -1, kernel_h))
    return arr
def motion_blur_v_images(img):
    arr = []
    for i in STRENGTHS:
        kernel_v = np.zeros((i, i))
        kernel_v[:, int((i - 1)/2)] = np.ones(i)
        kernel_v /= i
        arr.append(cv2.filter2D(img, -1, kernel_v))
    return arr