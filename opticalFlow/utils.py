from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import cv2
from scipy.signal import convolve2d


def get_image_pyramide(image: NDArray, kernel: tuple,
                       scale: float, sigma: float,
                       size: int) -> list:
    """Calculates a image pyramide and returns the images of the
    pyramide as a list.

    Args:
        image (NDArray): Image as a numpy array.
        kernel (tuple): Kernel size for the gaussian blur.
        scale (float): Scale to shrink the images.
        sigma (float): Kernel standard deviation.
        size (int): Size of the image pyramide.

    Raises:
        ValueError: Raise error of the image pyramide can't get created due to
        too large size or too small scale.

    Returns:
        list: Image pyramide as a list. The first image is the largest one.
    """
    if (scale**size)*min(image.shape[0:2]) < 1:
        error_msg = "Can't create image pyramide because the shape of "\
                    "one image is too small. Reduce the size or "\
                    "increase the scale."
        raise ValueError(error_msg)
    image_pyramide = []
    for _ in range(size):
        image_pyramide.append(image)
        height = np.shape(image)[1]
        width = np.shape(image)[0]
        image = cv2.GaussianBlur(image, kernel, sigma)
        dimension = (int(height*scale), int(width*scale))
        image = cv2.resize(image, dimension, interpolation=cv2.INTER_CUBIC)

    return image_pyramide


def get_gradient(im1: NDArray,
                 im2: NDArray,
                 kernelX: NDArray = np.array([[-1, 1], [-1, 1]]) * 0.25,
                 kernelY: NDArray = np.array([[-1, -1], [1, 1]]) * 0.25,
                 kernelT: NDArray = np.ones((2, 2)) * 0.25) \
                 -> Tuple[NDArray, NDArray, NDArray]:
    """This function calulates the image gradient in x, y and t
    direction for two images.

    Args:
        im1 (NDArray): Image one.
        im2 (NDArray): Image two.
        kernelX (NDArray, optional): Kernel for image gradient.
        Defaults to np.array([[-1, 1], [-1, 1]])*0.25.
        kernelY (NDArray, optional): Kernel for image gradient.
        Defaults to np.array([[-1, -1], [1, 1]])*0.25.
        kernelT (NDArray, optional): Kernel for image gradient.
        Defaults to np.ones((2, 2))*0.25.

    Returns:
        Tuple[NDArray, NDArray, NDArray]: Image gradients in x, y and t.
    """
    fx = convolve2d(im1, kernelX, "same") + convolve2d(im2, kernelX, "same")
    fy = convolve2d(im1, kernelY, "same") + convolve2d(im2, kernelY, "same")

    ft = convolve2d(im1, kernelT, "same") + convolve2d(im2, -kernelT, "same")
    return fx, fy, ft
