import numpy as np
import cv2


def get_image_pyramide(image: np.ndarray, kernel: tuple,
                       scale: float, sigma: float,
                       size: int) -> list:
    """Calculates a image pyramide and returns the images of the
    pyramide as a list.

    Args:
        image (np.ndarray): Image as a numpy array.
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
