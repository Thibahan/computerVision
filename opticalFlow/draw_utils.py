import numpy as np
from numpy.typing import NDArray
import cv2


def na(array: NDArray) -> NDArray:
    """Normalize a array between 0 and 1.

    Args:
        array (NDArray): Input array.

    Returns:
        NDArray: Normalized output array.
    """
    array = np.float32(array)
    return np.float32((array - np.min(array))/(np.max(array)-np.min(array)))


def mbcolor(flow: NDArray) -> NDArray:
    """Calculates the middlebury color for optical flow array

    Args:
        flow (NDArray): Input optical flow.

    Returns:
        NDArray: Optical flow in middlebury color scheme optical flow.
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    hsv = np.zeros([u.shape[0], u.shape[1], 3], np.uint8())
    hsv[..., 2] = 255

    mag, ang = cv2.cartToPolar(u, v)
    hsv[..., 0] = (ang * 180 / np.pi / 2)
    hsv[..., 1] = na(mag)*255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb
