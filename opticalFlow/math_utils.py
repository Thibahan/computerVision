import numpy as np
from numpy.typing import NDArray
import cv2


def grad_x(x: NDArray) -> NDArray:
    """Calculates the gradient in x direction of array-

    Args:
        x (NDArray): Input array.

    Returns:
        NDArray: Gradient in x direction of input.
    """
    dx = (np.roll(x, 1, axis=1) - np.roll(x, -1, axis=1)) / 2.0

    dx[0, :] = 0.0
    dx[:, -1] = 0
    dx[:, 0] = 0
    dx[-1, :] = 0

    return dx


def grad_y(x: NDArray) -> NDArray:
    """Calculates the gradient in y direction of array-

    Args:
        x (NDArray): Input array.

    Returns:
        NDArray: Gradient in y direction of input.
    """
    dy = (np.roll(x, 1, axis=0) - np.roll(x, -1, axis=0)) / 2.0

    dy[0, :] = 0.0
    dy[:, -1] = 0
    dy[:, 0] = 0
    dy[-1, :] = 0

    return dy


def grad_xy(x: NDArray) -> NDArray:
    """Calculates the gradient in xy direction of array-

    Args:
        x (NDArray): Input array.

    Returns:
        NDArray: Gradient in xy direction of input.
    """
    dxy = (
        np.roll(x, (1, 1))
        - np.roll(x, (1, -1))
        - np.roll(x, (-1, 1))
        + np.roll(x, (-1, -1))
    ) / 4.0

    dxy[0, :] = 0.0
    dxy[:, -1] = 0
    dxy[:, 0] = 0
    dxy[-1, :] = 0

    return dxy


def norm_of_gradient(term: NDArray) -> NDArray:
    """Calculates the normalized gradient of input array.

    Args:
        term (NDArray): Input array.

    Returns:
        NDArray: Normalized gradient.
    """
    term_x = (np.roll(term, 1, axis=1) - np.roll(term, -1, axis=1)) / 2.0
    term_y = (np.roll(term, 1, axis=0) - np.roll(term, -1, axis=0)) / 2.0

    term_x[0, :] = 0.0
    term_x[:, -1] = 0
    term_x[:, 0] = 0
    term_x[-1, :] = 0

    term_y[0, :] = 0.0
    term_y[:, -1] = 0
    term_y[:, 0] = 0
    term_y[-1, :] = 0

    norm = np.sqrt(term_x**2 + term_y**2)
    return norm


def robust_function_derivative(data_term: NDArray, epsilon: float) -> NDArray:
    """Calculated the derivative of the robust penalizer function
    from input array.

    Args:
        data_term (NDArray): Input array.
        epsilon (float): Penalizer function parameter.

    Returns:
        NDArray: Derivative of the robust penalizer function.
    """
    return 1.0 / (2 * np.sqrt(data_term + epsilon**2))


def sum_convolution(im: NDArray) -> NDArray:
    """Calculates the sum over convolution.

    Args:
        im (NDArray): Input image.

    Returns:
        NDArray: Sum.
    """
    kernel = np.array([[0.0, 0.5, 0.0],
                       [0.5, 0.0, 0.5],
                       [0.0, 0.5, 0.0]],
                      np.float32)
    im_sum = cv2.filter2D(im, -1, kernel, cv2.BORDER_REFLECT_101)

    return im_sum


def sum_convolution_neighbours(im: NDArray) -> NDArray:
    """Calculates the sum of neightbours over convolution.

    Args:
        im (NDArray): Input image.

    Returns:
        NDArray: Sum.
    """
    kernel = np.array([[0.0, 0.5, 0.0],
                       [0.5, 2.0, 0.5],
                       [0.0, 0.5, 0.0]],
                      np.float32)
    im_sum = cv2.filter2D(im, -1, kernel, cv2.BORDER_REFLECT_101)

    return im_sum
