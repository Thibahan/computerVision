import numpy as np
from numpy.typing import NDArray
import cv2
import sys

from segmentation.math_utils import initPhi, heaviside, mu, sigma, p, dheaviside, curvature

sys.path.append("../opticalFlow")
from utils import get_image_pyramide


def get_phi(
    image: NDArray, phi: NDArray, phi_old: NDArray, v: float = 0.001, dt: float = 0.5
) -> NDArray:
    """Calculates phi in one step.

    Args:
        image (NDArray):  Image to segment in to fore- and background.
        phi (NDArray):  Current phi.
        phi_old (NDArray): phi from last step.
        v (float, optional): Weight parameter. Defaults to 0.001.
        dt (float, optional): Definces the influence of last iteration.
        Defaults to 0.5.

    Returns:
        NDArray: Returns phi
    """
    H = heaviside(phi)
    mu1 = mu(image, H, 0)
    mu2 = mu(image, H, 1)
    sig1 = sigma(image, H, 0, mu1)
    sig2 = sigma(image, H, 1, mu2)
    p1 = p(image, mu1, sig1)
    p2 = p(image, mu2, sig2)
    dH = dheaviside(phi, 1)
    with np.errstate(divide="ignore"):
        lgp1 = np.nan_to_num(np.log(p1))
        lgp2 = np.nan_to_num(np.log(p2))
    dphi = dH * (lgp1 - lgp2 + v * curvature(phi))
    phi = phi_old + dt * dphi
    return phi


def iter_phi(
    image: NDArray,
    phi: NDArray,
    phi_old: NDArray,
    it: int,
    v: float = 0.001,
    dt: float = 0.5,
) -> NDArray:
    """Calculates the iterations of phi.

    Args:
        image (NDArray):  Image to segment in to fore- and background.
        phi (NDArray): Current phi.
        phi_old (NDArray): phi from last step.
        it (int): Number of max iterations.
        v (float, optional): Weight parameter. Defaults to 0.001.
        dt (float, optional): Definces the influence of last iteration.
        Defaults to 0.5.

    Returns:
        NDArray: Returns phi
    """
    iter = 0
    cost_old = 0
    while iter < it:
        phi = get_phi(image, phi, phi_old, v, dt)

        # Set all values to -1 or 1
        phi[phi <= -1] = -1
        phi[phi > 1] = 1

        iter += 1

        # Break loop of cost loss is not changing.
        # Otherwise save old values for next iteration.
        cost = np.sum((phi - phi_old) ** 2)
        if (cost - cost_old) ** 2 <= 0.000001:
            break
        else:
            phi_old = phi
            cost_old = cost
    return phi, phi_old


def set_foreground(phi: NDArray, image: NDArray) -> NDArray:
    """This function will set the foreground and background based on variance.
    """
    if not (np.var(image[phi > 0]) > np.var(image[phi < 0])):
        phi[phi == 1] = -2
        phi[phi != -2] = 1
        phi[phi == -2] = -1
    return phi


def filter_phi(phi):
    # Handle to large foreground.
    x, _ = np.where(phi == 1)
    sx, sy = np.shape(phi)
    if x.shape[0] > 0.9 * (sx * sy):
        return np.zeros_like(phi)
    else:
        return phi


def segment(image: NDArray, it: int, v: float = 0.001, dt: float = 0.5) -> NDArray:
    """This function will segment an image to foreground and background
    using levelsets and active contours.
    The forground will have the value 1 and the background -1.
    The segmentation is based on the segmentaion in the chapter
    "Region Based Active Contours" in
    "From pixels to regions: partial differential equations in image analysis"
    by Thomas Brox.

    Args:
        image (NDArray): Image to segment in to fore- and background.
        it (int): Number of max iterations.
        v (float, optional): Weight parameter. Defaults to 0.001.
        dt (float, optional): Definces the influence of last iteration.
        Defaults to 0.5.

    Returns:
        NDArray: Returns an array where the forground is 1 and the background -1.
    """
    # Handle empty image
    if np.max(image) == 0:
        return np.zeros_like(image)

    # Initialize parameters
    h, w = np.shape(image)[0], np.shape(image)[1]
    phi = initPhi(h, w)
    phi_old = phi

    # Iterate until cost is smaller than tolerance
    phi, _ = iter_phi(image, phi, phi_old, it, v, dt)

    phi = set_foreground(phi, image)

    phi = filter_phi(phi)
    return phi


def multiscale_segment(
    image: NDArray,
    it: int,
    v: float = 0.001,
    dt: float = 0.5,
    kernel: tuple = (15, 15),
    scale: float = 0.5,
    sig: float = 100,
    size: int = 6,
) -> NDArray:
    """This function will segment an image to foreground and background
    using levelsets, active contours and a multiscale approach with an image
    pyramide. The forground will have the value 1 and the background -1.
    The segmentation is based on the segmentaion in the chapter
    "Region Based Active Contours" in
    "From pixels to regions: partial differential equations in image analysis"
    by Thomas Brox.

    Args:
        image (NDArray): Image to segment in to fore- and background.
        it (int): Number of max iterations.
        v (float, optional): Weight parameter. Defaults to 0.001.
        dt (float, optional): Definces the influence of last iteration.
        kernel (tuple, optional): Kernel for image pyramide filter.
        Defaults to (15, 15).
        scale (float, optional): Scale for image pyramide. Defaults to 0.5.
        sig (float, optional): Sigma for filter in image pyramide. Defaults to 100.
        size (int, optional): Size of image pyramide. Defaults to 6.

    Returns:
        NDArray: Returns an array where the forground is 1 and the background -1.
    """
    # Handle empty image
    if np.max(image) == 0:
        return np.zeros_like(image)

    # Calculate image pyramide
    image_pyramide = get_image_pyramide(image, kernel, scale, sig, size)
    image = image_pyramide[-1]

    # Initialize parameters
    h, w = np.shape(image)[0], np.shape(image)[1]
    phi = initPhi(h, w)
    phi_old = phi

    # Iterate over image pyramide
    for i in range(size):
        # Inverse image pyramide to get last image first
        i_n = -i - 1
        image = image_pyramide[i_n]

        # Iterate until cost is smaller than tolerance
        phi, phi_old = iter_phi(image, phi, phi_old, it, v, dt)

        # Resize phi and phi old, if not last image of pyramide
        if i < size - 1:
            phi = cv2.resize(
                phi,
                (
                    np.shape(image_pyramide[i_n - 1])[1],
                    np.shape(image_pyramide[i_n - 1])[0],
                ),
                interpolation=cv2.INTER_CUBIC,
            )
            phi_old = cv2.resize(
                phi_old,
                (
                    np.shape(image_pyramide[i_n - 1])[1],
                    np.shape(image_pyramide[i_n - 1])[0],
                ),
                interpolation=cv2.INTER_CUBIC,
            )

    phi = set_foreground(phi, image)

    phi = filter_phi(phi)
    return phi
