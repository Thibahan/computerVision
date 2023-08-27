import numpy as np
from numpy.typing import NDArray


def initPhi(h: int, w: int, e: float = 5.0) -> NDArray:
    """This function will initalize phi for active contours.

    Args:
        h (int): Height of image.
        w (int): Width of image.
        e (float, optional): Adjustable parameter for initializing phi.
        Defaults to 5.0.

    Returns:
        NDArray: Returns the initialized phi.
    """
    phi_init = np.zeros([h, w])
    for i in range(0, h):
        for j in range(0, w):
            phi_init[i, j] = np.sin(i * np.pi / e) * np.sin(j * np.pi / e)
    return phi_init


def heaviside(phi):
    """Calculates the heaviside step function of input."""
    return 1 / 2.0 * (1 + 2 / np.pi * np.arctan(phi / 1.0))


def dheaviside(phi, e):
    """Calculates the derivative of heaviside step function of input."""
    return 1 / (np.pi * (e**2 + phi**2))


def abs_grad(phi):
    """Calculates the absolute gradient of input."""
    grad_y, grad_x = np.gradient(phi)
    return np.sqrt(grad_x**2 + grad_y**2)


def div(grad_y, grad_x):
    """Calculates the divergence of the input.

    Args:
        grad_y : y-gradient.
        grad_x : x-gradient.

    Returns:
        _type_: _description_
    """
    grad_yy, _ = np.gradient(grad_y)
    _, grad_xx = np.gradient(grad_x)
    return grad_xx + grad_yy


def mu(
    image: NDArray, H: NDArray, s: int, phi: NDArray = np.zeros_like([5, 5])
) -> NDArray:
    """Calculates my based on equation (5.33) in
    "From pixels to regions: partial differential equations in image analysis"
    by Thomas Brox.

    Args:
        image (NDArray): Image to calculate my.
        H (NDArray): Heaviside function output of phi.
        s (int): Type of my.
        phi (NDArray, optional): Kernel. Defaults to np.zeros_like([5, 5]).

    Returns:
        NDArray: Returns my.
    """
    invphi = (~phi.astype(bool)).astype(int)
    if s == 0:
        n = np.sum(image * H)
        d = np.sum(H)
    elif s == 1:
        n = np.sum(image * (1.0 - H))
        d = np.sum(1.0 - H)
    elif s == 2:
        n = np.sum(image * H * phi)
        d = np.sum(H * phi)
    elif s == 3:
        n = np.sum(image * (1.0 - H * invphi))
        d = np.sum(1.0 - H * invphi)
    else:
        print("s=0 for mu1 and s=1 for mu2")
    return n / d


def curvature(f):
    """Calculates the curvature of the input."""
    fy, fx = np.gradient(f)
    norm = np.sqrt(fx**2 + fy**2)
    Nx = fx / (norm + 1e-8)
    Ny = fy / (norm + 1e-8)
    return div(Ny, Nx)


def sigma(
    image: NDArray,
    H: NDArray,
    s: int,
    mu: NDArray,
    phi: NDArray = np.zeros_like([5, 5]),
) -> NDArray:
    """Calculates sigma based on equation (5.33) in
    "From pixels to regions: partial differential equations in image analysis"
    by Thomas Brox.

    Args:
        image (NDArray): Input image.
        H (NDArray): Output of heaviside function og phi.
        s (int): Type of sigma.
        mu (NDArray): My calculated based of equation (5.33)
        phi (NDArray, optional): Kernel. Defaults to np.zeros_like([5, 5]).

    Returns:
        NDArray: Returns sgima.
    """
    invphi = (~phi.astype(bool)).astype(int)
    if s == 0:
        n = np.sum((image - mu) ** 2 * H)
        d = np.sum(H)
    elif s == 1:
        n = np.sum((image - mu) ** 2 * (1.0 - H))
        d = np.sum(1.0 - H)
    elif s == 2:
        n = np.sum((image - mu) ** 2 * H * phi)
        d = np.sum(H * phi)
    elif s == 3:
        n = np.sum((image - mu) ** 2 * (1.0 - H * invphi))
        d = np.sum(1.0 - H * invphi)
    else:
        print("s=0 for sigma1 and s=1 for sigma2")
    return np.sqrt(n / d)


def p(s, mu, sig):
    """Calculates p based on equation (5.31) in
    "From pixels to regions: partial differential equations in image analysis"
    by Thomas Brox.
    """
    temp1 = 1 / (np.sqrt(2 * np.pi) * sig)
    temp2 = -(((s - mu) ** 2) / (2 * sig**2))
    p = temp1 * np.exp(temp2)
    return p
