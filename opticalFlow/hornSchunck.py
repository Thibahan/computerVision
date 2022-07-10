import numpy as np
from numpy.typing import NDArray
from traitlets import Int

from utils import get_gradient, get_mean_flow


def horn_schunck_flow(im1: NDArray, im2: NDArray,
                      alpha: Int = 1,
                      n_iter: Int = 100) -> NDArray:
    """This function will calculate the optilca flow in the Horn-Schunck method.
    Find more informations under: "Determining Optical Flow"
    by Berthold K.P. Horn and Brian G. Schunck

    Args:
        im1 (NDArray): Image one.
        im2 (NDArray): Image two.
        alpha (Int, optional): Parameter to control the weight
        of the smoothness term compared to the optical flow
        constrain. Defaults to 1.
        n_iter (Int, optional): Number of interations. Defaults to 100.

    Returns:
        NDArray: _description_
    """
    # Calculate the image gradients
    fx, fy, ft = get_gradient(im1, im2)

    # Get shape of image and create initial flow, u and v
    shape = im1.shape
    flow = np.empty([shape[0], shape[1], 2], dtype=np.float32)
    u_initial = np.zeros([im1.shape[0], im1.shape[1]], dtype=np.float32)
    v_initial = np.zeros([im1.shape[0], im1.shape[1]], dtype=np.float32)

    u = u_initial
    v = v_initial

    # Calculate the optical flow in u and c direction
    for _ in range(n_iter):
        u_avg, v_avg = get_mean_flow(u, v)
        der = (fx * u_avg + fy * v_avg + ft) / (alpha ** 2 + fx ** 2 + fy ** 2)
        u = u_avg - fx * der
        v = v_avg - fy * der

    # Create flow array
    flow[:, :, 0] = u
    flow[:, :, 1] = v
    return flow
