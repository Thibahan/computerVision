import numpy as np
from numpy.typing import NDArray
import cv2

from opticalFlow.utils import (get_gradient, get_mean_flow,
                               get_image_pyramide, warp_flow)


def horn_schunck_flow(im1: NDArray, im2: NDArray,
                      alpha: int = 1,
                      n_iter: int = 100) -> NDArray:
    """This function will calculate the optilca flow in the Horn-Schunck method.
    Find more informations under: "Determining Optical Flow"
    by Berthold K.P. Horn and Brian G. Schunck

    Args:
        im1 (NDArray): Image one.
        im2 (NDArray): Image two.
        alpha (int, optional): Parameter to control the weight
        of the smoothness term compared to the optical flow
        constrain. Defaults to 1.
        n_iter (int, optional): Number of interations. Defaults to 100.

    Returns:
        NDArray: Calculated optical flow.
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


def horn_schunck_pyr_flow(im1: NDArray, im2: NDArray,
                          alpha: int = 1, n_iter: int = 100,
                          kernel: tuple = (5, 5), eta: float = 0.8,
                          sigma: int = 1, size: int = 5) -> NDArray:
    """This function will calculate the optilca flow in the pyramidal
    Horn-Schunck method.
    Find more informations under: "Determining Optical Flow"
    by Berthold K.P. Horn and Brian G. Schunck and
    "Reliable estimation of dense optical flow fields with large displacements"
    by Luis Alvarez, Joachim Weickert, and Javier Sanchez

    Args:
        im1 (NDArray): Image one.
        im2 (NDArray): Image two.
        alpha (int, optional):  Parameter to control the weight
        of the smoothness term compared to the optical flow
        constrain. Defaults to 1.
        n_iter (int, optional): Number of interations. Defaults to 100.
        kernel (tuple, optional): Kernel size for the gaussian blur for
        creating the image pyramide. Defaults to (5, 5).
        eta (float, optional): Steps for creating the image pyramide.
        Defaults to 0.8.
        sigma (int, optional): Kernel standard deviation for image pyramide.
        Defaults to 1.
        size (int, optional): Size of image pyramide. Defaults to 5.

    Returns:
        NDArray: Calculated optical flow.
    """
    # calculate image pyramides
    p_im1 = get_image_pyramide(im1, kernel, eta, sigma, size)
    p_im2 = get_image_pyramide(im2, kernel, eta, sigma, size)
    scales_list = list(reversed([x.shape for x in p_im1]))
    coars_x = scales_list[0][0]
    coars_y = scales_list[0][1]
    # Initialze flow with zero
    flow = np.zeros([coars_x, coars_y, 2])
    # calculate pyr flow
    i = 1
    for (img1, img2) in zip(reversed(p_im1), reversed(p_im2)):
        # warp second image in next scale
        img2 = warp_flow(img2, flow)
        if i != len(scales_list):
            scale = scales_list[i]
        i += 1
        # calculate flow
        n_flow = horn_schunck_flow(img1, img2, alpha, n_iter)
        # add flow to init flow
        flow += n_flow
        # resize flow to next scale
        u = cv2.resize(flow[:, :, 0], tuple(reversed(scale)))
        v = cv2.resize(flow[:, :, 1], tuple(reversed(scale)))
        flow = np.zeros([scale[0], scale[1], 2])
        flow[:, :, 0] = u
        flow[:, :, 1] = v
    return flow
