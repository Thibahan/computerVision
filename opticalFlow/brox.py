import numpy as np
from opticalFlow.math_utils import (
    grad_x,
    grad_y,
    grad_xy,
    norm_of_gradient,
    robust_function_derivative,
    sum_convolution,
    sum_convolution_neighbours,
)
import cv2


def brox_optical_flow(
    flow0,
    im0,
    im1,
    alpha,
    gamma_d,
    gamma_g,
    iterations_fx2,
    iterations_fx3,
):
    # Get shape of image
    m0, n0 = im0.shape[:2]
    m, n = im0.shape[:2]

    # Initialize empty flow
    flow = np.zeros((m0, n0, 2), np.float32)

    zae = 0

    u_k = flow0[:, :, 0]
    v_k = flow0[:, :, 1]

    du_l = np.zeros((m, n), np.float32)
    du_m = np.zeros((m, n), np.float32)

    dv_l = np.zeros((m, n), np.float32)
    dv_m = np.zeros((m, n), np.float32)
    map_xy = np.indices((m, n), np.float32)
    map_x = map_xy[1]
    map_y = map_xy[0]

    im1_s_shift = cv2.remap(
        np.float32(im1),
        np.float32(map_x - u_k),
        np.float32(map_y - v_k),
        cv2.INTER_CUBIC,
        cv2.BORDER_CONSTANT,
        0,
    )

    im1_s_shift_x = grad_x(im1_s_shift)
    im1_s_shift_xx = grad_x(im1_s_shift_x)
    im1_s_shift_y = grad_y(im1_s_shift)
    im1_s_shift_yy = grad_y(im1_s_shift_y)
    im1_s_shift_xy = grad_xy(im1_s_shift)

    im0_s_x = grad_x(im0)
    im0_s_y = grad_y(im0)

    it_s = im1_s_shift - im0
    ixt_s = im1_s_shift_x - im0_s_x
    iyt_s = im1_s_shift_y - im0_s_y

    du_l = np.zeros((m, n), np.float32)
    du_m = np.zeros((m, n), np.float32)

    dv_l = np.zeros((m, n), np.float32)
    dv_m = np.zeros((m, n), np.float32)

    for _ in range(iterations_fx2):
        diff_term = robust_function_derivative(
            norm_of_gradient(u_k + du_l) ** 2 + norm_of_gradient(v_k + dv_l) ** 2,
            0.001,
        )

        data_term = gamma_d * robust_function_derivative(
            (it_s + du_l * im1_s_shift_x + dv_l * im1_s_shift_y) ** 2, 0.001
        )
        gradient_term = gamma_g * robust_function_derivative(
            (ixt_s + du_l * im1_s_shift_xx + dv_l * im1_s_shift_xy) ** 2
            + (iyt_s + du_l * im1_s_shift_xy + dv_l * im1_s_shift_yy) ** 2,
            0.001,
        )

        denominator_u = (
            data_term * (im1_s_shift_x**2)
            + gradient_term * (im1_s_shift_xx**2 + im1_s_shift_xy**2)
            + alpha * sum_convolution_neighbours(diff_term)
        )
        denominator_v = (
            data_term * (im1_s_shift_y**2)
            + gradient_term * (im1_s_shift_yy**2 + im1_s_shift_xy**2)
            + alpha * sum_convolution_neighbours(diff_term)
        )
        alpha_d_u = alpha / denominator_u
        alpha_d_v = alpha / denominator_v
        product_1_u = alpha * sum_convolution_neighbours(diff_term) * u_k
        product_1_v = alpha * sum_convolution_neighbours(diff_term) * v_k

        for _ in range(iterations_fx3):
            diff_div_u = diff_term * sum_convolution(u_k + du_m) + sum_convolution(
                diff_term * (u_k + du_m)
            )
            diff_div_v = diff_term * sum_convolution(v_k + dv_m) + sum_convolution(
                diff_term * (v_k + dv_m)
            )

            du_m = alpha_d_u * diff_div_u - (
                data_term * (im1_s_shift_x * (it_s + im1_s_shift_y * dv_m))
                + gradient_term
                * (
                    im1_s_shift_xx * (ixt_s + im1_s_shift_xy * dv_m)
                    + im1_s_shift_xy * (iyt_s + im1_s_shift_yy * dv_m)
                )
                + product_1_u
            ) / (denominator_u)

            dv_m = alpha_d_v * diff_div_v - (
                data_term * (im1_s_shift_y * (it_s + im1_s_shift_x * du_m))
                + gradient_term
                * (
                    im1_s_shift_xy * (ixt_s + im1_s_shift_xx * du_m)
                    + im1_s_shift_yy * (iyt_s + im1_s_shift_xy * du_m)
                )
                + product_1_v
            ) / (denominator_v)
            zae = zae + 1

        du_l = du_m
        dv_l = dv_m

        u_k = u_k + du_l
        v_k = v_k + dv_l

        u_k = cv2.medianBlur(np.float32(u_k), 3)
        v_k = cv2.medianBlur(np.float32(v_k), 3)

        u_k[0, :] = u_k[1, :]
        v_k[0, :] = v_k[1, :]

        u_k[m - 1, :] = u_k[m - 2, :]
        v_k[m - 1, :] = v_k[m - 2, :]

        u_k[:, 0] = u_k[:, 1]
        v_k[:, 0] = v_k[:, 1]

        u_k[:, n - 1] = u_k[:, n - 2]
        v_k[:, n - 1] = v_k[:, n - 2]

    flow[:, :, 0] = u_k
    flow[:, :, 1] = v_k

    return flow


def brox_pyr_optical_flow(
    flow0,
    im0,
    im1,
    alpha,
    gamma_d,
    gamma_g,
    sigma,
    w_sigma,
    nabla,
    iterations_fx1,
    iterations_fx2,
    iterations_fx3,
    verbose=False,
):
    m0, n0 = im0.shape[:2]

    flow = np.zeros((m0, n0, 2), np.float32)
    M = int(m0 * (1.0 + (1.0 - nabla**iterations_fx1) / (1.0 - nabla)) + 1)
    im1_pyr = np.zeros((M, n0), np.float32)
    im0_pyr = np.zeros((M, n0), np.float32)
    pyr_pos = np.zeros((iterations_fx1 + 1, 3), np.uint16)

    zae = 0
    maxiter = (iterations_fx1 + 1) * iterations_fx2 * iterations_fx3
    im0_scaled = im0 * 1.0
    im1_scaled = im1 * 1.0

    im1_pyr[:m0, :n0] = im1_scaled
    im0_pyr[:m0, :n0] = im0_scaled

    m, n = m0, n0
    m_pyr, _ = m0, n0
    pyr_pos[0, :] = np.c_[0, m, n]

    for l in range(1, iterations_fx1 + 1):
        m = int(np.floor(m0 * (nabla) ** l))
        n = int(np.floor(n0 * (nabla) ** l))

        m_pyr = m_pyr + m

        pyr_pos[l, :] = np.c_[m_pyr - m, m_pyr, n]

        im1_smooth = cv2.GaussianBlur(im1_scaled, (w_sigma, w_sigma), sigma)
        im0_smooth = cv2.GaussianBlur(im0_scaled, (w_sigma, w_sigma), sigma)

        im1_scaled = cv2.resize(im1_smooth, (n, m), interpolation=cv2.INTER_CUBIC)
        im1_pyr[m_pyr - m : m_pyr, :n] = im1_scaled

        im0_scaled = cv2.resize(im0_smooth, (n, m), interpolation=cv2.INTER_CUBIC)
        im0_pyr[m_pyr - m : m_pyr, :n] = im0_scaled

    for k in range(iterations_fx1 + 1):
        im1_s = im1_pyr[
            pyr_pos[iterations_fx1 - k, 0] : pyr_pos[iterations_fx1 - k, 1],
            : pyr_pos[iterations_fx1 - k, 2],
        ]
        im0_s = im0_pyr[
            pyr_pos[iterations_fx1 - k, 0] : pyr_pos[iterations_fx1 - k, 1],
            : pyr_pos[iterations_fx1 - k, 2],
        ]
        m, n = im1_s.shape[:2]

        map_xy = np.indices((m, n), np.float32)
        map_x = map_xy[1]
        map_y = map_xy[0]

        if k == 0:
            u_k = cv2.resize(flow0[:, :, 0], (n, m), interpolation=cv2.INTER_CUBIC) * (
                nabla**l
            )
            v_k = cv2.resize(flow0[:, :, 1], (n, m), interpolation=cv2.INTER_CUBIC) * (
                nabla**l
            )

            du_l = np.zeros((m, n), np.float32)
            du_m = np.zeros((m, n), np.float32)

            dv_l = np.zeros((m, n), np.float32)
            dv_m = np.zeros((m, n), np.float32)

        if k > 0:
            u_k = cv2.resize(u_k * (1.0 / nabla), (n, m), cv2.INTER_LINEAR)
            v_k = cv2.resize(v_k * (1.0 / nabla), (n, m), cv2.INTER_LINEAR)

        im1_s_shift = cv2.remap(
            np.float32(im1_s),
            np.float32(map_x - u_k),
            np.float32(map_y - v_k),
            cv2.INTER_CUBIC,
            cv2.BORDER_CONSTANT,
            0,
        )

        im1_s_shift_x = grad_x(im1_s_shift)
        im1_s_shift_xx = grad_x(im1_s_shift_x)
        im1_s_shift_y = grad_y(im1_s_shift)
        im1_s_shift_yy = grad_y(im1_s_shift_y)
        im1_s_shift_xy = grad_xy(im1_s_shift)

        im0_s_x = grad_x(im0_s)
        im0_s_y = grad_y(im0_s)

        it_s = im1_s_shift - im0_s
        ixt_s = im1_s_shift_x - im0_s_x
        iyt_s = im1_s_shift_y - im0_s_y

        du_l = np.zeros((m, n), np.float32)
        du_m = np.zeros((m, n), np.float32)

        dv_l = np.zeros((m, n), np.float32)
        dv_m = np.zeros((m, n), np.float32)

        for i in range(iterations_fx2):
            diff_term = robust_function_derivative(
                norm_of_gradient(u_k + du_l) ** 2 + norm_of_gradient(v_k + dv_l) ** 2,
                0.001,
            )

            data_term = gamma_d * robust_function_derivative(
                (it_s + du_l * im1_s_shift_x + dv_l * im1_s_shift_y) ** 2, 0.001
            )
            gradient_term = gamma_g * robust_function_derivative(
                (ixt_s + du_l * im1_s_shift_xx + dv_l * im1_s_shift_xy) ** 2
                + (iyt_s + du_l * im1_s_shift_xy + dv_l * im1_s_shift_yy) ** 2,
                0.001,
            )

            denominator_u = (
                data_term * (im1_s_shift_x**2)
                + gradient_term * (im1_s_shift_xx**2 + im1_s_shift_xy**2)
                + alpha * sum_convolution_neighbours(diff_term)
            )
            denominator_v = (
                data_term * (im1_s_shift_y**2)
                + gradient_term * (im1_s_shift_yy**2 + im1_s_shift_xy**2)
                + alpha * sum_convolution_neighbours(diff_term)
            )
            alpha_d_u = alpha / denominator_u
            alpha_d_v = alpha / denominator_v
            product_1_u = alpha * sum_convolution_neighbours(diff_term) * u_k
            product_1_v = alpha * sum_convolution_neighbours(diff_term) * v_k

            for j in range(iterations_fx3):
                diff_div_u = diff_term * sum_convolution(u_k + du_m) + sum_convolution(
                    diff_term * (u_k + du_m)
                )
                diff_div_v = diff_term * sum_convolution(v_k + dv_m) + sum_convolution(
                    diff_term * (v_k + dv_m)
                )

                du_m = alpha_d_u * diff_div_u - (
                    data_term * (im1_s_shift_x * (it_s + im1_s_shift_y * dv_m))
                    + gradient_term
                    * (
                        im1_s_shift_xx * (ixt_s + im1_s_shift_xy * dv_m)
                        + im1_s_shift_xy * (iyt_s + im1_s_shift_yy * dv_m)
                    )
                    + product_1_u
                ) / (denominator_u)

                dv_m = alpha_d_v * diff_div_v - (
                    data_term * (im1_s_shift_y * (it_s + im1_s_shift_x * du_m))
                    + gradient_term
                    * (
                        im1_s_shift_xy * (ixt_s + im1_s_shift_xx * du_m)
                        + im1_s_shift_yy * (iyt_s + im1_s_shift_xy * du_m)
                    )
                    + product_1_v
                ) / (denominator_v)
                zae = zae + 1
                if verbose:
                    print(str(zae) + "/" + str(maxiter), end="\r", flush=True)
            du_l = du_m
            dv_l = dv_m

        u_k = u_k + du_l
        v_k = v_k + dv_l

        u_k = cv2.medianBlur(np.float32(u_k), 3)
        v_k = cv2.medianBlur(np.float32(v_k), 3)

        u_k[0, :] = u_k[1, :]
        v_k[0, :] = v_k[1, :]

        u_k[m - 1, :] = u_k[m - 2, :]
        v_k[m - 1, :] = v_k[m - 2, :]

        u_k[:, 0] = u_k[:, 1]
        v_k[:, 0] = v_k[:, 1]

        u_k[:, n - 1] = u_k[:, n - 2]
        v_k[:, n - 1] = v_k[:, n - 2]

    flow[:, :, 0] = u_k
    flow[:, :, 1] = v_k

    return flow
