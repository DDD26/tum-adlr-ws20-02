import numpy as np
import scipy.ndimage as ndimage  # Image interpolation

# For testing of sobel filter
from wzk import numeric_derivative

# Finding Use voxel_size + lower_left as inputs instead of using limits and inferring those values
#  The advantage of voxel_size + lower_left as argument is that voxel_size is a scalar and lower_left can be omitted in
#  most of the cases so the standard signature is much more concise,
#  the trade off is that some functions have now two parameters instead of just one
#  Additionally limits has always 2*n_dim dimensions


# Obstacle Image to Distance Image
def obstacle_img2dist_img(*, img, voxel_size, add_boundary=True):
    """
    Calculate the signed distance field from an 2D/3D image of the world.
    Obstacles are 1/True, free space is 0/False.
    The distance image is of the same shape as the input image and has positive values outside objects and negative
    values inside objects see 'CHOMP - signed distance field' (10.1177/0278364913488805)
    The voxel_size is used to scale the distance field correctly (the shape of a single pixel / voxel)
    """

    n_voxels = np.array(img.shape)

    if not add_boundary:
        # Main function
        #                                         # EDT wants objects as 0, rest as 1
        dist_img = ndimage.distance_transform_edt(-img.astype(int) + 1, sampling=voxel_size)
        dist_img_complement = ndimage.distance_transform_edt(img.astype(int), sampling=voxel_size)
        dist_img[img] = - dist_img_complement[img]  # Add interior information

    else:
        # Additional branch, to include boundary filled with obstacles
        obstacle_img_wb = np.ones(n_voxels + 2, dtype=bool)
        inner_image_idx = tuple(map(slice, np.ones(img.ndim, dtype=int), (n_voxels + 1)))
        obstacle_img_wb[inner_image_idx] = img

        dist_img = obstacle_img2dist_img(img=obstacle_img_wb, voxel_size=voxel_size, add_boundary=False)
        dist_img = dist_img[inner_image_idx]

    return dist_img


def obstacle_img2dist_img_derv(*,
                               dist_img=None,                         # either
                               obstacle_img=None, add_boundary=True,  # or
                               voxel_size):
    """
    Use Sobel-filter to get the derivative of the edt
    """

    if dist_img is None:
        dist_img = obstacle_img2dist_img(img=obstacle_img, voxel_size=voxel_size, add_boundary=add_boundary)
    return img2sobel(img=dist_img, voxel_size=voxel_size)


# Obstacle Image to Distance Function (including the radii of the spheres)
def obstacle_img2dist_fun(*,
                          obstacle_img=None, add_boundary=True,  # A
                          dist_img=None,                         # B
                          voxel_size, lower_left, interp_order=1, radius_spheres=None):
    if dist_img is None:
        dist_img = obstacle_img2dist_img(img=obstacle_img, voxel_size=voxel_size, add_boundary=add_boundary)
    dist_fun = img2interpolation_fun(img=dist_img, order=interp_order, voxel_size=voxel_size, lower_left=lower_left)

    if radius_spheres is None:
        return dist_fun
    else:
        def dist_fun_r(x):
            r_spheres_temp = __create_radius_temp(radius=radius_spheres, shape=x.shape)
            return dist_fun(x=x) - r_spheres_temp

        return dist_fun_r


def obstacle_img2dist_derv(*,
                           dist_img_derv=None,                    # A
                           obstacle_img=None, add_boundary=True,  # B 1
                           dist_img=None,                         # B 2
                           interp_order=1, voxel_size, lower_left):
    if dist_img_derv is None:
        dist_img_derv = obstacle_img2dist_img_derv(obstacle_img=obstacle_img, add_boundary=add_boundary,
                                                   dist_img=dist_img, voxel_size=voxel_size)
    dist_derv = img_derv2interpolation_fun(img_derv=dist_img_derv, order=interp_order,
                                           voxel_size=voxel_size, lower_left=lower_left)
    return dist_derv


# Distance function to Cost Function (CHOMP)
def dist_fun2cost_fun(*, dist_fun_r, eps):
    def __cost_fun(x):
        return cost_fun(distance=dist_fun_r(x=x), eps=eps)

    return __cost_fun


def dist_fun_derv2cost_fun_derv(*, dist_fun_r, dist_derv, eps):
    def __cost_fun_derv(x):
        # Distance
        dist = dist_fun_r(x=x)

        # Cost, cost_derv
        cost, _cost_derv = cost_derv(distance=dist, eps=eps)
        _cost_derv = dist_derv(x=x) * _cost_derv[..., np.newaxis]

        return cost, _cost_derv

    return __cost_fun_derv


# Main functions
def obstacle_img2cost_fun(*, eps,
                          r_spheres=None, voxel_size,
                          lower_left, interp_order=0,  # For A+B
                          obstacle_img=None, add_boundary=True,  # A
                          dist_img=None,  # or B
                          dist_fun_r=None):  # or C
    """
    Interpolate the distance field (the obstacle cost function to be more precise) at each point (continuous).
    The optimizer is happier with the interpolated version, but it is hard to ensure that the interpolation is
    conservative, so the direct variant should be preferred. (actually)
    """

    # DO NOT INTERPOLATE the EDT -> not conservative -> use order=0 / 'nearest'
    if dist_fun_r is None:
        dist_fun_r = obstacle_img2dist_fun(voxel_size=voxel_size, interp_order=interp_order, radius_spheres=r_spheres,
                                           obstacle_img=obstacle_img, add_boundary=add_boundary,
                                           dist_img=dist_img, lower_left=lower_left)
    cost_fun_ = dist_fun2cost_fun(dist_fun_r=dist_fun_r, eps=eps)

    return cost_fun_


def obstacle_img2cost_fun_derv(*, eps, interp_order=1,
                               obstacle_img=None, add_boundary=True,  # Either
                               dist_img=None, dist_img_derv=None,     # Or
                               dist_fun_r=None, dist_derv=None,       # Or
                               voxel_size, lower_left, r_spheres):
    if dist_fun_r is None:
        dist_fun_r = obstacle_img2dist_fun(voxel_size=voxel_size, interp_order=interp_order, radius_spheres=r_spheres,
                                           obstacle_img=obstacle_img, add_boundary=add_boundary,
                                           dist_img=dist_img, lower_left=lower_left)

    if dist_derv is None:
        dist_derv = obstacle_img2dist_derv(obstacle_img=obstacle_img,  add_boundary=add_boundary,
                                           dist_img_derv=dist_img_derv, interp_order=interp_order,
                                           voxel_size=voxel_size, lower_left=lower_left)

    cost_fun_derv_ = dist_fun_derv2cost_fun_derv(dist_fun_r=dist_fun_r, dist_derv=dist_derv, eps=eps)
    return cost_fun_derv_


def obstacle_img2functions(*, obstacle_img=None, add_boundary=True,
                           dist_img=None,
                           dist_img_derv=None,
                           voxel_size, lower_left, spheres_radius,
                           interp_order_dist, interp_order_grad,
                           eps):
    if dist_img is None:
        dist_img = obstacle_img2dist_img(img=obstacle_img, voxel_size=voxel_size, add_boundary=add_boundary)

    if dist_img_derv is None:
        dist_img_derv = obstacle_img2dist_img_derv(dist_img=dist_img, voxel_size=voxel_size)

    dist_fun_r = obstacle_img2dist_fun(dist_img=dist_img, radius_spheres=spheres_radius, interp_order=interp_order_dist,
                                       voxel_size=voxel_size, lower_left=lower_left)

    if interp_order_dist != interp_order_grad:
        dist_fun_r2 = obstacle_img2dist_fun(dist_img=dist_img, radius_spheres=spheres_radius,
                                            interp_order=interp_order_grad,
                                            voxel_size=voxel_size, lower_left=lower_left)
    else:
        dist_fun_r2 = dist_fun_r

    dist_derv = obstacle_img2dist_derv(dist_img_derv=dist_img_derv, interp_order=interp_order_grad,
                                       voxel_size=voxel_size, lower_left=lower_left)

    cost_fun_ = dist_fun2cost_fun(dist_fun_r=dist_fun_r, eps=eps)
    cost_fun_derv_ = dist_fun_derv2cost_fun_derv(dist_fun_r=dist_fun_r2, dist_derv=dist_derv, eps=eps)

    return dist_fun_r, dist_derv, cost_fun_, cost_fun_derv_


def cost_fun(distance, eps, return_indices=False):
    cost = distance.copy()

    # 1. Greater than epsilon
    idx_eps_ = np.nonzero(cost > eps)
    cost[idx_eps_] = 0
    # 2. Between zero and epsilon
    idx_0_eps = np.nonzero(cost > 0)
    cost[idx_0_eps] = 1 / (2 * eps) * (cost[idx_0_eps] - eps) ** 2
    # 3. Less than zero
    idx__0 = np.nonzero(cost < 0)
    cost[idx__0] = - cost[idx__0] + 1 / 2 * eps

    if return_indices:
        return cost, idx_eps_, idx_0_eps, idx__0
    else:
        return cost


def cost_derv(distance, eps):
    # A. Cost Function
    cost, idx_eps_, idx_0_eps, idx__0 = cost_fun(distance=distance, eps=eps, return_indices=True)

    # B. Cost Derivative
    _cost_derv = cost.copy()
    # cost_derv[idx_eps_] = 0                                    # 1.
    _cost_derv[idx_0_eps] = 1 / eps * (distance[idx_0_eps] - eps)  # 2.
    _cost_derv[idx__0] = -1  # 3.

    return cost, _cost_derv


# Helper
def __create_radius_temp(radius, shape):
    if np.size(radius) == 1:
        return radius
    d_spheres = np.nonzero(np.array(shape) == np.size(radius))[0][0]
    r_temp_shape = np.ones(len(shape) - 1, dtype=int)
    r_temp_shape[d_spheres] = np.size(radius)
    return radius.reshape(r_temp_shape)


def img2sobel(img, voxel_size):
    """
    Calculate the derivative of an image in each direction of the image, using the sobel filter.
    """

    sobel = np.zeros((img.ndim,) + img.shape)
    for d in range(img.ndim):  # Treat image boundary like obstacle
        sobel[d, ...] = ndimage.sobel(img, axis=d, mode='constant', cval=0)

    # Check appropriate scaling of sobel filter, should be correct
    sobel /= (8 * voxel_size)  # The shape of the voxels is already accounted for in the distance image
    return sobel


# Image interpolation
def img2interpolation_fun(*, img, order=1, mode='nearest',
                          voxel_size, lower_left):
    """
    Return a function which interpolates between the pixel values of the image (regular spaced grid) by using
    'scipy.ndimage.map_coordinates'. The resulting function takes as input argument either a np.array or a list of
    world coordinates (!= image coordinates)
    The 'order' keyword indicates which order of interpolation to use. Standard is linear interpolation (order=1).
    For order=0 no interpolation is performed and the value of the nearest grid cell is chosen. Here the values between
    the different cells jump and aren't continuous.

    """

    factor = 1 / voxel_size

    def interp_fun(x):
        x2 = x.copy()
        if x2.ndim == 1:
            x2 = x2[np.newaxis, :]

        # Map physical coordinates to image indices
        if np.any(lower_left != 0):
            x2 -= lower_left

        x2 *= factor
        x2 -= 0.5

        return ndimage.map_coordinates(input=img, coordinates=x2.T, order=order, mode=mode).T

    return interp_fun


def img_derv2interpolation_fun(*, img_derv, voxel_size, lower_left, order=1, mode='nearest'):
    """
    Interpolate images representing derivatives (ie from soble filter). For each dimension there is a derivative /
    layer in the image.
    Return the results combined as an (x, n_dim) array for the derivatives at each point for each dimension.
    """

    n_dim = img_derv.shape[0]

    fun_list = []
    for d in range(n_dim):
        fun_list.append(img2interpolation_fun(img=img_derv[d, ...], order=order, mode=mode, voxel_size=voxel_size,
                                              lower_left=lower_left))

    def fun_derv(x):
        res = np.empty_like(x)
        for _d in range(n_dim):
            res[..., _d] = fun_list[_d](x=x)

        return res

    return fun_derv


# Numeric derivatives
def cost_fun_derv_num(obstacle_cost_fun, eps=1e-5):
    def _cost_fun_derv(x):
        return (obstacle_cost_fun(x),
                numeric_derivative(fun=obstacle_cost_fun, x=x, eps=eps, axis=-1))

    return _cost_fun_derv


def dist_fun_derv_num(dist_fun, eps=1e-5):
    def _dist_fun_derv(x):
        return dist_fun(x), numeric_derivative(fun=dist_fun, x=x, eps=eps, axis=-1)

    return _dist_fun_derv


def dist_derv_num(dist_fun, eps=1e-5):
    def _dist_derv(x):
        return numeric_derivative(fun=dist_fun, x=x, eps=eps, axis=-1)

    return _dist_derv
