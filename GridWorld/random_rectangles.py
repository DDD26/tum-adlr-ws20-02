import numpy as np

from wzk import get_first_non_empty, fill_with_air_left, safe_add_small2big, initialize_image_array


def create_rectangles(*, n, size_limits=(1, 10), n_voxels=(64, 64),
                      special_dim=None):
    """
    Create a series of randomly placed and randomly sized rectangles.
    The bottom left corner and the shape in each dimension are returned.
    All values are pixel based. so there are no obstacles only occupying half a cell.
    The flag wall_dims indicates, if the obstacles should be walls, rather than general cubes. For walls one dimension
    is significant thinner than the others. The possible dimensions are defined in the tuple 'wall_dims'.
    """

    # Convert obstacle dimensions to number of occupied grid cells
    n_dim = np.size(n_voxels)

    if np.isscalar(n):
        n = int(n)
        rect_pos = np.zeros((n, n_dim), dtype=int)
        for d in range(n_dim):
            rect_pos[:, d] = np.random.randint(low=0, high=n_voxels[d] - size_limits[0] + 1, size=n)
        rect_size = np.random.randint(low=size_limits[0], high=size_limits[1] + 1, size=(n, n_dim))

        if special_dim is not None and special_dim is not (None, None):
            dimensions, size = special_dim
            if isinstance(dimensions, int):
                dimensions = (dimensions,)

            rect_size[list(range(n)), np.random.choice(dimensions, size=n)] = size

        # Crop rectangle shape, if the created obstacle goes over the boundaries of the world
        diff = rect_pos + rect_size
        ex = np.where(diff > n_voxels, True, False)
        rect_size[ex] -= (diff - n_voxels)[ex]

        return rect_pos, rect_size

    else:

        rect_pos = np.empty(len(n), dtype=object)
        rect_size = np.empty(len(n), dtype=object)

        for i, nn in enumerate(n):
            rect_pos[i], rect_size[i] = create_rectangles(n_voxels=n_voxels, n=nn,
                                                          size_limits=size_limits,
                                                          special_dim=special_dim)
        return rect_pos, rect_size


def rectangles2image(*, rect_pos, rect_size, n_voxels=(64, 64), n_samples=None):
    """
    Create an image / a world out of rectangles with given position and shapes.
    The image is black/white (~True/False -> bool): False for free, True for obstacle.
    """

    if n_samples is None:
        n_obstacles, n_dim = rect_pos.shape
        obstacle_img = initialize_image_array(n_voxels=n_voxels, n_dim=n_dim)

        for i in range(n_obstacles):
            ll_i = np.array(rect_pos[i])
            ur_i = ll_i + np.array(rect_size[i])
            obstacle_img[tuple(map(slice, ll_i, ur_i))] = True

    else:
        n_dim = rect_pos[0].shape[0]
        obstacle_img = initialize_image_array(n_voxels=n_voxels, n_dim=n_dim, n_samples=n_samples)
        for i in range(n_samples):
            obstacle_img[i] = rectangles2image(n_voxels=n_voxels, rect_pos=rect_pos[i], rect_size=rect_size[i],
                                               n_samples=None)

    return obstacle_img


def create_rectangle_image(*, n, size_limits=(1, 10), n_voxels=(64, 64),
                           special_dim=None, return_rectangles=False):

    if np.isscalar(n):
        n_samples = None
    else:
        n_samples = len(n)

    rect_pos, rect_size = create_rectangles(n=n, size_limits=size_limits, n_voxels=n_voxels, special_dim=special_dim)
    img = rectangles2image(rect_pos=rect_pos, rect_size=rect_size, n_voxels=n_voxels, n_samples=n_samples)

    if return_rectangles:
        return img, (rect_pos, rect_size)
    else:
        return img


def clear_space(*, img, safety_idx=None, safety_margin=21, n_dim, n_voxels):

    safety_area = initialize_image_array(n_voxels=safety_margin, n_dim=n_dim, initialization='zeros')
    safety_img = initialize_image_array(n_voxels=n_voxels, n_dim=n_dim, initialization='ones')
    safety_img = safe_add_small2big(idx=safety_idx, small_img=safety_area, big_img=safety_img)

    img = np.logical_and(img, safety_img)
    return img


def rect_lists2arr(rect_pos, rect_size, n_rectangles_max):

    n = len(rect_pos)
    n_dim = np.shape(get_first_non_empty(rect_pos))[1]

    rect_pos_all = np.zeros((n, n_rectangles_max, n_dim))
    rect_size_all = np.zeros((n, n_rectangles_max, n_dim))

    for i in range(n):
        fill_with_air_left(arr=rect_pos[i], out=rect_pos_all[i])
        fill_with_air_left(arr=rect_size[i], out=rect_size_all[i])

    return rect_pos_all, rect_size_all


# Unused
def ll_ur2size(ll, ur):
    return ur - ll


# Unused
def ll_size2ur(ll, size):
    return ll + size


# Unused
def center_size2ll_ur(center, size):
    assert size % 2 == 1
    size2 = (size - 1) // 2
    ll = center - size2
    ur = center + size2 + 1
    return ll, ur
