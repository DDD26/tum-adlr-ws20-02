import numpy as np

from wzk import safe_unify


def limits2voxel_size(shape, limits):
    voxel_size = np.diff(limits, axis=-1)[:, 0] / np.array(shape)
    return safe_unify(x=voxel_size)


def __mode2offset(voxel_size, mode='c'):
    """Modes
        'c': center
        'b': boundary

    """
    if mode == 'c':
        return voxel_size / 2
    elif mode == 'b':
        return 0
    else:
        raise NotImplementedError(f"Unknown offset mode{mode}")


def grid_x2i(*, x, voxel_size, lower_left):
    """
    Get the indices of the grid cell at the coordinates 'x' in a grid with symmetric cells.
    Always use mode='boundary'
    """

    if x is None:
        return None

    return np.asarray((x - lower_left) / voxel_size, dtype=int)


def grid_i2x(*, i, voxel_size, lower_left, mode='c'):
    """
    Get the coordinates of the grid at the index 'o' in a grid with symmetric cells.
    borders: 0 | 2 | 4 | 6 | 8 | 10
    centers: | 1 | 3 | 5 | 7 | 9 |
    """

    if i is None:
        return None

    offset = __mode2offset(voxel_size=voxel_size, mode=mode)
    return np.asarray(lower_left + offset + i * voxel_size, dtype=float)


# Unused
def get_grid_boundaries(*, limits, n_voxels):
    """
    Coordinates of grid borders defined by shape of world and number of grid cells in one direction.
    len() = n_voxels + 1
    limits = [min, max]
    """

    return np.linspace(limits[0], limits[1], n_voxels + 1)


# Unused
def get_grid_centers(*, limits, n_voxels):
    """
    Coordinates of grid centers defined by shape of world and number of grid cells in one direction.
    len() = n_voxels
    limits = [min, max]
    """
    voxel_size = (limits[1] - limits[0]) / n_voxels
    return np.linspace(limits[0] + voxel_size / 2, limits[1] - voxel_size / 2, n_voxels)


# Unused
def get_x_inside_cell(x, voxel_size, lower_left):
    return (x - lower_left) % voxel_size


# Unused
def grid_boundaries2world_n_voxels(grid_b):
    """
    Helper to quickly get the 'world_size' and 'n_voxels' from the coordinates of the grid.
    """

    world_size = grid_b[-1] - grid_b[0]
    n_voxels = len(grid_b) - 1
    return world_size, n_voxels


# Unused
def grid_centers2world_n_voxels(grid_c):
    """
    Helper to quickly get the 'world_size' and 'n_voxels' from the centers of the grid.
    """

    world_size = grid_c[-1] - grid_c[0]
    world_size += ((grid_c[-1] - grid_c[-2]) + grid_c[1] - (grid_c[0])) / 2
    n_voxels = len(grid_c)
    return world_size, n_voxels


# Unused
def grid2voxel_size(grid):
    return grid[1] - grid[0]
