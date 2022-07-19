import numpy
from chainer import cuda


def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.

    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)

    Returns (numpy.ndarray): (batch_size, num_point,)

    """
    return ((x - y) ** 2).sum(axis=2)


def farthest_point_sampling(pts, k, initial_idx=None, metrics=l2_norm,
                            skip_initial=False, indices_dtype=numpy.int32,
                            distances_dtype=numpy.float32):
    """Batch operation of farthest point sampling

    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python

    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        k (int): number of points to sample
        initial_idx (int): initial index to start farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
        xp (numpy or cupy):
        indices_dtype (): dtype of output `indices`
        distances_dtype (): dtype of output `distances`

    Returns (tuple): `indices` and `distances`.
        indices (numpy.ndarray or cupy.ndarray): 2-dim array (batch_size, k, )
            indices of sampled farthest points.
            `pts[indices[i, j]]` represents `i-th` batch element of `j-th`
            farthest point.
        distances (numpy.ndarray or cupy.ndarray): 3-dim array
            (batch_size, k, num_point)

    """
    if pts.ndim == 2:
        # insert batch_size axis
        pts = pts[None, ...]
    assert pts.ndim == 3
    xp = cuda.get_array_module(pts)
    batch_size, num_point, coord_dim = pts.shape
    indices = xp.zeros((batch_size, k, ), dtype=indices_dtype)

    # distances[bs, i, j] is distance between i-th farthest point `pts[bs, i]`
    # and j-th input point `pts[bs, j]`.
    distances = xp.zeros((batch_size, k, num_point), dtype=distances_dtype)
    if initial_idx is None:
        indices[:, 0] = xp.random.randint(len(pts))
    else:
        indices[:, 0] = initial_idx

    batch_indices = xp.arange(batch_size)
    farthest_point = pts[batch_indices, indices[:, 0]]
    # minimum distances to the sampled farthest point
    try:
        min_distances = metrics(farthest_point[:, None, :], pts)
    except Exception as e:
        import IPython; IPython.embed()

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[:, 0] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, 0]]
        min_distances = metrics(farthest_point[:, None, :], pts)

    distances[:, 0, :] = min_distances
    for i in range(1, k):
        indices[:, i] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, i]]
        dist = metrics(farthest_point[:, None, :], pts)
        distances[:, i, :] = dist
        min_distances = xp.minimum(min_distances, dist)
    return indices, distances


if __name__ == '__main__':
    # when num_point = 10000 & k = 1000 & batch_size = 32,
    # CPU takes 6 sec, GPU takes 0.5 sec.

    from contextlib import contextmanager
    from time import time

    @contextmanager
    def timer(name):
        t0 = time()
        yield
        t1 = time()
        print('[{}] done in {:.3f} s'.format(name, t1-t0))

    # batch_size = 32
    # num_point = 10000
    # coord_dim = 2
    # k = 1000
    # do_plot = False
    batch_size = 3
    num_point = 100
    coord_dim = 2
    k = 5
    do_plot = True

    device = 0
    print('num_point', num_point, 'device', device)
    if device == -1:
        pts = numpy.random.uniform(0, 1, (batch_size, num_point, coord_dim))
    else:
        import cupy
        pts = cupy.random.uniform(0, 1, (batch_size, num_point, coord_dim))

    with timer('1st'):
        farthest_indices, distances = farthest_point_sampling(pts, k)

    with timer('2nd'):  # time measuring twice.
        farthest_indices, distances = farthest_point_sampling(pts, k)

    with timer('3rd'):  # time measuring twice.
        farthest_indices, distances = farthest_point_sampling(
            pts, k, skip_initial=True)

    # with timer('gpu'):
    #     farthest_indices = farthest_point_sampling_gpu(pts, k)
    print('farthest_indices', farthest_indices.shape, type(farthest_indices))

    if do_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os
        pts = cuda.to_cpu(pts)
        farthest_indices = cuda.to_cpu(farthest_indices)
        if not os.path.exists('results'):
            os.mkdir('results')
        for index in range(batch_size):
            fig, ax = plt.subplots()
            plt.grid(False)
            plt.scatter(pts[index, :, 0], pts[index, :, 1], c='k', s=4)
            plt.scatter(pts[index, farthest_indices[index], 0], pts[index, farthest_indices[index], 1], c='r', s=4)
            # plt.show()
            plt.savefig('results/farthest_point_sampling_{}.png'.format(index))

        # --- To extract farthest_points, you can use this kind of advanced indexing ---
        farthest_points = pts[numpy.arange(batch_size)[:, None],
                          farthest_indices, :]
        print('farthest_points', farthest_points.shape)
        for index in range(batch_size):
            farthest_pts_index = pts[index, farthest_indices[index], :]
            print('farthest', farthest_points[index].shape,
                  farthest_pts_index.shape,
                  numpy.sum(farthest_points[index] - farthest_pts_index))
            assert numpy.allclose(farthest_points[index], farthest_pts_index)
