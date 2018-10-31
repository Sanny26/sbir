import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import pdb
import math

from scipy.spatial.distance import cdist


def sample_pts_image(image, threshold=100, npoints=100, radius=2, aperture=5):
    dimg = image.copy()
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(image, threshold, threshold * 3, aperture)
    py, px = np.gradient(image)
    points = [index for index, val in np.ndenumerate(dst)
              if val == 255 and index[0] < py.shape[0] and index[1] < py.shape[1]]
    h, w = image.shape
    while len(points) > npoints:
        newpoints = points
        xr = range(0, w, radius)
        yr = range(0, h, radius)
        for p in points:
            if p[0] not in yr and p[1] not in xr:
                newpoints.remove(p)
                if len(points) <= npoints:
                    T = np.zeros((npoints, 1))
                    for i, (y, x) in enumerate(points):
                        radians = math.atan2(py[y, x], px[y, x])
                        T[i] = radians + 2 * math.pi * (radians < 0)
                    return points, np.asmatrix(T)
        radius += 1
    T = np.zeros((npoints, 1))
    for i, (y, x) in enumerate(points):
        radians = math.atan2(py[y, x], px[y, x])
        T[i] = radians + 2 * math.pi * (radians < 0)
    return points, np.asmatrix(T)


def compute_SCD(points, window_size=None):
    '''
    Computes the Shape Context Feature Descriptor for the given input points.
    Params:
        points: List of (x, y) tuple. Sampled points from the image
        window_size: int. If nothing is passed, the descripor is calculated at global level, otherwise descriptor is 
                    calculated w.r.t local neighbourhood given by the window_size.
    '''
    nbins_r=5
    nbins_theta=12
    r_inner=0.1250
    r_outer=2.0
    nbins = nbins_theta * nbins_r
    descriptor = np.zeros((t_points, nbins))
    
    t_points = len(points)
    r_array = cdist(points, points)
    am = r_array.argmax()
    max_points = [am // t_points, am % t_points]
    r_array_n = r_array // r_array.mean()
    r_bin_edges = np.logspace(np.log10(r_inner), np.log10(r_outer), nbins_r)
    r_array_q = np.zeros((t_points, t_points), dtype=int)
    
    for m in range(nbins_r):
        r_array_q += (r_array_n < r_bin_edges[m])

    fz = r_array_q > 0

    theta_array = cdist(points, points, lambda u, v: math.atan2((v[1] - u[1]), (v[0] - u[0])))
    norm_angle = theta_array[max_points[0], max_points[1]]
    theta_array = (theta_array - norm_angle * (np.ones((t_points, t_points)) - np.identity(t_points)))
    theta_array[np.abs(theta_array) < 1e-7] = 0

    theta_array_2 = theta_array + 2 * math.pi * (theta_array < 0)
    theta_array_q = (1 + np.floor(theta_array_2 // (2 * math.pi / nbins_theta))).astype(int)

    points = np.array(points)
    for i in range(t_points):
        sn = np.zeros((nbins_r, nbins_theta))
        if window_size is not None:
            x, y = points[i]
            local_x = np.where((points[:, 0] >= (x-window_size)) & (points[:, 0] <= (x+window_size)))
            local_y = np.where((points[:, 0] >= y-window_size) & (points[:, 0] <= y+window_size))
            local_points = np.intersect1d(local_x, local_y)
            for j in local_points:
                if (fz[i, j]):
                    sn[r_array_q[i, j] - 1, theta_array_q[i, j] - 1] += 1
        else:
            for j in range(t_points):
                if (fz[i, j]):
                    sn[r_array_q[i, j] - 1, theta_array_q[i, j] - 1] += 1
        descriptor[i] = sn.reshape(nbins)

    return descriptor


if __name__ == "__main__":
    img = cv2.imread('/home/sanny/sbir/benchmark/images/0/32707061.jpg')
    points, _ = sample_pts_image(img, npoints=500)
    
    feat = np.zeros((500, 60, 6)) 
    for i, size in enumerate([5, 10, 15, 20, 25, 30]):
        feat[:, :, i] = compute_SCD(points, size)
    pdb.set_trace()