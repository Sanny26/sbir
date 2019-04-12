import numpy as np
import pdb
import math

from settings import sample_points, spark_params

from skimage.io import imread
from skimage.color import gray2rgb
from scipy.spatial.distance import cdist


def plot_feature_points(patch, points, patch_centre):
    patch2 = gray2rgb(patch)
    patch2[points[:, 0], points[:, 1], :] = (0, 255, 0)
    patch2[patch_centre[0], patch_centre[1], :] = (255, 0, 0)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(patch, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(patch2)
    plt.show()


def gen_spark_descriptors(img, points, nbins_r, nbins_theta, window_size):
    if len(np.unique(img)) == 1:
        return []
    bins_r = np.linspace(0, window_size*np.sqrt(2)/2, nbins_r)
    bins_theta = np.linspace(-np.pi, np.pi, nbins_theta)
    descriptor = []

    for i, point in enumerate(points):
        pt_hist = np.zeros((len(bins_r), len(bins_theta)))

        xlow = max(0, point[0]-window_size//2)
        xhigh = min(img.shape[0], point[0]+window_size//2)
        ylow = max(0, point[1]-window_size//2)
        yhigh = min(img.shape[1], point[1]+window_size//2)
        img[point[0], point[1]] = 2
        patch = img[xlow: xhigh, ylow:yhigh]
        patch_centre = (patch == 2).nonzero()
        patch_centre = (patch_centre[0][0], patch_centre[1][0])
        img[point[0], point[1]] = 0
        if np.count_nonzero(patch == 255) == 0:
            continue
        sketch_points = (patch == 255).nonzero()
        sketch_points = np.array(sketch_points).transpose()

        # bin directions to get better results.
        if sketch_points.shape[0] > 1:
            theta_array = cdist(np.array([patch_centre]), sketch_points, lambda u, v: math.atan2((v[1] - u[1]), (v[0] - u[0])))
            direction_bins = np.arange(theta_array.min(), theta_array.max(), 0.05)
            direction_array = []

            if theta_array.min() == theta_array.max():
                direction_array = theta_array
            else:
                for each in theta_array[0]:
                    dbin = (each >= direction_bins).nonzero()[0][-1]
                    direction_array.append(direction_bins[dbin])

            # get first feature line points in every direction.
            direction_array = np.array(direction_array)
            unique_directions = np.unique(direction_array)

            # t_points = []
            for direction in unique_directions:
                pos = (direction_array == direction).nonzero()[0]
                dpoints = sketch_points[pos]
                darray = cdist(np.array([patch_centre]), dpoints)
                # t_points.append(sketch_points[pos[darray.argmin()]])
                bin_r = (bins_r <= darray[0, darray.argmin()]).nonzero()[0][-1]
                bin_theta = (bins_theta <= direction).nonzero()[0][-1]
                pt_hist[bin_r, bin_theta] += 1
            # t_points = np.array(points)
            # plot_feature_points(patch, points, patch_centre)
            descriptor.append(pt_hist.flatten())

        else:
            direction = cdist(np.array([patch_centre]), sketch_points,
                              lambda u, v: math.atan2((v[1] - u[1]), (v[0] - u[0])))[0][0]
            distance = cdist(np.array([patch_centre]), sketch_points)[0][0]
            bin_r = (bins_r <= distance).nonzero()[0][-1]
            bin_theta = (bins_theta <= direction).nonzero()[0][-1]
            pt_hist[bin_r, bin_theta] += 1
            descriptor.append(pt_hist.flatten())

    return np.array(descriptor)


if __name__ == "__main__":
    img = imread("data/test_edge.png", as_gray=True)

    points = (255-img).nonzero()
    rand_ind = np.random.randint(0, len(points[0]), size=(sample_points))
    rpoints = [points[0][rand_ind], points[1][rand_ind]]
    rpoints = np.array(rpoints).transpose()

    desc = gen_spark_descriptors(img, rpoints, **spark_params)
    pdb.set_trace()
