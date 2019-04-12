import numpy as np
from skimage.feature import hog
from skimage.io import imread
from settings import hog_params, sample_points
import pdb


def get_hog_descriptors(img, points, window, orientations, cells_per_block, max_window_size):
    """Find the hog descriptors for a point around a fixed window."""
    descriptors = []
    for p in points:
        x, y = p
        if x - (max_window_size - 1) < 0 or x + max_window_size > img.shape[0] or y - (max_window_size - 1) < 0 or y + max_window_size > img.shape[1]:
            continue
        im_slice = img[x - (window-1): x + window, y - (window-1): y + window]
        pixels_per_cell = (2 * window - 1) // cells_per_block[0], (2 * window - 1) // cells_per_block[1]
        feature = hog(im_slice, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        descriptors.append(feature)
    return np.array(descriptors)


if __name__ == "__main__":
    img = imread("test_edge.png", as_gray=True)
    edge_pixels = img.nonzero()
    indices = np.arange(len(edge_pixels[0]))

    try:
        samples = np.random.choice(indices, (sample_points), False)
    except ValueError:
        samples = indices

    points = np.array([[edge_pixels[0][i], edge_pixels[1][i]] for i in samples])
    descriptors = get_hog_descriptors(img, points, **hog_params)
    pdb.set_trace()
