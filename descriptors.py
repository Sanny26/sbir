"""Generate all descriptors."""

from data import read_data
from hog import get_hog_descriptors
import numpy as np
import pdb
import pickle
from settings import train_file, sample_points
from settings import hog_params, hog_file
from settings import sc_params, sc_file
from shape_context import shape_context


def gen_hog_descriptors(img, points, window_sizes, orientations, cells_per_block):
    """Generate Hog descriptors using different window sizes."""
    max_window_size = max(window_sizes)

    descriptors = get_hog_descriptors(img, points, window_sizes[0], orientations, cells_per_block, max_window_size)

    for window in window_sizes[1::]:
        desc = get_hog_descriptors(img, points, window, orientations, cells_per_block, max_window_size)
        descriptors = np.hstack((descriptors, desc))

    return descriptors


def create_and_save_hog_words(data_file, hog_params, hog_file):
    """Create hog descriptors and save them to a pickle file."""
    print("Reading images.")
    X, y, filenames = read_data(data_file, test=True)

    hog_words = {}

    for i, img in enumerate(X):
        print("Processing image ", i)
        edge_pixels = img.nonzero()
        indices = np.arange(len(edge_pixels[0]))

        try:
            samples = np.random.choice(indices, (sample_points), False)
        except ValueError:
            samples = indices

        points = np.array([[edge_pixels[0][i], edge_pixels[1][i]] for i in samples])
        descriptors = gen_hog_descriptors(img, points, **hog_params)
        hog_words[filenames[i]] = descriptors
    pdb.set_trace()
    pickle.dump(hog_words, open(hog_file, "wb"))


def create_and_save_sc_words(data_file, sc_params, sc_file):
    """Create hog descriptors and save them to a pickle file."""
    print("Reading images.")
    X, y, filenames = read_data(data_file, test=True)

    sc_words = {}

    for i, img in enumerate(X):
        print("Processing image ", i)
        edge_pixels = img.nonzero()
        indices = np.arange(len(edge_pixels[0]))

        try:
            samples = np.random.choice(indices, (sample_points), False)
        except ValueError:
            samples = indices

        points = np.array([[edge_pixels[0][i], edge_pixels[1][i]] for i in samples])
        descriptors = gen_sc_descriptors(img, points, **sc_params)
        sc_words[filenames[i]] = descriptors
    pdb.set_trace()
    pickle.dump(sc_words, open(sc_file, "wb"))


def gen_sc_descriptors(points, window_sizes, nbins_r, nbins_theta, r_inner, r_outer):
    """Generate Hog descriptors using different window sizes."""
    max_window_size = max(window_sizes)

    descriptors = shape_context(points, window=window_sizes[0], max_window_size=max_window_size, nbins_r=nbins_r,
                                nbins_theta=nbins_theta, r_inner=r_inner, r_outer=r_outer)

    for window in window_sizes[1::]:
        desc = shape_context(points, window=window, max_window_size=max_window_size, nbins_r=nbins_r,
                             nbins_theta=nbins_theta, r_inner=r_inner, r_outer=r_outer)
        descriptors = np.hstack((descriptors, desc))

    return descriptors


if __name__ == "__main__":
    create_and_save_hog_words(train_file, hog_params, hog_file)
    create_and_save_sc_words(train_file, sc_params, sc_file)
    pdb.set_trace()