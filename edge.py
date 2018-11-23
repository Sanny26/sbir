"""Edge detection using Canny edge detector for non-sketch images."""

import os
# import pdb

from settings import data_folder, edge_folder
from settings import edge_params

from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.io import imread, imsave


def convert_to_sketch(img, use_params=True):
    """Convert a non-sketch image to sketch space using edge detection."""
    if len(img.shape) == 3:
        img = rgb2gray(img)

    if use_params:
        edges = canny(img, **edge_params)
    else:
        edges = canny(img)
    return edges * 255


def convert_images(data_folder, edge_folder):
    """Generate non-sketch images and save to edge folder."""
    folders = os.listdir(data_folder)
    print("Converting images to edge sketches")
    for label in folders:
        path = os.path.join(data_folder, label)
        edge_path = os.path.join(edge_folder, label)

        if not os.path.isdir(edge_path):
            os.mkdir(edge_path)
        files = os.listdir(path)

        for f in files:
            image_path = os.path.join(path, f)
            print("Processing image: ", image_path)
            img = imread(image_path, as_gray="True")
            edge_img = convert_to_sketch(img, use_params=True)
            # edge_img = convert_to_sketch(img, use_params=False)
            imsave(os.path.join(edge_path, f.strip(".jpg") + ".png"), edge_img)

    return True


if __name__ == "__main__":
    convert_images(data_folder, edge_folder)
