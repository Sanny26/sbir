import os
import numpy as np
from settings import test_folder, edge_folder
from settings import train_file, test_file
from skimage.io import imread


def make_train_val_file(folder, train_file):
    """Make train and validation file."""
    folders = os.listdir(folder)
    t_file = open(train_file, "w")

    for label in folders:
        path = os.path.join(folder, label)
        files = os.listdir(path)

        for f in files:
            t_file.write("{},{}\n".format(os.path.join(folder, label, f), int(label)))

    t_file.close()


def make_test_file(folder, output_file):
    """Make test file."""
    files = os.listdir(folder)
    t_file = open(output_file, "w")

    for f in files:
        label = f.strip(".png")
        t_file.write("{},{}\n".format(os.path.join(folder, f), int(label)))

    t_file.close()


def read_data(filename, test=False):
    """Read images from train and test files."""
    f = open(filename, "r")
    X = []
    Y = []
    filenames = []

    for l in f:
        data = l.strip().split(',')
        filenames.append(data[0])

        img = imread(data[0], as_gray=True)
        if test:
            img = (1 - img) * 255
            img = img.astype(np.uint8)

        X.append(img)
        Y.append(int(data[1]))

    return np.array(X), np.array(Y), filenames


if __name__ == "__main__":
    make_train_val_file(edge_folder, train_file)
    make_test_file(test_folder, test_file)
