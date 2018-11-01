
data_folder = "/home/chris/Downloads/benchmark/images/"
test_folder = "/home/chris/Downloads/benchmark/sketches/"
edge_folder = "/home/chris/Downloads/benchmark/edge/"

hog_file = "/home/chris/Downloads/hog_file.pkl"
sc_file = "/home/chris/Downloads/sc_file.pkl"


train_file = "data/train.txt"
test_file = "data/test.txt"

sample_points = 500

codebook_size = 750

bof_features = "/home/chris/Downloads/bof_features.pkl"
bof_model = "bagoffeature_" + str(codebook_size) + ".pkl"

edge_params = {
                "sigma": 5,
                "low_threshold": 0.05,
                "high_threshold": 0.2
              }

hog_params = {
                "window_sizes": [5, 10, 15, 20, 25, 30],
                "orientations": 8,
                "cells_per_block": (4, 4)
             }


sc_params = {
                "window_sizes": [5, 10, 15, 20, 25, 30],
                "nbins_r": 5,
                "nbins_theta": 12,
                "r_inner": 0.1250,
                "r_outer": 2.0
            }
