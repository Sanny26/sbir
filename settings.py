root_folder = "/home/santhoshini/sbir/"

data_folder = "{}{}".format(root_folder, "benchmark/images/")
test_folder = "{}{}".format(root_folder, "benchmark/sketches/")
edge_folder = "{}{}".format(root_folder, "benchmark/edges/")

hog_file = "{}{}".format(root_folder, "hog_file.pkl")
sc_file = "{}{}".format(root_folder, "sc_file.pkl")
spark_file = "{}{}".format(root_folder, "spark.pkl")

train_file = "data/train.txt"
test_file = "data/test.txt"

sample_points = 500

codebook_size = 750

bof_features = "{}{}".format(root_folder, "bof_features.pkl")
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

spark_params = {
                    "nbins_r": 5,
                    "nbins_theta": 12
               }
