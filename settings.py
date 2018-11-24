
#data_folder = "/home/chris/Downloads/benchmark/images/"
#test_folder = "/home/chris/Downloads/benchmark/sketches/"
#edge_folder = "/home/chris/Downloads/benchmark/edges/"

#hog_file = "/home/chris/Downloads/hog_file.pkl"
#sc_file = "/home/chris/Downloads/sc_file.pkl"
#spark_file = "/home/chris/Downloads/spark.pkl"

data_folder = "/home/santhoshini/sbir/benchmark/images/"
test_folder = "/home/santhoshini/sbir/benchmark/sketches/"
edge_folder = "/home/santhoshini/sbir/benchmark/edges/"

hog_file = "/home/santhoshini/sbir/hog_file.pkl"
sc_file = "/home/santhoshini/sbir/sc_file.pkl"
spark_file = "/home/santhoshini/sbir/spark.pkl"

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

spark_params = {
                    "nbins_r": 5,
                    "nbins_theta": 12,
                    "window_size": 200
               }
