# NOTE: Don't keep a root folder, data may not be in the root folder.
#       Ideally variables in settings.py should be constants that are user defined
#       Not a good idea to generate them assuming some file structure.

data_folder = "/home/chris/Downloads/benchmark/images/"
test_folder = "/home/chris/Downloads/benchmark/sketches/"
edge_folder = "/home/chris/Downloads/benchmark/edge/"
doc_folder = "/home/chris/Downloads/benchmark/docs/"

hog_file = "/home/chris/Downloads/hog_file.pkl"
sc_file = "/home/chris/Downloads/sc_file.pkl"
spark_file = "/home/chris/Downloads/spark.pkl"

# data_folder = "/home/santhoshini/sbir/benchmark/images/"
# test_folder = "/home/santhoshini/sbir/benchmark/sketches/"
# edge_folder = "/home/santhoshini/sbir/benchmark/edges/"
#
# hog_file = "/home/santhoshini/sbir/hog_file.pkl"
# sc_file = "/home/santhoshini/sbir/sc_file.pkl"
# spark_file = "/home/santhoshini/sbir/spark.pkl"

train_file = "data/train.txt"
test_file = "data/test.txt"

sample_points = 500

codebook_size = 750

hog_model = "hog_model" + str(codebook_size) + ".pkl"
sc_model = "sc_model" + str(codebook_size) + ".pkl"
spark_model = "spark_model" + str(codebook_size) + ".pkl"

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
