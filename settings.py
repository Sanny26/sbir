# Chris files.
data_folder = "/home/chris/Downloads/benchmark/images/"
test_folder = "/home/chris/Downloads/benchmark/sketches/"
edge_folder = "/home/chris/Downloads/benchmark/edges/"
docs_folder = "/home/chris/Downloads/benchmark/docs/"

hog_file = "/home/chris/Downloads/hog_file.pkl"
sc_file = "/home/chris/Downloads/sc_file.pkl"
spark_file = "/home/chris/Downloads/spark.pkl"


train_file = "/home/chris/sbir/data/train.txt"
test_file = "/home/chris/sbir/data/test.txt"

sample_points = 500

codebook_size = 750

hog_model = "/home/chris/sbir/hog_model" + str(codebook_size) + ".pkl"
sc_model = "/home/chris/sbir/sc_model" + str(codebook_size) + ".pkl"
spark_model = "/home/chris/sbir/spark_model" + str(codebook_size) + ".pkl"

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

tfidf_model = "/home/chris/sbir/tfidf.pkl"
