"""Code for training bag of features using words."""


import os
import pickle
import pdb

import numpy as np
from sklearn.cluster import KMeans

from descriptors import (gen_spark_descriptors, gen_hog_descriptors, gen_sc_descriptors)
from settings import (hog_file, sc_file, spark_file)
from settings import (hog_model, sc_model, spark_model)
from settings import codebook_size, docs_folder, sample_points

from settings import (spark_params, hog_params, sc_params)


def train_bof_model(filename, model_file, num_words=100):
    """Train a bag of features model given a feature."""
    print("Reading data")
    data = pickle.load(open(filename, "rb"))
    keys = list(data.keys())
    errors = []
    raw_data = [data[k] for k in keys]
    for i, r in enumerate(raw_data):
        if type(r) is list:
            errors.append(keys[i])
        elif r.shape[1] != 60:
            errors.append(keys[i])
    pdb.set_trace() 
    print("Building vectors")
    raw_data = np.vstack(raw_data)
    print(raw_data.shape)

    print("Training model")
    model = KMeans(n_clusters=num_words, max_iter=500, verbose=1, n_init=5)
    model.fit(raw_data)

    pickle.dump(model, open(model_file, "wb"))


def create_docs(foldername, model_files, feature_files, num_words):
    """Makes a document for the image containing words."""
    word_index = 0
    final_words = {}

    for model_file, filename in zip(model_files, feature_files):
        data = pickle.load(open(filename, "rb"))
        model = pickle.load(open(model_file, "rb"))
        index_add = sum(num_words[0:word_index])
        word_index += 1

        for k in data:
            words = model.predict(data[k])
            words = words + index_add
            if k not in final_words:
                final_words[k] = list(words)
            else:
                final_words[k].extend(list(words))

    for k in final_words:
        i_name = k.split('/')[-1].split(".")[0] + ".txt"
        f = open(os.path.join(foldername, i_name))
        out = "\n".join([str(x) for x in final_words[k]])
        f.write(out)
        f.close()


def get_models(model_files):
    return [pickle.load(open(x, "rb")) for x in model_files]


# NOTE: models and num_words should be in order HoG, SC, Spark
def get_words(img, models, num_words):
    """Given an image find all the words in the image using pre-trained models."""
    # Hog descriptors
    edge_pixels = img.nonzero()
    indices = np.arange(len(edge_pixels[0]))

    try:
        samples = np.random.choice(indices, (sample_points), False)
    except ValueError:
        samples = indices

    points = np.array([[edge_pixels[0][i], edge_pixels[1][i]] for i in samples])
    hog_feature = gen_hog_descriptors(img, points, **hog_params)

    # SC descriptors
    edge_pixels = img.nonzero()
    indices = np.arange(len(edge_pixels[0]))

    try:
        samples = np.random.choice(indices, (sample_points), False)
    except ValueError:
        samples = indices

    points = np.array([[edge_pixels[0][i], edge_pixels[1][i]] for i in samples])
    sc_feature = gen_sc_descriptors(points, **sc_params)

    # Spark descriptors
    points = (255-img).nonzero()
    rand_ind = np.random.randint(0, len(points[0]), size=(sample_points))
    rpoints = [points[0][rand_ind], points[1][rand_ind]]
    rpoints = np.array(rpoints).transpose()
    spark_feature = gen_spark_descriptors(img, rpoints, **spark_params)

    # Find words
    features = [hog_feature, sc_feature, spark_feature]
    word_index = 0
    all_words = []

    for model, feature in zip(models, features):
        index_add = sum(num_words[0:word_index])
        word_index += 1

        words = model.predict(feature) + index_add
        all_words.extend(words)

    return all_words


if __name__ == "__main__":
    # train_bof_model(hog_file, hog_model, codebook_size)
    # train_bof_model(sc_file, sc_model, codebook_size)
    train_bof_model(spark_file, spark_model, codebook_size)

    # create_docs(docs_folder,
    #             [hog_model, sc_model, spark_model],
    #             [hog_file, sc_file, spark_file],
    #             [codebook_size, codebook_size, codebook_size])
