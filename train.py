"""Train classifier and test it using features from images."""

import numpy as np

from data import read_docs_create_distributions
from data import read_data, get_distribution

from bof import get_words, get_models
from settings import test_file, docs_folder
from settings import codebook_size
from settings import hog_model, sc_model, spark_model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def get_word_distribution(dataset, model_files, num_words, sketch_images=True):
    """Get the word distribution given the image and word models."""
    print("Reading word models")
    models = get_models(model_files)
    X = []

    for i, img in enumerate(dataset):
        print("Processing ", i)
        words = get_words(img, models, num_words, sketch_images)
        vec = get_distribution(words, sum(num_words))
        X.append(vec)

    return np.array(X)


if __name__ == "__main__":
    num_words = [codebook_size, codebook_size, codebook_size]
    model_files = [hog_model, sc_model, spark_model]

    # Read train data
    print("Read training data")
    X_train, y_train = read_docs_create_distributions(docs_folder, sum(num_words))

    # Read test data
    print("Read testing data")
    X_test, y_test = read_data(test_file)
    X_test = get_word_distribution(X_test, model_files, num_words, sketch_images=True)

    # Train model
    print("Training model")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='samples')

    print("Accuracy: {} \n F1 Score: {}".format(acc, f1))
