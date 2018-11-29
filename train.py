"""Train classifier and test it using features from images."""

import numpy as np

# import pickle
from data import read_docs_create_distributions
from data import read_data, get_distribution

from bof import get_words, get_models
from settings import test_file, docs_folder
from settings import codebook_size
from settings import hog_model, sc_model, spark_model

# from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score


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
    X_test, y_test, _ = read_data(test_file)
    X_test = get_word_distribution(X_test, model_files, num_words, sketch_images=True)

    # NOTE: You can save test set so that you can later try different models
    # pickle.dump((X_test, y_test), open("test.pkl", "wb"))
    # X_test, y_test = pickle.load(open("test.pkl", "rb"))

    # Train model
    print("Training model")
    gamma = 3e-3
    model = SVC(kernel="rbf", gamma=gamma, C=1)
    model.fit(X_train, y_train)

    # Testing
    print("Testing model")

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print("Accuracy: {} \n F1 Score: {}".format(acc, acc))
