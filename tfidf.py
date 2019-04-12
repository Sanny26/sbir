"""Retrieval using Tf-idf and cached bag of words histograms."""
import os
import numpy as np
import pickle
import pdb

from settings import docs_folder, data_folder
from settings import codebook_size
from settings import tfidf_model

from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf


def find_matching_docs(matrix, words, doc_names, num_items=10):
    scores = np.sum(matrix[::, words], axis=1).T
    docs_indices = np.flip(np.argsort(scores), axis=1)
    # pdb.set_trace()
    return [doc_names[docs_indices[0, x]] for x in range(num_items)]


def find_images(docs):
    images = []
    for d in docs:
        filename = d.split("/")[-1].split(".")[0]
        img_name = "/".join(filename.split("_")) + ".jpg"
        images.append(img_name)
    return images

if __name__ == "__main__":
    docs = os.listdir(docs_folder)
    doc_names = [os.path.join(docs_folder, x) for x in docs]
    words = {str(x): x for x in range(sum([codebook_size, codebook_size, codebook_size]))}

    print("Training model")
    model = Tfidf(input='filename',
                  ngram_range=(1, 2), vocabulary=words)
    matrix = model.fit_transform(doc_names)

    pickle.dump((doc_names, matrix), open(tfidf_model, "wb"))

    test_words = [12, 23, 1323, 234, 214, 1224, 1532]
    docs_match = find_matching_docs(matrix, test_words, doc_names)
    imgs = find_images(docs_match)
    pdb.set_trace()
