import os

import numpy
import scipy
from sklearn import decomposition
from transformers import BertTokenizer

from configs.config import Config
from data.load import load

embedding_path = '../data/PMAL/embeddings_model_0.npy'


def calculate_distances(embedding_path):
    with open(embedding_path, 'rb') as f:
        embeddings = numpy.load(f)

    classes = load('../data/pickled_data_train.pkl', None, False)
    classes = classes.tensors[2].numpy()
    class_vals = numpy.sort(numpy.unique(classes))

    # with open('../data/PMAL/distances/model_0/distances_class_0.npy', 'rb') as f:
    #     test = numpy.load(f)

    # splits = []

    for val in class_vals:
        idx = numpy.where(classes == val)[0]
        split = embeddings[idx]
        pca = decomposition.PCA()
        split_pca = pca.fit_transform(split)
        # cov_estimate = numpy.linalg.inv(numpy.matmul(split, split.T))
        distance = scipy.spatial.distance.cdist(split_pca, split_pca, metric='mahalanobis')

        with open('../data/PMAL/distances/model_0/distances_class_' + str(val) + '.npy', 'wb') as f:
            numpy.save(f, distance)
        with open('../data/PMAL/PCA/pca_class_' + str(val) + '.pkl', 'wb') as f:
            numpy.save(f, pca)

        print("Class " + str(val) + " distances calculated.")

    return


calculate_distances(embedding_path)
