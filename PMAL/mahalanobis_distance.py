import os

import numpy
import scipy
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

    # with open('../data/PMAL/distances_model_0_class_0.npy', 'rb') as f:
    #     test = numpy.load(f)

    # splits = []

    for val in class_vals:
        idx = numpy.where(classes == val)[0]
        split = embeddings[idx]
        distance = scipy.spatial.distance.cdist(split, split, metric='mahalanobis')

        with open('../data/PMAL/distances_model_0_class_' + str(val) + '.npy', 'wb') as f:
            numpy.save(f, distance)

        print("Class " + str(val) + " distances calculated.")

    return


calculate_distances(embedding_path)
