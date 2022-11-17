import os
import pickle
import tqdm


import numpy
import scipy
from sklearn import decomposition
from transformers import BertTokenizer

from configs.config import Config
from data.load import load

embedding_path = '../data/embeddings/embeddings_model_0.npy'
embedding_path_2 = '../data/embeddings/embeddings_model_1.npy'


def calculate_distances(embedding_path):
    with open(embedding_path, 'rb') as f:
        embeddings = numpy.load(f)

    with open(embedding_path_2, 'rb') as f:
        embeddings_2 = numpy.load(f)

    classes = load('../data/pickled_data_train.pkl', None, False)
    classes = classes.tensors[2].numpy()
    class_vals = numpy.sort(numpy.unique(classes))

    splits = []
    pca = decomposition.PCA(10)
    split_pca = pca.fit_transform(embeddings)
    print("PCA computed")
    IV = numpy.cov(split_pca, rowvar=False)
    print("COV computed")
    IV = numpy.linalg.matrix_power(IV, -1)
    print("IV computed")
    split_pca_2 = pca.fit_transform(embeddings_2)
    print("PCA 2 computed")
    IV_2 = numpy.cov(split_pca_2, rowvar=False)
    print("COV 2 computed")
    IV_2 = numpy.linalg.matrix_power(IV_2, -1)
    print("IV 2 computed")
    distances = numpy.zeros([split_pca.shape[0], split_pca.shape[0]])
    distances_2 = numpy.zeros([split_pca.shape[0], split_pca.shape[0]])
    for i in tqdm.tqdm(range(split_pca.shape[0])):
        for j in range(split_pca.shape[0]):
            distance = (split_pca[0] - split_pca[1]).dot(IV).dot(split_pca[0]-split_pca[1]).T
            distances[i, j] = numpy.sqrt(distance)

            distance_2 = (split_pca_2[0] - split_pca_2[1]).dot(IV_2).dot(split_pca_2[0]-split_pca_2[1]).T
            distances_2[i, j] = numpy.sqrt(distance_2)

    with open('../data/PMAL/distances/model_0/distances_all.npy', 'wb') as f:
        numpy.save(f, distances)

    with open('../data/PMAL/distances/model_1/distances_all.npy', 'wb') as f:
        numpy.save(f, distances_2)

    # for val in class_vals:
    #     idx = numpy.where(classes == val)[0]
    #     split = embeddings[idx]
    #     split_2 = embeddings_2[idx]
    #     pca = decomposition.PCA(250)
    #     split_pca = pca.fit_transform(split)
    #     distance = scipy.spatial.distance.cdist(split_pca, split_pca, metric='mahalanobis')
    #
    #     with open('../data/PMAL/distances/model_0/distances_class_' + str(val) + '.npy', 'wb') as f:
    #         numpy.save(f, distance)
    #     with open('../data/PMAL/PCA/pca_class_' + str(val) + '.pkl', 'wb') as f:
    #         pickle.dump(pca, f)
    #
    #     split_pca = pca.fit_transform(split_2)
    #     distance = scipy.spatial.distance.cdist(split_pca, split_pca, metric='mahalanobis')
    #
    #     with open('../data/PMAL/distances/model_1/distances_class_' + str(val) + '.npy', 'wb') as f:
    #         numpy.save(f, distance)
    #
    #     print("Class " + str(val) + " distances calculated.")


calculate_distances(embedding_path)
