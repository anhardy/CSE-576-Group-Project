import os
import pickle
import tqdm

import torch
import numpy
import scipy
from sklearn import decomposition
from transformers import BertTokenizer

from configs.config import Config
from data.load import load

embedding_path = '../data/embeddings/embeddings_model_0.npy'
embedding_path_2 = '../data/embeddings/embeddings_model_1.npy'


# Calculates pairwise distances between each sample in the class only, with respect to the entire covariance matrix
def calculate_distances_per_class_full_cov():
    with open(embedding_path, 'rb') as f:
        embeddings = numpy.load(f)

    with open(embedding_path_2, 'rb') as f:
        embeddings_2 = numpy.load(f)
    classes = load('../data/pickled_data_train.pkl', None, False)
    classes = classes.tensors[2].numpy()
    class_vals = numpy.sort(numpy.unique(classes))

    # with open('../data/PMAL/distances/model_0/distances_class_0.npy', 'rb') as f:
    #     test = numpy.load(f)

    # splits = []

    IV = numpy.cov(embeddings, rowvar=False)
    print("COV computed")
    IV = numpy.linalg.matrix_power(IV, -1)
    print("IV computed")
    IV_2 = numpy.cov(embeddings_2, rowvar=False)
    print("COV 2 computed")
    IV_2 = numpy.linalg.matrix_power(IV_2, -1)
    print("IV 2 computed")

    for val in class_vals:
        idx = numpy.where(classes == val)[0]
        split = embeddings[idx]
        split_2 = embeddings_2[idx]
        distance = scipy.spatial.distance.pdist(split, metric='mahalanobis', VI=IV)

        with open('../data/PMAL/distances/model_0/distances_class_' + str(val) + '.npy', 'wb') as f:
            numpy.save(f, distance)

        distance = scipy.spatial.distance.pdist(split_2, metric='mahalanobis', VI=IV_2)

        with open('../data/PMAL/distances/model_1/distances_class_' + str(val) + '.npy', 'wb') as f:
            numpy.save(f, distance)

        print("Class " + str(val) + " distances calculated.")


# Calculates pairwise distances within each class, with respect to the covariance matrix of only that class. Requires
#   PCA if class observations < Embedding features
def calculate_distances_per_class():
    with open(embedding_path, 'rb') as f:
        embeddings = numpy.load(f)

    with open(embedding_path_2, 'rb') as f:
        embeddings_2 = numpy.load(f)
    classes = load('../data/pickled_data_train.pkl', None, False)
    classes = classes.tensors[2].numpy()
    class_vals = numpy.sort(numpy.unique(classes))

    # with open('../data/PMAL/distances/model_0/distances_class_0.npy', 'rb') as f:
    #     test = numpy.load(f)

    # splits = []

    for val in class_vals:
        idx = numpy.where(classes == val)[0]
        split = embeddings[idx]
        split_2 = embeddings_2[idx]
        pca = decomposition.PCA(250)
        split_pca = pca.fit_transform(split)
        distance = scipy.spatial.distance.pdist(split_pca, metric='mahalanobis')

        with open('../data/PMAL/distances/model_0/distances_class_' + str(val) + '.npy', 'wb') as f:
            numpy.save(f, distance)
        with open('../data/PMAL/PCA/pca_class_' + str(val) + '.pkl', 'wb') as f:
            pickle.dump(pca, f)

        split_pca = pca.fit_transform(split_2)
        distance = scipy.spatial.distance.pdist(split_pca, metric='mahalanobis')

        with open('../data/PMAL/distances/model_1/distances_class_' + str(val) + '.npy', 'wb') as f:
            numpy.save(f, distance)

        print("Class " + str(val) + " distances calculated.")


# Calculates pairwise distances for ENTIRE dataset using Numpy operations. Whether in a loop or with scipy, this takes
#   a VERY long time unless PCA is used to significantly reduce embedding feature space
def calculate_distances_all_numpy():
    with open(embedding_path, 'rb') as f:
        embeddings = numpy.load(f)

    with open(embedding_path_2, 'rb') as f:
        embeddings_2 = numpy.load(f)

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
    distances_2 = numpy.zeros([split_pca_2.shape[0], split_pca_2.shape[0]])
    for i in tqdm.tqdm(range(split_pca.shape[0])):
        for j in range(i, split_pca.shape[0]):
            distance = numpy.sqrt((split_pca[i] - split_pca[j]).dot(IV).dot(split_pca[i] - split_pca[j]).T)
            distances[i, j] = distance
            distances[j, i] = distance

            distance_2 = numpy.sqrt((split_pca_2[i] - split_pca_2[j]).dot(IV_2).dot(split_pca_2[i] - split_pca_2[j]).T)
            distances_2[i, j] = distance_2
            distances_2[j, i] = distance_2

    with open('../data/PMAL/distances/model_0/distances_all.npy', 'wb') as f:
        numpy.save(f, distances)

    with open('../data/PMAL/distances/model_1/distances_all.npy', 'wb') as f:
        numpy.save(f, distances_2)


# Computes pairwise distances for entire dataset using tensors and GPU. Significantly faster when using larger
#   feature sizes but will still take a long time. Should not be used for small embedding feature sizes as it
#   will diminish performance compared to CPU operations.
def calculate_distances_all_torch():
    with open(embedding_path, 'rb') as f:
        embeddings = numpy.load(f)

    with open(embedding_path_2, 'rb') as f:
        embeddings_2 = numpy.load(f)

    pca = decomposition.PCA()
    split_pca = torch.tensor(pca.fit_transform(embeddings)).to('cuda')
    print("PCA computed")
    IV = torch.cov(split_pca.T)
    print("COV computed")
    IV = torch.inverse(IV)
    print("IV computed")
    split_pca_2 = torch.tensor(pca.fit_transform(embeddings_2)).to('cuda')
    print("PCA 2 computed")
    IV_2 = torch.cov(split_pca_2.T)
    print("COV 2 computed")
    IV_2 = torch.inverse(IV_2)
    print("IV 2 computed")
    distances = torch.zeros([split_pca.size()[0], split_pca.size()[0]]).to('cuda')
    distances_2 = torch.zeros([split_pca.size()[0], split_pca.size()[0]]).to('cuda')
    for i in tqdm.tqdm(range(split_pca.size()[0])):
        for j in range(i, split_pca.size()[0]):
            diff = split_pca[i] - split_pca[j]
            distance = torch.sqrt(torch.dot(diff, torch.matmul(IV, diff)))
            # distance = (split_pca[i] - split_pca[j]).dot(IV).dot(split_pca[i]-split_pca[j]).T
            distances[i, j] = distance
            distances[j, i] = distance

            diff = split_pca_2[i] - split_pca_2[j]
            distance_2 = torch.sqrt(torch.dot(diff, torch.matmul(IV_2, diff)))
            # distance_2 = (split_pca_2[i] - split_pca_2[j]).dot(IV_2).dot(split_pca_2[i]-split_pca_2[j]).T
            distances_2[i, j] = distance_2
            distances_2[j, i] = distance_2

    with open('../data/PMAL/distances/model_0/distances_all.npy', 'wb') as f:
        numpy.save(f, distances.detach().cpu().numpy())

    with open('../data/PMAL/distances/model_1/distances_all.npy', 'wb') as f:
        numpy.save(f, distances_2.detach().cpu().numpy())


# calculate_distances_all_torch()
# calculate_distances_all_numpy()
calculate_distances_per_class_full_cov()

