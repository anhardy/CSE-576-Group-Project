import numpy
import numpy as np
import scipy.spatial.distance

from data.load import load


def calc_robustness():
    classes = load('../data/pickled_data_train.pkl', None, False)
    classes = classes.tensors[2].numpy()
    class_vals = numpy.sort(numpy.unique(classes))
    for val in class_vals:
        r_scores = []
        candidates = []
        candidate_scores = []
        with open('../data/PMAL/distances/model_0/distances_class_' + str(val) + '.npy', 'rb') as f:
            model_0_samples = scipy.spatial.distance.squareform(numpy.load(f))
        with open('../data/PMAL/distances/model_1/distances_class_' + str(val) + '.npy', 'rb') as f:
            model_1_samples = scipy.spatial.distance.squareform(numpy.load(f))
        model_0_samples = numpy.nan_to_num(model_0_samples)
        model_1_samples = numpy.nan_to_num(model_1_samples)

        for i, sample in enumerate(model_0_samples):
            model_0_sample = model_0_samples[i]
            model_1_sample = model_1_samples[i]
            euc_dist = scipy.spatial.distance.euclidean(model_0_sample, model_1_sample)
            r_score = numpy.exp(-euc_dist)
            r_scores.append(r_score)

        max_r = max(r_scores)
        print("Max Robustness Score: " + str(max_r))
        for i, r_score in enumerate(r_scores):
            if r_score > 0.8 * max_r:
                candidates.append(i)
                candidate_scores.append(r_score)
        print("Num candidates: " + str(len(candidates)))
        with open('../data/PMAL/candidates/per_class_full_cov/indices/indices_class_' + str(val) + '.npy', 'wb') as f:
            numpy.save(f, numpy.array(candidates))
        with open('../data/PMAL/candidates/per_class_full_cov/scores/scores_class_' + str(val) + '.npy', 'wb') as f:
            numpy.save(f, numpy.array(candidate_scores))
    return


def calc_robustness_full_set():
    classes = load('../data/pickled_data_train.pkl', None, False)
    classes = classes.tensors[2].numpy()
    class_vals = numpy.sort(numpy.unique(classes))
    model_0_distances = "G:\\NLP Data\\distances\\model_0\\distances_all_PCA.npy"
    model_1_distances = "G:\\NLP Data\\distances\\model_1\\distances_all_PCA.npy"
    # mmap_mode is set to r or else you'll be loading two 20 gigabyte matrices into memory
    distances_1 = np.load(model_0_distances, mmap_mode='r')
    distances_2 = np.load(model_1_distances, mmap_mode='r')
    for val in class_vals:
        r_scores = []
        candidates = []
        candidate_scores = []
        indices = np.where(classes == val)[0]

        model_0_samples = distances_1[indices][:, indices]
        model_1_samples = distances_2[indices][:, indices]

        for i, sample in enumerate(model_0_samples):
            model_0_sample = model_0_samples[i]
            model_1_sample = model_1_samples[i]
            euc_dist = scipy.spatial.distance.euclidean(model_0_sample, model_1_sample)
            r_score = numpy.exp(-euc_dist)
            r_scores.append(r_score)

        max_r = max(r_scores)
        print("Max Robustness Score: " + str(max_r))
        for i, r_score in enumerate(r_scores):
            if r_score > 0.8 * max_r:
                candidates.append(i)
                candidate_scores.append(r_score)
        print("Num candidates: " + str(len(candidates)))
        with open('../data/PMAL/candidates/per_class_full_cov_pca/indices/indices_class_' + str(val) + '.npy', 'wb') as f:
            numpy.save(f, numpy.array(candidates))
        with open('../data/PMAL/candidates/per_class_full_cov_pca/scores/scores_class_' + str(val) + '.npy', 'wb') as f:
            numpy.save(f, numpy.array(candidate_scores))
    return


# calc_robustness()
calc_robustness_full_set()
