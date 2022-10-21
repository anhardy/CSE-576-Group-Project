import numpy
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
            model_0_samples = numpy.load(f)
        with open('../data/PMAL/distances/model_1/distances_class_' + str(val) + '.npy', 'rb') as f:
            model_1_samples = numpy.load(f)
        model_0_samples = numpy.nan_to_num(model_0_samples)
        model_1_samples = numpy.nan_to_num(model_1_samples)

        for i, sample in enumerate(model_0_samples):
            model_0_sample = model_0_samples[i]
            model_1_sample = model_1_samples[i]
            euc_dist = scipy.spatial.distance.euclidean(model_0_sample, model_1_sample)
            r_score = numpy.exp(-euc_dist)
            r_scores.append(r_score)

        max_r = max(r_scores)
        for i, r_score in enumerate(r_scores):
            if r_score > 0.8 * max_r:
                candidates.append(i)
                candidate_scores.append(r_score)
        with open('../data/PMAL/candidates/indices/indices_class_' + str(val) + '.npy', 'wb') as f:
            numpy.save(f, candidates)
        with open('../data/PMAL/candidates/scores/scores_class_' + str(val) + '.npy', 'wb') as f:
            numpy.save(f, candidate_scores)
    return


calc_robustness()