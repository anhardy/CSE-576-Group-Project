import pickle

import numpy
import numpy as np
from scipy.special import softmax


# Loss function defined in PMAL paper
def L_p(samples, true_labels, candidate_labels, candidates, scale, margin):
    all_distances = []
    for true_label in numpy.unique(true_labels):
        true_idx_candidates = numpy.where(candidate_labels == true_label)
        true_set_candidates = candidates[true_idx_candidates]
        true_idx_samples = numpy.where(true_labels == true_label)
        true_set_samples = samples[true_idx_samples]
        distances = []
        distances_true = distance(true_set_samples, true_set_candidates, scale)
        for label in numpy.unique(candidate_labels):
            if label == true_label:
                continue
            candidate_idx = numpy.where(candidate_labels == label)
            candidate_set = candidates[candidate_idx]
            dist = distance(true_set_samples, candidate_set, scale)
            distances.append(dist)
            # dist_sum += abs(distance(true_set_samples, true_set_candidates) - min_dist + margin)
        distances = numpy.stack(distances)
        min_dist = numpy.min(distances, axis=0)
        dist_sum = numpy.abs(distances_true - min_dist + margin)
        all_distances.append(dist_sum)
    all_distances = numpy.concatenate(all_distances)

    return numpy.mean(all_distances)


# Optimization loss function combining cross-entropy loss with L_p loss
def opt_loss(samples, p_sets, true_set, margin, scale, balance, ce_loss):
    return ce_loss + balance * L_p(samples, p_sets, true_set, scale, margin)


def reject(sample, prediction, p_set, scale, prob, dist_thresh, prob_thresh):
    dist = distance(sample, p_set, scale)
    # Reject  all samples that fail to meet either distance or probability threshold.
    rejections = numpy.where((dist > dist_thresh) | (prob < prob_thresh))[0]
    prediction[rejections] = 100

    # End result should be a vector of dimensionality n x 1 where n is number of samples in this set
    return prediction, dist


def distance(sample, p_set, scale):
    # Calculate z_att
    z_a = z_att(sample, p_set, scale)
    # Calculate distance
    dist = 1 - numpy.sum(sample * z_a, axis=1) / (
            numpy.linalg.norm(sample.T, axis=0) * numpy.linalg.norm(z_a.T, axis=0))

    return dist


def z_att(sample, prototype_set, scale):
    z = numpy.dot(softmax(numpy.dot(sample, prototype_set.T) / numpy.sqrt(scale), axis=1), prototype_set)

    return z
