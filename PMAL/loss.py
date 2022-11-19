import pickle

import numpy
from scipy.special import softmax


def L_p():
    pass


def opt_loss(margin, scale, balance, ce_loss):
    return ce_loss + balance * L_p(scale, margin)


def reject(sample, prediction, p_set, scale, prob, dist_thresh, prob_thresh):
    dist = distance(sample, p_set, scale)
    if dist <= dist_thresh and prob >= prob_thresh:
        return prediction
    else:
        return 100
    # End result should be a vector of dimensionality n x 1 where n is number of samples in this set


def distance(sample, p_set, scale):
    # Calculate z_att
    z_a = z_att(sample, p_set, scale)
    # Calculate distance
    dist = 1 - numpy.dot(sample.T, z_a.T) / (numpy.linalg.norm(sample.T) * numpy.linalg.norm(z_a))
    return dist


def z_att(sample, prototype_set, scale):
    z = numpy.dot(softmax(numpy.dot(sample.T, prototype_set.T) / numpy.sqrt(scale)), prototype_set)

    return z
