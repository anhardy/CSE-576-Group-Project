import pickle

import numpy
from torch.nn.functional import softmax


def L_p():
    pass


def opt_loss(margin, scale, balance, ce_loss):
    return ce_loss + balance * L_p(scale, margin)


def reject(sample, prediction, p_set, scale, prob, dist_thresh, prob_thresh):
    dist = distance(sample, prediction, p_set, scale)
    if dist <= dist_thresh and prob >= prob_thresh:
        return prediction
    else:
        return 100


def distance(sample, p_set, scale):
    # Calculate z_att
    z_a = z_att(sample, p_set, scale)
    # Calculate distance
    dist = 1 - numpy.dot(sample.T, z_a) / (numpy.linalg.norm(sample.t) * numpy.linalg.norm(z_a))
    return dist


def z_att(sample, prototype_set, scale):
    z = numpy.dot(softmax(numpy.dot(sample.T, prototype_set) / numpy.sqrt(scale)), prototype_set)

    return z
