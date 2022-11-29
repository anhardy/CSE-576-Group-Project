import numpy
import torch
from torch.utils.data import TensorDataset

from data.load import load


# Loads high quality candidates. Maps each label to a set of embeddings
def load_candidates(embedding_path, train_data):
    # Retrieve embeddings for full training data
    with open(embedding_path, 'rb') as f:
        embeddings = numpy.load(f)
    f.close()

    embedding_map = {}

    # Get labels
    classes = train_data.tensors[2].numpy()
    class_vals = numpy.sort(numpy.unique(classes))
    # For each in domain class
    for val in class_vals:
        if val == 100:
            continue

        # Load corresponding candidate
        idx = numpy.where(classes == val)[0]
        split = embeddings[idx]

        candidate_path = 'data/PMAL/candidates/indices/indices_class_' + str(val) + '.npy'

        with open(candidate_path, 'rb') as f:
            candidates = numpy.load(f)

        candidate_embeddings = split[candidates]
        embedding_map[val] = candidate_embeddings

    return embedding_map


# Loads high quality candidates in a method more suitable for training with them
def load_candidates_train(train_data):
    tensors = []
    masks = []
    truths = []

    classes = train_data.tensors[2].numpy()
    class_vals = numpy.sort(numpy.unique(classes))
    # For each class
    for val in class_vals:
        if val == 100:
            continue
        # Load corresponding candidates as tensors
        idx = numpy.where(classes == val)[0]
        split = train_data.tensors[0].numpy()[idx]
        mask = train_data.tensors[1].numpy()[idx]
        truth = train_data.tensors[2].numpy()
        truth = truth[idx]
        # truth = numpy.expand_dims(truth, 1)

        candidate_path = 'data/PMAL/candidates/indices/indices_class_' + str(val) + '.npy'

        with open(candidate_path, 'rb') as f:
            candidates = numpy.load(f)

        candidate_tensors = split[candidates]
        mask = mask[candidates]
        truth = truth[candidates]
        tensors.append(candidate_tensors)
        masks.append(mask)
        truths.append(truth)

    # Create tensor dataset

    dataset = TensorDataset(torch.tensor(numpy.concatenate(tensors)), torch.tensor(numpy.concatenate(masks)),
                            torch.tensor(numpy.concatenate(truths)))

    return dataset
