import numpy

from data.load import load


def load_candidates(embedding_path, train_data):
    with open(embedding_path, 'rb') as f:
        embeddings = numpy.load(f)
    f.close()

    embedding_map = {}

    classes = train_data.tensors[2].numpy()
    class_vals = numpy.sort(numpy.unique(classes))
    for val in class_vals:
        idx = numpy.where(classes == val)[0]
        split = embeddings[idx]

        candidate_path = 'data/PMAL/candidates/indices/indices_class_' + str(val) + '.npy'

        with open(candidate_path, 'rb') as f:
            candidates = numpy.load(f)

        candidate_embeddings = split[candidates]
        embedding_map[val] = candidate_embeddings

    return embedding_map