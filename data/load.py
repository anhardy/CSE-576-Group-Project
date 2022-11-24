import pickle
import sys

import pandas as pd
from torch.utils.data import TensorDataset

from data.preprocess import preprocess


def load(path, tokenizer, pickle_data, train=True):
    # Load data based on file type
    if ".xlsx" in path:
        df = pd.read_excel(path)
        new_path = path.replace('.xlsx', '.pkl')
    elif '.csv' in path:
        df = pd.read_csv(path)
        new_path = path.replace('.csv', '.pkl')
    # Expects a TensorDataset object saved as a .pkl file
    elif '.pkl' in path:
        file = open(path, 'rb')
        dataset = pickle.load(file)
        return dataset
    else:
        print("File type not yet supported")
        sys.exit(-1)

    X, attn_mask, y, domains = preprocess(df, tokenizer)

    dataset = TensorDataset(X, attn_mask, y)
    # Save dataset as .pkl to save on preprocessing time. Tokenization is slow.
    if pickle_data:
        if train:
            file = open(new_path, 'wb')
        else:
            file = open(new_path, 'wb')

        pickle.dump(dataset, file)

    return dataset
