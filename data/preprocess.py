import os

import numpy
import torch


def preprocess(data, tokenizer):
    # Drop rows with NaN values
    data = data.dropna()
    x = list(data['review'].values)
    y = torch.tensor(data['real_label'].values)
    # x_n = numpy.array(x)
    # y_n = numpy.array(list(data['real_label'].values))
    # test = os.getcwd()
    # with open('data/x_numpy.npy', 'wb') as f:
    #     numpy.save(f, x_n)
    # with open('data/y_numpy.npy', 'wb') as f:
    #     numpy.save(f, y_n)

    # 0 - 99 are closed set. All OOD considered class 100
    y[y > 99] = 100
    domains = data['product_domain']

    print("Tokenizing...")

    x = tokenizer(x, return_tensors='pt', padding=True, truncation=True)

    print("Tokenization complete.")

    encoded_x = x['input_ids']
    attn_mask = x['attention_mask']

    return encoded_x, attn_mask, y, domains
