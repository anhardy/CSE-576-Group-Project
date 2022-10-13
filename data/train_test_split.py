# This script is for splitting training data into a train and test split. This is NOT intended for validation.
# This is meant to create a specific test set to test models trained with a separate validation split while
# removing those samples from training/validation entirely. For the time being, this script is intended to be run
# separately from everything else and has no way to run it directly from the main script.

import pandas as pd

# How much of training data to keep
train_retention = 0.95
path = 'train.xlsx'

if ".xlsx" in path:
    df = pd.read_excel(path)
elif ".csv" in path:
    df = pd.read_csv(path)


data = df.dropna()

train_df = data.sample(frac=train_retention)

test_df = data.drop(train_df.index)

train_df.to_excel('train_split.xlsx')
test_df.to_excel('test_split.xlsx')

