import torch
from torch.utils.data import random_split, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertForSequenceClassification

from data.load import load
from training.validate import validate

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"


def test(config):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.num_outputs)
    model.load_state_dict(torch.load(config.load_path))
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dataset = load(config.test_path, tokenizer, config.pickle_data, False)

    test_dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=config.batch_size
    )

    test_f1, test_loss = validate(model, test_dataloader, device)
