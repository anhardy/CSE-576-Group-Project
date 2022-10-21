import os

import numpy
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import random_split, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertForSequenceClassification

from data.load import load

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"


def get_embeddings(config):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.num_outputs)
    model.load_state_dict(torch.load(config.load_path, map_location=torch.device(device)))
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dataset = load(config.train_path, tokenizer, config.pickle_data, False)

    test_dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=config.batch_size
    )

    embeddings = test(model, test_dataloader, device)

    with open('data/PMAL/embeddings_model_1.npy', 'wb') as f:
        numpy.save(f, embeddings)

    return


def test(model, dataloader, device):
    model.eval()
    total_eval_loss = 0
    pred = []
    truth = []
    embeddings = []
    for i, batch in enumerate(dataloader):
        if i % 10 == 0:
            print("Batch " + str(i) + " of " + str(len(dataloader)))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():
            result = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=True,
                           output_hidden_states=True)

        loss = result.loss
        logits = result.logits
        # Retrieve embedding of classification token (first token in sequence)
        embedding = result.hidden_states[12][:, 0].detach().cpu().numpy()
        embeddings.append(embedding)

        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy().argmax(1)
        label_ids = b_labels.to('cpu').numpy()

        pred.append(logits)
        truth.append(label_ids)
    pred = numpy.concatenate(pred)
    truth = numpy.concatenate(truth)
    embeddings = numpy.concatenate(embeddings)
    report = classification_report(truth, pred)
    print(report)
    avg_val_loss = total_eval_loss / len(dataloader)
    f1 = f1_score(truth, pred, average='weighted')

    # validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    return embeddings
