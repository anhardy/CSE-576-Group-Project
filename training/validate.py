import numpy
import torch
from sklearn.metrics import classification_report, f1_score
from PMAL.loss import reject

from PMAL.load_candidates import load_candidates


def validate(model, dataloader, device):
    model.eval()
    total_eval_loss = 0
    pred = []
    truth = []
    for i, batch in enumerate(dataloader):
        if i % 10 == 0:
            print("Batch " + str(i) + " of " + str(len(dataloader)))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():
            result = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=True)

        loss = result.loss
        logits = result.logits

        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy().argmax(1)
        label_ids = b_labels.to('cpu').numpy()

        pred.append(logits)
        truth.append(label_ids)
    pred = numpy.concatenate(pred)
    truth = numpy.concatenate(truth)
    report = classification_report(truth, pred)
    print(report)
    avg_val_loss = total_eval_loss / len(dataloader)
    f1 = f1_score(truth, pred, average='weighted')

    # validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    return f1, avg_val_loss


def test_ood(model, dataloader, device, dataset):
    candidate_embeddings = load_candidates('data/embeddings/train/test_embeddings.npy', dataset)
    model.eval()
    total_eval_loss = 0
    pred = []
    truth = []
    for i, batch in enumerate(dataloader):
        if i % 10 == 0:
            print("Batch " + str(i) + " of " + str(len(dataloader)))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():
            result = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=True)

        loss = result.loss
        logits = result.logits

        total_eval_loss += loss.item()

        prediction = logits.detach().cpu().numpy().argmax(1)
        label_ids = b_labels.to('cpu').numpy()

        pred.append(prediction)
        truth.append(label_ids)
    pred = numpy.concatenate(pred)
    truth = numpy.concatenate(truth)
    report = classification_report(truth, pred)
    print(report)
    avg_val_loss = total_eval_loss / len(dataloader)
    f1 = f1_score(truth, pred, average='weighted')

    # validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    return f1, avg_val_loss
