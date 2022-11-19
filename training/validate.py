import numpy
import torch
from sklearn.metrics import classification_report, f1_score
from PMAL.loss import reject
from torch.nn.functional import softmax

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


def test_ood(model, dataloader, device, dataset, train_set):
    candidate_embeddings = load_candidates('data/embeddings/train/train_embeddings.npy', train_set)
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
            result = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=True,
                           output_hidden_states=True)
        embedding = result.hidden_states[12][:, 0].detach().cpu().numpy()

        loss = result.loss
        logits = result.logits

        total_eval_loss += loss.item()

        prediction = logits.detach().cpu().numpy().argmax(1)
        classes = numpy.unique(prediction)
        prob = softmax(logits).detach().cpu().numpy().max(1)
        for pred_class in classes:
            # TODO get embeddings of test samples that predicted this class
            samples = embedding[numpy.where(prediction==pred_class)]
            pred_set = prediction[numpy.where(prediction==pred_class)]
            prob_set = prob[numpy.where(prediction==pred_class)]
            candidate_set = candidate_embeddings[pred_class]
            new_pred_set = reject(samples, pred_set, candidate_set, 5, prob_set, 0.5, 0.5)
        # sample = numpy.expand_dims(embedding[0], 1)
        # pred = reject(sample, prediction, candidate_embeddings[pred], 2, prob, 5, 0.5)

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
