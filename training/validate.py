import numpy
import pandas
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


# Testing out of domain samples
def test_ood(model, dataloader, device, original_labels, authors, reviews, products, overalls, num_words, train_set):
    candidate_embeddings = load_candidates('data/embeddings/train/train_embeddings.npy', train_set)
    model.eval()
    total_eval_loss = 0
    pred = []
    truth = []
    all_logits = []
    distances = []
    probabilities = []
    for i, batch in enumerate(dataloader):
        if i % 10 == 0:
            print("Batch " + str(i) + " of " + str(len(dataloader)))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():
            result = model(b_input_ids, attention_mask=b_input_mask, return_dict=True,
                           output_hidden_states=True)
        embedding = result.hidden_states[12][:, 0].detach().cpu().numpy()

        # loss = result.loss
        logits = result.logits

        # total_eval_loss += loss.item()

        prediction = logits.detach().cpu().numpy().argmax(1)
        classes = numpy.unique(prediction)
        prob = softmax(logits).detach().cpu().numpy().max(1)
        probabilities.append(prob)
        logits = logits.detach().cpu().numpy()
        # For every class predicted this batch
        for pred_class in classes:
            # Reject using candidates from this set
            indices = numpy.where(prediction == pred_class)
            samples = embedding[indices]
            pred_set = prediction[indices]
            prob_set = prob[indices]
            candidate_set = candidate_embeddings[pred_class]
            new_pred_set, dist = reject(samples, pred_set, candidate_set, 1, prob_set, 0.99, 0.99)
            distances.append(dist)
            prediction[indices] = new_pred_set

        label_ids = b_labels.to('cpu').numpy()

        pred.append(prediction)
        truth.append(label_ids)
        all_logits.append(logits)

    pred = numpy.concatenate(pred)
    truth = numpy.concatenate(truth)
    all_logits = numpy.concatenate(all_logits)
    report = classification_report(truth, pred)
    distances = numpy.concatenate(distances)
    probabilities = numpy.concatenate(probabilities)
    dist_OOD = distances[numpy.where(truth == 100)].mean()
    prob_OOD = probabilities[numpy.where(truth == 100)].mean()
    dist_ID = distances[numpy.where(truth != 100)].mean()
    prob_ID = probabilities[numpy.where(truth != 100)].mean()
    print("Avg OOD distance: " + str(dist_OOD))
    print("Avg OOD prob: " + str(prob_OOD))
    print("Avg ID distance: " + str(dist_ID))
    print("Avg ID prob: " + str(prob_ID))
    print(report)
    avg_val_loss = total_eval_loss / len(dataloader)
    f1 = f1_score(truth, pred, average='weighted')

    print("f1: " + str(f1))
    pred[numpy.where(pred == 100)] = -1
    logit_col = [i for i in range(logits.shape[1])]
    columns = ['Author', 'Label', 'Prediction', 'Review', 'Product_Domain', 'Overall', '#Words'] + logit_col
    report_frame = pandas.DataFrame(numpy.column_stack([authors, original_labels, pred, reviews, products, overalls,
                                                        num_words, all_logits]), columns=columns)

    report_frame.to_csv('ood_set_3.csv')

    # validation_time = format_time(time.time() - t0)

    # print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    return f1, avg_val_loss
