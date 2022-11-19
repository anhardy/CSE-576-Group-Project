import datetime
import time

import torch
from torch.utils.data import random_split, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertForSequenceClassification

from PMAL.load_candidates import load_candidates
from data.load import load
from training.validate import validate

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))

    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(config):
    # loss_fcn = nn.CrossEntropyLoss
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.num_outputs)
    if config.load_train is True:
        model.load_state_dict(torch.load(config.load_path))
    model.to(device)

    optimizer_opts = {"lr": config.lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-5}
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_opts)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config.lr, max_lr=config.lr * 3, cycle_momentum=False)

    dataset = load(config.train_path, tokenizer, config.pickle_data)

    candidate_embeddings = load_candidates('data/embeddings/train/train_embeddings.npy', dataset)

    train_size = int((1 - config.validation_split) * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_set,
        sampler=RandomSampler(train_set),
        batch_size=config.batch_size
    )
    validation_dataloader = DataLoader(
        val_set,
        sampler=SequentialSampler(val_set),
        batch_size=config.batch_size
    )

    max_f1 = -9999999999
    min_loss = 999999999

    for epoch in range(config.epochs):
        # Training step
        model.train()
        total_train_loss = 0
        t0 = time.time()
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)

                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()
            result = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=True)
            loss = result.loss
            logits = result.logits
            total_train_loss += loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # Validation step
        val_f1, val_loss = validate(model, validation_dataloader, device)

        # Best model criteria
        print("f1: " + str(val_f1))
        print("max f1: " + str(max_f1))
        if val_f1 > max_f1:
            max_f1 = val_f1
            if config.save_mode == 1:
                print("Highest validation score. Saving model.")
                torch.save(model.state_dict(), config.save_path)

        if val_loss < min_loss:
            min_loss = val_loss
            if config.save_mode == 0:
                print("Lowest validation loss. Saving model.")
                torch.save(model.state_dict(), config.save_path)
