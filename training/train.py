import torch
from torch import nn

from transformers import BertTokenizer, BertForSequenceClassification

from data.load import load
from training.validate import validate

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"


def train(config):
    loss_fcn = nn.CrossEntropyLoss
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.num_outputs)
    if config.load_train is True:
        # TODO unsure if can load state dict this way with bert model
        model.load_state_dict(torch.load(config.load_path))
    model.to(device)

    optimizer_opts = {"lr": config.lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-5}
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_opts)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=1e-3 * 3, cycle_momentum=False)

    X, y = load(config.train_path, tokenizer)

    # TODO validation split and class balancing if needed

    for epoch in range(config.epochs):
        # Train

        val_f1, val_loss = validate(X_val, y_val, loss_fcn)
        continue
