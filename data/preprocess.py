import torch


def preprocess(data, tokenizer):
    # Drop rows with NaN values
    data = data.dropna()
    x = list(data['review'].values)
    y = torch.tensor(data['label'].values)
    domains = data['product_domain']

    print("Tokenizing...")

    x = tokenizer(x, return_tensors='pt', padding=True, truncation=True)

    print("Tokenization complete.")

    encoded_x = x['input_ids']
    attn_mask = x['attention_mask']

    return encoded_x, attn_mask, y, domains
