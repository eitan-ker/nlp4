from __future__ import annotations
import torch
import os
import pickle

def save_checkpoint(model, optimizer, tokenizer, num_batches, path="checkpoint.pth"):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "num_batches": num_batches,
    }, path)
    with open(path + ".tokenizer", "wb") as f:
        pickle.dump(tokenizer, f)

def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    with open(path + ".tokenizer", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer, checkpoint["num_batches"]


if __name__ == '__main__':
    import torch
    from torch import nn
    from torch import optim
    from transformer import TransformerLM
    import data
    import lm

    seq_len = 128
    batch_size = 64
    data_path = "data/"
    n_layers = 6
    n_heads = 6
    embed_size = 192
    mlp_hidden_size = embed_size * 4

    learning_rate = 5e-4
    gradient_clipping = 1.0

    num_batches_to_train = 50000

    tokenizer, tokenized_data = data.load_data(data_path)
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

    model: torch.nn.Module = TransformerLM(
        n_layers,
        n_heads,
        embed_size,
        seq_len,
        tokenizer.vocab_size(),
        mlp_hidden_size,
        with_residuals=True,
    )

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95])

    checkpoint_path = "checkpoint.pth"
    start_batch = 0

    if os.path.exists(checkpoint_path):
        tokenizer, start_batch = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Loaded checkpoint from batch {start_batch}")
    else:
        start_batch = 0

    model.train()

    num_batches = start_batch
    while True:
        for batch in data.batch_items(data_iter, batch_size):
            if num_batches >= num_batches_to_train:
                break

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            logits = model(batch_x)
            loss = lm.compute_loss(logits, batch_y)

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            num_batches += 1

            if num_batches % 1000 == 0:
                save_checkpoint(model, optimizer, tokenizer, num_batches, checkpoint_path)
                print(f"Checkpoint saved at batch {num_batches}")

            if num_batches % 10 == 0:
                print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
                if num_batches % 100 == 0:
                    for _ in range(1):
                        model.eval()
                        sampled = tokenizer.detokenize(model.sample_continuation(tokenizer.tokenize("Hello"), 500))
                        model.train()
                        print(f"Model sample: '''{sampled}'''")
                    print("")
