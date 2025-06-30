# from __future__ import annotations
# import torch
# from torch import nn
# import torch.nn.functional as F
#
# def batch_to_labeled_samples(batch: torch.IntTensor) -> [torch.IntTensor, torch.IntTensor]:
#     raise Exception("Not implemented.")
#     # TODO implement this.
#     # The batches that we get from the reader have corpus-sequences of length max-context + 1.
#     # We need to translate them to input/output examples, each of which is shorter by one.
#     # That is, if our input is of dimension (b x n) our output is two tensors, each of dimension (b x n-1)
#     inputs = batch[:,:] # TODO fix this
#     labels = batch[:,:] # TODO fix this
#     return (inputs, labels)
#
# def compute_loss(logits, gold_labels):
#     # logits size is (batch, seq_len, vocab_size)
#     # gold_bales size is (batch, seq_len)
#     # NOTE remember to handle padding (ignore them in loss calculation!)
#     # NOTE cross-entropy expects other dimensions for logits
#     # NOTE you can either use cross_entropy from PyTorch, or implement the loss on your own.
#     return ...




# lm.py

import torch
import torch.nn.functional as F

def batch_to_labeled_samples(batch):
    # batch: list of token lists, each of length seq_len + 1
    if isinstance(batch, torch.Tensor):
      batch = batch.detach().clone().long()
    else:
      batch = torch.tensor(batch, dtype=torch.long)
    batch_x = batch[:, :-1]  # input: all but last token
    batch_y = batch[:, 1:]   # target: all but first token
    return batch_x, batch_y

def compute_loss(logits, targets):
    # logits: (batch_size, seq_len, vocab_size)
    # targets: (batch_size, seq_len)
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    return F.cross_entropy(logits, targets)

