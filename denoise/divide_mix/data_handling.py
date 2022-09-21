"""
Customised implementation of the original paper:
    DivideMix: Learning With Noisy Labels as Semi-Supervised Learning
    https://openreview.net/pdf?id=HJgExaVtwr
    https://github.com/LiJunnan1992/DivideMix
"""


import csv
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# load vocab and encoder as globals
vocab = torch.load('../../data/vocab.pt')
encode_text = lambda x: vocab(list(x))


def read_data(path):
    with open(path, 'r') as csvfile:
        train_data = list(csv.reader(csvfile))[1:] # skip col name
        sents, lbls = [], []
        for s, l in train_data:
            sents.append(s)
            lbls.append(l)
    return sents, lbls

def apply_random_masking(seq, num_tokens):
    """
    Mask `num_tokens` as 0 at random positions per sequence. 
    """
    dist = torch.rand(seq.shape)
    m, _ = torch.topk(dist, num_tokens)
    return seq * (dist < m)


class CleavageDataset(Dataset):
    def __init__(
        self,
        terminus,
        mode,
        pred=None,
        probability=None,
    ):

        self.mode = mode

        if self.mode == "test":
            self.seq, self.lbl = read_data(f"../../data/{terminus}_test.csv")
        elif self.mode == 'evaluate':
            self.seq, self.lbl = read_data(f"../../data/{terminus}_val.csv")
        else:
            sequences, labels = read_data(f"../../data/{terminus}_train.csv")

            if self.mode == "train_before_GMM":
                self.seq, self.lbl = sequences, labels
            elif self.mode == "labeled" and pred is not None and probability is not None:
                # get indices of labels considered 'clean' by GMM
                self.prob = probability[pred]
                self.seq, self.lbl = np.array(sequences)[pred], np.array(labels)[pred]
 
            elif self.mode == "unlabeled" and pred is not None:
                # get indices of labels considered 'noisy' by GMM
                self.seq = np.array(sequences)[~pred] 
            else:
                raise Exception(
                    "Either args `pred` and/or `probability` need to be given when loading `mode`='train'."
                )

    def __getitem__(self, idx):
        if self.mode == "labeled":
            return self.seq[idx], self.lbl[idx], self.prob[idx]

        elif self.mode == "unlabeled":
            return self.seq[idx]

        else: 
            return self.seq[idx], self.lbl[idx] 

    def __len__(self):
        return len(self.seq)


class MaskedNoGMMBatch:
    def __init__(self, batch):
        ordered_batch = list(zip(*batch))
        raw_seq = torch.tensor(
            [encode_text(seq) for seq in ordered_batch[0]], dtype=torch.int64
        )
        self.seq = apply_random_masking(raw_seq, 1)
        lbl = torch.tensor([int(l) for l in ordered_batch[1]], dtype=torch.long)
        self.lbl = F.one_hot(lbl, num_classes=2).to(torch.float)

    def pin_memory(self):
        self.seq = self.seq.pin_memory()
        self.lbl = self.lbl.pin_memory()
        return self


def masked_no_gmm_wrapper(batch):
    return MaskedNoGMMBatch(batch)


class LabeledMaskedBatch:
    def __init__(self, batch):
        ordered_batch = list(zip(*batch))
        raw_seq = torch.tensor(
            [encode_text(seq) for seq in ordered_batch[0]], dtype=torch.int64
        )
        self.seq1 = apply_random_masking(raw_seq, 1)
        self.seq2 = apply_random_masking(raw_seq, 1)
        lbl = torch.tensor([int(l) for l in ordered_batch[1]], dtype=torch.long)
        self.lbl = F.one_hot(lbl, num_classes=2).to(torch.float)
        self.prob = torch.tensor(ordered_batch[2], dtype=torch.float)

    def pin_memory(self):
        self.seq1 = self.seq1.pin_memory()
        self.seq2 = self.seq2.pin_memory()
        self.lbl = self.lbl.pin_memory()
        self.prob = self.prob.pin_memory()
        return self


def labeled_masked_wrapper(batch):
    return LabeledMaskedBatch(batch)


class UnlabeledMaskedBatch:
    def __init__(self, batch):
        # ordered_batch = list(zip(*batch))
        raw_seq = torch.tensor(
            [encode_text(seq) for seq in batch], dtype=torch.int64
        )
        self.seq1 = apply_random_masking(raw_seq, 1)
        self.seq2 = apply_random_masking(raw_seq, 1)

    def pin_memory(self):
        self.seq1 = self.seq1.pin_memory()
        self.seq2 = self.seq2.pin_memory()
        return self


def unlabeled_masked_wrapper(batch):
    return UnlabeledMaskedBatch(batch)


class EvalBatch:
    def __init__(self, batch):
        ordered_batch = list(zip(*batch))
        self.seq = torch.tensor(
            [encode_text(seq) for seq in ordered_batch[0]], dtype=torch.int64
        )
        lbl = torch.tensor([int(l) for l in ordered_batch[1]], dtype=torch.long)
        self.lbl = F.one_hot(lbl, num_classes=2).to(torch.float)

    def pin_memory(self):
        self.seq = self.seq.pin_memory()
        self.lbl = self.lbl.pin_memory()
        return self


def eval_wrapper(batch):
    return EvalBatch(batch)


class CleavageLoader:
    def __init__(
        self,
        batch_size,
        num_workers,
    ):

        self.batch_size = batch_size
        self.num_workers = num_workers

    def load(self, terminus, mode, pred=None, prob=None):
        if mode == "warmup":
            train_no_gmm = CleavageDataset(
                terminus=terminus,
                mode="train_before_GMM",
            )

            warmup_loader = DataLoader(
                dataset=train_no_gmm,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=masked_no_gmm_wrapper,
                pin_memory=True,
                num_workers=self.num_workers,
            )
            return warmup_loader

        elif mode == "train":
            labeled_dataset = CleavageDataset(
                terminus=terminus, mode="labeled", pred=pred, probability=prob
            )

            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=labeled_masked_wrapper,
                pin_memory=True,
                num_workers=self.num_workers,
            )

            unlabeled_dataset = CleavageDataset(
                terminus=terminus,
                mode="unlabeled",
                pred=pred,
            )

            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=unlabeled_masked_wrapper,
                pin_memory=True,
                num_workers=self.num_workers,
            )
            return labeled_trainloader, unlabeled_trainloader

        elif mode == "divide_by_GMM":
            into_GMM_dataset = CleavageDataset(
                terminus=terminus,
                mode="train_before_GMM",
            )

            into_GMM_loader = DataLoader(
                dataset=into_GMM_dataset,
                batch_size=self.batch_size,
                collate_fn=eval_wrapper,
                pin_memory=True,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return into_GMM_loader

        elif mode == "evaluate":
            eval_dataset = CleavageDataset(
                terminus=terminus,
                mode="evaluate",
            )

            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                collate_fn=eval_wrapper,
                pin_memory=True,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return eval_loader

        elif mode == "test":
            test_dataset = CleavageDataset(
                terminus=terminus,
                mode="test",
            )

            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                collate_fn=eval_wrapper,
                pin_memory=True,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return test_loader
