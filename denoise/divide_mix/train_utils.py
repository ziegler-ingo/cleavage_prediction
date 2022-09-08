"""
Customised implementation of the original paper:
    DivideMix: Learning With Noisy Labels as Semi-Supervised Learning
    https://openreview.net/pdf?id=HJgExaVtwr
    https://github.com/LiJunnan1992/DivideMix
"""


import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class SemiLoss:
    def __call__(self, out_x, lbl_x, out_u, lbl_u, lambda_u, epoch, warm_up, rampup_len):
        probs_u = torch.softmax(out_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(out_x, dim=1) * lbl_x, dim=1))
        Lu = torch.mean((probs_u - lbl_u) ** 2)
        return Lx, Lu, linear_rampup(lambda_u, epoch, warm_up, rampup_len)


class NegEntropy:
    def __call__(self, out):
        probs = torch.softmax(out, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def warmup(model, optimizer, loss_func, conf_penalty, dataloader):
    epoch_loss, num_correct, total = 0, 0, 0
    preds, lbls = [], []
    model.train()

    for batch in tqdm(dataloader, desc="warmup: ", file=sys.stdout, unit="batches"):
        in_x, lbl_x = batch.seq, batch.lbl
        in_x, lbl_x = in_x.to(device), lbl_x.to(device)

        out = model(in_x)
        # CrossEntropyLoss
        loss = loss_func(out, lbl_x)

        # penalty for confident predictions for asymmetric noise
        loss += conf_penalty(out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        non_ohe_labels = lbl_x.argmax(dim=1)
        num_correct += (out.argmax(dim=1) == non_ohe_labels).sum().item()
        total += in_x.shape[0]
        preds.extend(out[:, 1].detach().tolist())
        lbls.extend(non_ohe_labels.detach().tolist())
    return epoch_loss / total, num_correct / total, roc_auc_score(lbls, preds)


def train(
    epoch,
    model1,
    model2,
    optimizer,
    loss_func,
    num_warm_up_epochs,
    num_classes,
    lambda_u,
    temp,
    beta_dist,
    batch_size,
    labeled_loader,
    unlabeled_loader,
    rampup_len,
    named_model,
):
    epoch_loss, total = 0, 0

    # fix one model while the other one trains
    model1.train()
    model2.eval()

    unlabeled_train_iter = iter(unlabeled_loader)
    num_iter = (len(labeled_loader.dataset) // batch_size) + 1

    for batch_idx, batch in enumerate(
        tqdm(
            labeled_loader,
            desc=f"train {named_model}: ",
            file=sys.stdout,
            unit="batches",
        )
    ):
        in_x, in_x2, labels_x, w_x = batch.seq1, batch.seq2, batch.lbl, batch.prob

        # re-use unlabeled_loader as long as there are batches in labeled_loader
        try:
            unlabeled_batch = unlabeled_train_iter.next()
            in_u, in_u2 = unlabeled_batch.seq1, unlabeled_batch.seq2
        except:
            unlabeled_train_iter = iter(unlabeled_loader)
            unlabeled_batch = unlabeled_train_iter.next()
            in_u, in_u2 = unlabeled_batch.seq1, unlabeled_batch.seq2

        w_x = w_x.view(-1, 1).to(torch.float)
        in_x, in_x2, labels_x, w_x = (
            in_x.to(device),
            in_x2.to(device),
            labels_x.to(device),
            w_x.to(device),
        )

        in_u, in_u2 = in_u.to(device), in_u2.to(device)

        with torch.no_grad():
            # for labeled samples: co-refinement + temperature sharpening
            out_x = model1(in_x)
            out_x2 = model1(in_x2)

            px = (torch.softmax(out_x, dim=1) + torch.softmax(out_x2, dim=1)) / 2
            px = w_x * labels_x + (1 - w_x) * px

            ptx = px ** (1 / temp)
            targets_x = (ptx / ptx.sum(dim=1, keepdim=True)).detach()

            # for unlabeled samples: co-guessing + temperature sharpening
            out_u11 = model1(in_u)
            out_u12 = model1(in_u2)
            out_u21 = model2(in_u)
            out_u22 = model2(in_u2)

            pu = (
                torch.softmax(out_u11, dim=1)
                + torch.softmax(out_u12, dim=1)
                + torch.softmax(out_u21, dim=1)
                + torch.softmax(out_u22, dim=1)
            ) / 4

            ptu = pu ** (1 / temp)
            targets_u = (ptu / ptu.sum(dim=1, keepdim=True)).detach()

        ### MixMatch
        # lambda interpolation factor for mixed input and targets
        lam = beta_dist.sample()
        lam = max(lam, 1 - lam)

        all_ins = torch.cat([in_x, in_x2, in_u, in_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        # shuffle all inputs to generate new pairs
        idx = torch.randperm(all_ins.shape[0])

        in_a, in_b = all_ins, all_ins[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_target = lam * target_a + (1 - lam) * target_b
        # inputs are mixed up in forward pass
        logits = model1(in_a, in_b, lam, interpolate=True)
        logits_x, logits_u = logits[: batch_size * 2], logits[batch_size * 2 :]

        # custom SemiLoss
        Lx, Lu, lam_u = loss_func(
            out_x=logits_x,
            lbl_x=mixed_target[: batch_size * 2],
            out_u=logits_u,
            lbl_u=mixed_target[batch_size * 2 :],
            lambda_u=lambda_u,
            epoch=epoch + batch_idx / num_iter,
            warm_up=num_warm_up_epochs,
            rampup_len=rampup_len,
        )

        # regularization
        prior = (torch.ones(num_classes) / num_classes).to(device)
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lam_u * Lu + penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        total += idx.shape[0]
    return epoch_loss / total


def process_gmm(model, dataloader, loss_func):
    losses = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc="GMM processing: ", file=sys.stdout, unit="batches"
        ):
            in_x, lbl_x = batch.seq, batch.lbl
            in_x, lbl_x = in_x.to(device), lbl_x.to(device)

            out = model(in_x)
            # nn.CrossEntropyLoss with reduction=None, returns per sample loss
            loss = loss_func(out, lbl_x)
            losses.extend(loss.detach().tolist())

    losses = np.array(losses)
    # normalize losses between 0 and 1
    norm_losses = ((losses - losses.min()) / losses.ptp())[:, np.newaxis]

    # fit two component GMM to loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=1234)
    gmm.fit(norm_losses)
    prob = gmm.predict_proba(norm_losses)
    # get value of smaller mean dist
    prob = prob[:, gmm.means_.argmin()]
    # out shape: (batch_size)
    return prob, losses.mean(), losses, norm_losses


def evaluate(model1, model2, loss_func, dataloader):
    num_correct, total = 0, 0
    preds, lbls, losses = [], [], []
    model1.eval()
    model2.eval()

    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc="evaluate: ", file=sys.stdout, unit="batches"
        ):
            in_x, lbl_x = batch.seq, batch.lbl
            in_x, lbl_x = in_x.to(device), lbl_x.to(device)
            out1 = model1(in_x)
            out2 = model2(in_x)
            out = out1 + out2

            # nn.CrossEntropyLoss with reduction=None, returns per sample loss
            loss = loss_func(out, lbl_x)
            losses.extend(loss.detach().tolist())

            non_ohe_labels = lbl_x.argmax(dim=1)
            num_correct += (out.argmax(dim=1) == non_ohe_labels).sum().item()
            total += in_x.shape[0]
            preds.extend(out[:, 1].detach().tolist())
            lbls.extend(non_ohe_labels.detach().tolist())

    losses = np.array(losses)
    return num_correct / total, roc_auc_score(lbls, preds), losses.mean()


def linear_rampup(lambda_u, current_epoch, warm_up, rampup_len):
    current = np.clip((current_epoch - warm_up) / rampup_len, 0.0, 1.0)
    return lambda_u * current


