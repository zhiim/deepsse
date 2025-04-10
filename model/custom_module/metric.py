import numpy as np
from sklearn import metrics


def average_precision(target, output):
    epsilon = 1e-8

    indices = output.argsort()[::-1]
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]

    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)
    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    ap = []
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap_ = average_precision(targets, scores)
        # if ap_ is not None:
        #     ap.append(ap_)
        ap.append(ap_)

    return 100 * np.array(ap).mean()


def micro_ap(targs, preds):
    return metrics.average_precision_score(targs, preds, average="micro")


def f1_score(targs, preds):
    preds = (preds >= 0.5).astype(np.int32)
    return metrics.f1_score(targs, preds, average="macro", zero_division=np.nan)
