import numpy as np
import torch

from utils.util import to_torch


def prepare_for_inf(data):
    data = data[np.newaxis]
    data = to_torch(data).to(torch.float32)
    return data


def default(data, inference=False):
    "do not process data"
    if inference:
        data = prepare_for_inf(data.astype(np.float32))
    return data


# -- Dataset Processing ------------------------------------------------
# Functions for processing data in dataset generation
# ----------------------------------------------------------------------


def cal_cov(data, use_phase=False, inference=False):
    """Calculate covariance matrix of data"""
    cov = np.cov(data)
    # normalize
    cov = cov / np.linalg.norm(cov)

    if use_phase:
        cov = np.concatenate(
            (
                np.real(cov)[np.newaxis],
                np.imag(cov)[np.newaxis],
                np.angle(cov)[np.newaxis],
            ),
            axis=0,
        ).astype(np.float32)
    else:
        cov = np.concatenate(
            (np.real(cov)[np.newaxis], np.imag(cov)[np.newaxis]), axis=0
        ).astype(np.float32)

    if inference:
        cov = prepare_for_inf(cov)

    return cov


def cal_da_music(data, use_phase=False, inference=False):
    if use_phase:
        data = np.concatenate(
            (np.real(data), np.imag(data), np.angle(data)), axis=0
        ).astype(np.float32)
    else:
        data = np.concatenate((np.real(data), np.imag(data)), axis=0)

    if inference:
        data = prepare_for_inf(data)
    return data


# -- Label Processing --------------------------------------------------
# Functions for processing label in dataset generation
# ----------------------------------------------------------------------


def cal_multi_hot(data, lowwer_bound=-90, upper_bound=90, num_classes=180):
    """Calculate multi-hot encoding of data"""
    idx = data - lowwer_bound / (upper_bound - lowwer_bound) * num_classes
    idx = idx.astype(np.int32)

    multi_hot = np.zeros(num_classes, dtype=np.int32)

    multi_hot[idx] = 1

    return multi_hot


def cal_regression_label(data, num_max_signal=4):
    """Calculate the DA-MUSIC label of the input data."""
    num_signal = data.size
    data = np.append(data, np.zeros(num_max_signal + 1 - num_signal))
    data[-1] = num_signal

    return data


# -- Ouput Processing --------------------------------------------------
# Functions for processing output in model inference
# ----------------------------------------------------------------------


def cal_class_prob(data):
    return torch.sigmoid(data).detach().numpy()


def rad_2_deg(data):
    return np.rad2deg(data)
