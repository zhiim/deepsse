import logging

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)


class H5DataHandlerX:
    """A helper class to load multiple datasets from h5 files.

    Args:
        data_path: a list of path of different datasets
        label_path: a list of path of different labels
    """

    def __init__(self, data_path, label_path):
        self._datas = []
        self._data_lens = []  # length of each dataset

        # load data from h5 file
        for dp in data_path:
            data_file = h5py.File(dp, "r")

            data = data_file["data"]
            if not isinstance(data, h5py.Dataset):
                raise ValueError("Dataset not found in data file.")
            self._datas.append(data[:])

            self._data_lens.append(data.len())

        # load labels from pt file
        self._labels = []
        for lp in label_path:
            label_file = h5py.File(lp, "r")

            label = label_file["labels"]
            if not isinstance(label, h5py.Dataset):
                raise ValueError("Dataset not found in data file.")

            self._labels.append(label[:])

    def __len__(self):
        """Return the total number of data samples."""
        return np.sum(self._data_lens)

    def close(self):
        """Close all the opened files."""
        pass

    def get_data(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the data and label of a specific index."""
        if index < 0 or index >= len(self):
            raise ValueError("Index out of range.")

        idx_sum = 0
        for i, dl in enumerate(self._data_lens):
            if index < idx_sum + dl:
                return self._datas[i][index - idx_sum], self._labels[i][
                    index - idx_sum
                ]

            idx_sum += dl

        logger.warning("Dataset index out of range.")
        return self._datas[0][0], self._labels[0][0]


class H5DataHandler:
    """A helper class to load multiple datasets from h5 files.

    Lazy loading, data is not loaded into memory. Useful for large datasets.

    Args:
        data_path: a list of path of different datasets
        label_path: a list of path of different labels
    """

    def __init__(self, data_path, label_path):
        self._data_files = []
        self._data_lens = []  # length of each dataset

        # load data from h5 file
        for dp in data_path:
            data_file = h5py.File(dp, "r")
            self._data_files.append(data_file)
            self._data_lens.append(len(data_file["data"]))  # type: ignore

        # load labels from pt file
        self._label_files = []
        for lp in label_path:
            self._label_files.append(h5py.File(lp, "r"))

    def __len__(self):
        """Return the total number of data samples."""
        return np.sum(self._data_lens)

    def close(self):
        """Close all the opened files."""
        for df in self._data_files:
            df.close()
        for lf in self._label_files:
            lf.close()

    def get_data(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the data and label of a specific index."""
        if index < 0 or index >= len(self):
            raise ValueError("Index out of range.")

        idx_sum = 0
        for i, dl in enumerate(self._data_lens):
            if index < idx_sum + dl:
                return self._data_files[i]["data"][
                    index - idx_sum
                ], self._label_files[i]["labels"][index - idx_sum]

            idx_sum += dl

        logger.warning("Dataset index out of range.")
        return self._data_files[0]["data"][0], self._label_files[0]["labels"][0]


def z_score_norm(data, is_complex=False):
    """Z score normalization"""
    if is_complex:
        data_abs = torch.abs(data)
        data_abs = (data_abs - torch.mean(data_abs)) / torch.std(data_abs)
        data = data_abs * torch.exp(1j * torch.angle(data))
        return data
    return (data - torch.mean(data)) / torch.std(data)
