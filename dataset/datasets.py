from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset

from dataset.util import H5DataHandler, H5DataHandlerX


class BaseDataset(ABC, Dataset):
    @abstractmethod
    def __init__(self):
        """A custom base Dataset, load data from data path.

        Dataset must be loaded into self._data from file in __init__()
        """
        raise NotImplementedError()

    def __getitem__(self, index):
        data, label = self._get_data_label_from_dataset(index)

        data = self._data_processing(data)
        label = self._label_processing(label)

        return data, label

    @abstractmethod
    def __len__(self):
        # return length of this dataset
        raise NotImplementedError()

    @abstractmethod
    def _get_data_label_from_dataset(self, index):
        """Get data and label indexed by `index` from dataset

        Retruns:
            data: input data of network
            label: label
        """
        raise NotImplementedError()

    @abstractmethod
    def _data_processing(self, data):
        """Transform data into a form that matches the network input

        Should return a torch.tensor object
        """
        raise NotImplementedError()

    @abstractmethod
    def _label_processing(self, label):
        """Transform label into a form that matches the network output

        Should return a torch.tensor object
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        """Close the dataset"""
        raise NotImplementedError()


class CommonDataset(BaseDataset):
    """A Common Dataset for general usage."""

    def __init__(self, data_path, label_path, add_channel=False, lazy=True):
        if lazy:
            self._data_handler = H5DataHandler(data_path, label_path)
        else:
            self._data_handler = H5DataHandlerX(data_path, label_path)

        self._add_channel = add_channel

    def close(self):
        self._data_handler.close()

    def __len__(self):
        return len(self._data_handler)

    def _get_data_label_from_dataset(self, index):
        data, label = self._data_handler.get_data(index)

        return torch.tensor(data), torch.tensor(label)

    def _data_processing(self, data):
        if self._add_channel:
            return torch.unsqueeze(data, dim=0).to(torch.float32)
        return data.to(torch.float32)

    def _label_processing(self, label):
        return label


class SignalNumClassifierDataset(CommonDataset):
    def __init__(self, data_path, label_path, num_antennas):
        super().__init__(data_path, label_path)

        self._num_antennas = num_antennas

    def _data_processing(self, data):
        return data.to(torch.float32)

    def _label_processing(self, label):
        return label[-1].to(torch.int64)
