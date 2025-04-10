from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Base class for all models"""

    @abstractmethod
    def forward(self, x):
        """Forward pass logic

        Args:
            model_input: model input

        Returns:
            model output
        """
        raise NotImplementedError()

    def __str__(self):
        """Model prints with number of trainable parameters"""
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())

        return (
            super().__str__()
            + "\nParameters: {}".format(
                sum([np.prod(p.size()) for p in trainable_params])
            )
            + "\nTrainable parameters: {}".format(
                sum([np.prod(p.size()) for p in self.parameters()])
            )
        )


class BaseModule(ABC, nn.Module):
    def __str__(self):
        """Model prints with number of trainable parameters"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    @classmethod
    def build_model(cls, **kwargs):
        params = {
            k: v
            for k, v in kwargs.items()
            if k in cls.__init__.__code__.co_varnames
        }
        return cls(**params)
