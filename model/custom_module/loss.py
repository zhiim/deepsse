import math
from itertools import permutations

import torch
from torch import nn


class BCELoss(nn.BCELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
        use_sigmoid=False,
    ):
        """AsymmetricLoss implements an asymmetric loss function, which is
        useful for imbalanced classification problems.

        It has two separate parameters to control the weight of positive and
        negative classes (gamma_pos and gamma_neg).
        It also includes an option to clip the prediction values for the
        negative class to ignore easy negative classes.

        Args:
            gamma_neg (float, optional): The focusing parameter for the negative
                class, default is 4.
            gamma_pos (float, optional): The focusing parameter for the positive
                class, default is 1.
            clip (float, optional): The clipping parameter for the negative
                class, default is 0.05.
            eps (float, optional): A small constant to prevent log of zero,
                default is 1e-8.
            disable_torch_grad_focal_loss (bool, optional): A flag to disable
                the gradient computation for the focal loss part, default is
                True.

        References:
            Ridnik, Tal, Emanuel Ben-Baruch, Nadav Zamir, Asaf Noy, Itamar
            Friedman, Matan Protter, and Lihi Zelnik-Manor. “Asymmetric Loss for
            Multi-Label Classification,” 82–91, 2021.
        """
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.use_sigmoid = use_sigmoid

    def forward(self, x, y):
        """ "The forward method calculates the asymmetric loss for the given
        inputs.

        Args:
            x (torch.Tensor): The input logits.
            y (torch.Tensor): The targets (multi-label binarized vector).

        Returns:
            torch.Tensor: The calculated loss.
        """

        # Calculating Probabilities
        if self.use_sigmoid:
            x_sigmoid = torch.sigmoid(x)
        else:
            x_sigmoid = x
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping (probability shifting)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean(1).sum()


class RMSPELoss(nn.Module):
    """root mean squared periodic error used in DA-MUSIC.

    References:
        Merkofer, Julian P., Guy Revach, Nir Shlezinger, Tirza Routtenberg, and
        Ruud J. G. van Sloun. “DA-MUSIC: Data-Driven DoA Estimation via Deep
        Augmented MUSIC Algorithm.” arXiv, January 11, 2023.
        https://doi.org/10.48550/arXiv.2109.10581.
    """

    def __init__(self):
        """Initializes the RMSPE loss function.

        Args:
            for_da_music (bool, optional): A flag to indicate if the loss is
                used for DA-MUSIC, default is False.
        """
        super().__init__()

    def _permute_prediction(self, prediction: torch.Tensor):
        """
        Generates all the available permutations of the given prediction tensor.

        Args:
            prediction (torch.Tensor): The input tensor for which permutations
                are generated.

        Returns:
            torch.Tensor: A tensor containing all the permutations of the input
                tensor.

        Examples:
            >>> prediction = torch.tensor([1, 2, 3])
            >>>> permute_prediction(prediction)
                torch.tensor([[1, 2, 3],
                              [1, 3, 2],
                              [2, 1, 3],
                              [2, 3, 1],
                              [3, 1, 2],
                              [3, 2, 1]])

        """
        device = prediction.device
        torch_perm_list = []
        for p in list(
            permutations(range(prediction.shape[0]), prediction.shape[0])
        ):
            torch_perm_list.append(
                prediction.index_select(
                    0, torch.tensor(list(p), dtype=torch.int64).to(device)
                )
            )
        predictions = torch.stack(torch_perm_list, dim=0)
        return predictions

    def forward(self, preds, targs):
        device = preds.device
        batch_size = preds.size(0)

        rmspes = torch.zeros(batch_size, device=device)
        # because the number of signals in each sample may be different,
        # we need to iterate over the number of signals in each sample
        for i in range(batch_size):
            pred = preds[i]
            targ = targs[i]
            num_signal = int(targ[-1])
            pre_permutations = self._permute_prediction(pred[:num_signal])
            doa_diff = (
                (pre_permutations - targ[:num_signal] / 180 * math.pi)
                + math.pi / 2
            ) % math.pi - math.pi / 2
            rmspes[i] = (
                math.sqrt(1 / num_signal)
                * torch.linalg.norm(doa_diff.to(torch.float32), dim=1).min()
            )

        return torch.mean(rmspes)
