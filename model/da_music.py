import math

import torch
import torch.nn as nn

from model.base_model import BaseModel
from utils.util import get_device


class DAMUSICClassifier(BaseModel):
    def __init__(
        self,
        num_antennas,
        da_music_weight_path,
        antenna_spacing=0.5,
        grid=1,
    ):
        super().__init__()

        self._da_music = DeepAugmentMusic(
            num_antennas=num_antennas,
            antenna_spacing=antenna_spacing,
            grid=grid,
        ).to(get_device())
        self._da_music.load_state_dict(
            torch.load(
                da_music_weight_path,
                map_location=get_device(),
                weights_only=True,
            )
        )
        self._da_music.eval()

        self.signal_num_estimator = nn.Sequential(
            nn.Linear(
                in_features=2 * num_antennas,
                out_features=2 * num_antennas * num_antennas,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=2 * num_antennas * num_antennas,
                out_features=2 * num_antennas * num_antennas,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=2 * num_antennas * num_antennas,
                out_features=2 * num_antennas * num_antennas,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=2 * num_antennas * num_antennas,
                out_features=num_antennas - 1,
            ),
        )

    def forward(self, x):
        eig_val = self._da_music.get_eig_val(x)
        num_signal = self.signal_num_estimator(
            torch.concatenate((eig_val.real, eig_val.imag), dim=1)
        )

        return num_signal


class DeepAugmentMusic(BaseModel):
    def __init__(self, num_antennas, antenna_spacing, grid):
        """
        Reference:
            Merkofer, Julian P., Guy Revach, Nir Shlezinger, Tirza Routtenberg,
            and Ruud J. G. van Sloun. “DA-MUSIC: Data-Driven DoA Estimation via
            Deep Augmented MUSIC Algorithm.” arXiv, January 11, 2023.
            https://doi.org/10.48550/arXiv.2109.10581.
        """
        super().__init__()

        self._num_antennas = num_antennas
        self._angle_grids = torch.arange(-90, 90, grid)

        # ━━ 1. get covariance matrix ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.norm = nn.BatchNorm1d(2 * num_antennas)  # norm in feature
        self.gru = nn.GRU(
            input_size=2 * num_antennas, hidden_size=2 * num_antennas
        )
        self.linear = nn.Linear(
            in_features=2 * num_antennas,
            out_features=2 * num_antennas * num_antennas,
        )

        # ━━ 2. get the weight of eigenvectors ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.eig_vec_pro = nn.Sequential(
            nn.Linear(
                in_features=2 * num_antennas, out_features=2 * num_antennas
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=2 * num_antennas, out_features=2 * num_antennas
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=2 * num_antennas, out_features=2 * num_antennas
            ),
            nn.ReLU(),
            nn.Linear(in_features=2 * num_antennas, out_features=num_antennas),
            nn.Sigmoid(),
        )

        # ━━ 3. peak finder ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.peak_finder = nn.Sequential(
            nn.Linear(
                in_features=self._angle_grids.shape[0],
                out_features=2 * num_antennas,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=2 * num_antennas, out_features=2 * num_antennas
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=2 * num_antennas, out_features=2 * num_antennas
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=2 * num_antennas, out_features=num_antennas - 1
            ),
        )

        self._get_steering_vectors(num_antennas, antenna_spacing)

    def _get_steering_vectors(self, num_antenna, antenna_spacing):
        antenna_position = (
            (torch.arange(0, num_antenna, 1) * antenna_spacing)
            .view(-1, 1)
            .to(torch.float)
        )
        delay = antenna_position @ torch.sin(self._angle_grids).view(1, -1)

        steering_vectors = torch.exp(-2j * math.pi * delay)

        self.register_buffer("steering_vectors", steering_vectors)

    def _get_cov(self, x):
        # x: (batch_size, 2 * num_antennas, num_snapshots)
        # `2 * num_antennas` is the number of feature
        x = self.norm(x)  # norm in feature
        x = torch.permute(
            x, (2, 0, 1)
        )  # (num_snapshots, batch_size, 2 * num_antennas)
        _, x = self.gru(x)  # (batch_size, 1, 2 * num_antennas)
        x = self.linear(x)  # (batch_size, 1, 2 * num_antennas * num_antennas)
        x = x.reshape(-1, 2 * self._num_antennas, self._num_antennas)

        x = torch.complex(
            x[:, : self._num_antennas, :], x[:, self._num_antennas :, :]
        )

        return x

    def _get_noise_space(self, eig_val, eig_vec):
        prob = self.eig_vec_pro(torch.cat((eig_val.real, eig_val.imag), dim=1))
        prob = torch.diag_embed(prob)

        # NOTE: the `eig_vec` is parted into real and imaginary parts to prevent
        #       the gradient from being calculated on the phase of the complex
        # see this for reason: https://pytorch.org/docs/stable/generated/torch.svd.html
        noise_space = torch.complex(
            torch.bmm(prob, eig_vec.real), torch.bmm(prob, eig_vec.imag)
        )

        return noise_space

    def _cal_spectrum(self, noise_space):
        device = noise_space.device
        v = noise_space.transpose(1, 2).conj() @ self.steering_vectors.to(
            device
        )

        spectrum = 1 / torch.linalg.norm(v, axis=1) ** 2

        return spectrum.to(torch.float32)

    def get_eig_val(self, x):
        cov = self._get_cov(x)
        eig_val, _ = torch.linalg.eig(cov)

        return eig_val

    def forward(self, x):
        cov = self._get_cov(x)

        eig_val, eig_vec = torch.linalg.eig(cov)

        noise_space = self._get_noise_space(eig_val, eig_vec)

        spectrum = self._cal_spectrum(noise_space)

        estimates = self.peak_finder(spectrum)

        return estimates
