from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class MetricTracker:
    def __init__(self, *keys):
        """A Tracker used to record training data, such as loss and performance
        metric

        Args:
            *keys: metrics to track when training model. eg. val_loss
        """
        # build a talbe with keys as index
        self._data = pd.DataFrame(
            index=pd.Index(keys),
            columns=pd.Index(["total", "counts", "average"]),
        )
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data.loc[:, col] = 0

    def update(self, key, value, n=1):
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = (
            self._data.loc[key, "total"] / self._data.loc[key, "counts"]
        )

    def avg(self, key):
        return self._data.loc[key, "average"]

    def result(self):
        return dict(self._data.average)


class Result:
    def __init__(
        self,
        doa_result: np.ndarray[Any, np.dtype[np.float64]],
        result_type: Literal["spectrum", "angle"] = "spectrum",
        normalize: bool = False,
    ):
        """A class to store the result of DOA algorithm.

        Args:
            doa_result (np.ndarray[Any, np.dtype[np.float64]]): The result array
                from the DOA algorithm.
            result_type (Literal["spectrum", "angle"], optional): The type of
                result, either "spectrum" or "angle". Defaults to "spectrum".
        """
        self._doa_result = doa_result
        if normalize and result_type == "spectrum":
            res_min = self._doa_result.min()
            res_max = self._doa_result.max()
            self._doa_result = (self._doa_result - res_min) / (
                res_max - res_min
            )
        self._result_type = result_type

    @property
    def type(self):
        """The type of the result ("spectrum" or "angle")."""
        return self._result_type

    @property
    def result(self) -> dict[str, Any]:
        """The result as a dictionary.

        Returns:
            dict["type": Any, "result": Any]: A dictionary containing the result
                type and the DOA result array.
        """
        return {"type": self._result_type, "result": self._doa_result}

    def __str__(self):
        return str(self.result)

    def angle_result(self, num_signal: int = 1, peak_threshold: float = 0.0):
        """Get the angle result.

        Args:
            num_signal (int, optional): The number of signals to consider.
                Defaults to 1.
            height (float, optional): The height threshold for peak detection.
                Defaults to 0.0.

        Returns:
            dict[str, Any]: A dictionary containing the angles and heights. If
                result_type is "angle", all "heights" is set to 1.0.
        """
        if self._result_type == "angle":
            return {
                "angles": self._doa_result,
                "heights": np.ones(num_signal) * 1.0,
            }
        else:
            # find peaks and peak heights
            peaks_idx, heights = find_peaks(
                self._doa_result, height=peak_threshold, distance=1
            )
            idx = heights["peak_heights"].argsort()[-num_signal:]
            peaks_idx = peaks_idx[idx]
            heights = heights["peak_heights"][idx]

            idx_offset = len(self._doa_result) // 2
            result = {"angles": peaks_idx - idx_offset, "heights": heights}

        return result
