import itertools as it
import random
from abc import ABC, abstractmethod

import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import utils.doa_py.doa_py.arrays as array_module
import utils.doa_py.doa_py.signals as signal_module
from utils.util import init_object

random.seed(3407)


class BaseDataGenerator(ABC):
    def __init__(self, config, rng):
        """Generator used to generate DOA Estimation Dataset

        Args:
            config: a dict read from json config file which contains
                configuration of dataset generation
            rng (np.random.Generator): random generator used to generator random
                sequence
        """
        # create an array from config
        self._array = init_object(array_module, config["array"])
        self._array.set_rng(rng)
        # create an signal from config
        self._signal = init_object(signal_module, config["signal"])
        self._signal.set_rng(rng)

        # repeatly generate signals in one fixed condition
        self._doas = self._get_doas(config["doas"])
        self._unit = config["doas"]["unit"]

    def one_data(self, snr, nsmaples, doa=None, data_process=None):
        if doa is None:
            doa = random.choice(self._doas)

        data = self._array.received_signal(
            signal=self._signal,
            snr=snr,
            nsmaples=nsmaples,
            angle_incidence=doa,
            unit=self._unit,
        )
        if data_process is not None:
            data = data_process(data)

        return data, doa

    def _get_data_shape(self, data, data_process):
        if data_process is not None:
            data = data_process(data)
        return data.shape

    def _get_doas_chunk(self, doas, save_threthod):
        if save_threthod is None or save_threthod > len(doas):
            yield doas
        else:
            for i in range(0, len(doas), save_threthod):
                yield doas[i : i + save_threthod]

    def _gen_data(self, snr, nsamples, doa, data_process):
        data = self._array.received_signal(
            signal=self._signal,
            snr=snr,
            nsamples=nsamples,
            angle_incidence=doa,
            unit=self._unit,
        )
        if data_process is not None:
            data = data_process(data)

        return data

    def _gen_label(self, doa, label_process):
        label = doa
        if label_process is not None:
            label = label_process(doa)

        return label

    def generate(
        self,
        data_path,
        label_path,
        num_repeat,
        snrs,
        nsamples,
        data_process=None,
        label_process=None,
        num_data=None,
        save_threthod=None,
        parallelism=False,
        n_jobs=-1,
        paral_mode="threads",
    ):
        """Generate Dataset

        Args:
            data_path: save path of dataset
            label_path: save path of label
            data_process (function, optional): Function used to processing
                generated array signal before saved into dataset. Defaults to
                None.
            label_process (function, optional): Function used to processing
                labels. Defaults to None.
            num_data (_type_, optional): Limit the maximun number of data in
                Dataset. Defaults to None.
        """
        # 计算数据集的数据条数
        total_size = len(snrs) * len(self._doas) * num_repeat
        if num_data is not None:
            total_size = min(total_size, num_data)

        data_file = h5py.File(data_path, "w")
        # 生成一个信号用于确定datset的维度
        data_shape = self._get_data_shape(
            self._array.received_signal(
                signal=self._signal,
                snr=snrs[0],
                nsamples=nsamples,
                angle_incidence=self._doas[0],
                unit=self._unit,
            ),
            data_process,
        )
        data = data_file.create_dataset(
            "data",
            (total_size, *data_shape),
        )

        label_file = h5py.File(label_path, "w")
        label_shape = self._get_data_shape(self._doas[0], label_process)
        labels = label_file.create_dataset(
            "labels",
            (total_size, *label_shape),
        )

        num_data_snr = num_data // len(snrs) if num_data is not None else None
        doas = self._doas * num_repeat
        if num_data_snr is None or num_data_snr > len(doas):
            num_limited = False
            doa_chunks = list(self._get_doas_chunk(doas, save_threthod))
        else:
            num_data_orphan = num_data - num_data_snr * len(snrs)
            num_limited = True

        print(
            "Generating config:\n",
            "snrs: {}\n".format(snrs),
            "doas in every snr: {}\n".format(len(self._doas)),
            "number of signal in every doa: {}\n".format(num_repeat),
            "number of data in this dataset: {}\n".format(total_size),
            "--------------------------------------",
        )

        current_idx = 0

        for snr_count, snr in enumerate(
            tqdm(snrs, "Generating array signal dataset")
        ):
            if num_limited:
                if not isinstance(num_data_snr, int):
                    raise ValueError(
                        "num_data_snr must be an integer, but got {}".format(
                            type(num_data_snr)
                        )
                    )
                doa_chunks = list(
                    self._get_doas_chunk(
                        random.sample(
                            doas,
                            num_data_snr + num_data_orphan
                            if snr_count == len(snrs) - 1
                            else num_data_snr,
                        ),
                        save_threthod,
                    ),
                )

            for i, doa_chunk in enumerate(
                tqdm(
                    doa_chunks,
                    "Generating signal at {} dB".format(snr),
                    leave=False,
                )
            ):
                data_ = []
                labels_ = []

                tqdm.write("Generating signal data in chunk {}".format(i + 1))
                if parallelism:
                    try:
                        with Parallel(
                            n_jobs=n_jobs, prefer=paral_mode
                        ) as parallel:
                            data_ = parallel(
                                delayed(self._gen_data)(
                                    snr, nsamples, doa, data_process
                                )
                                for doa in tqdm(doa_chunk, leave=False)
                            )
                    except Exception as e:
                        print(f"Parallel processing error: {e}")
                        return
                else:
                    data_ = [
                        self._gen_data(snr, nsamples, doa, data_process)
                        for doa in tqdm(doa_chunk, leave=False)
                    ]

                tqdm.write("Generating labels in chunk {}".format(i + 1))
                if parallelism:
                    try:
                        with Parallel(
                            n_jobs=n_jobs, prefer=paral_mode
                        ) as parallel:
                            labels_ = parallel(
                                delayed(self._gen_label)(doa, label_process)
                                for doa in tqdm(doa_chunk, leave=False)
                            )
                    except Exception as e:
                        print(f"Parallel processing error: {e}")
                        return
                else:
                    labels_ = [
                        self._gen_label(doa, label_process)
                        for doa in tqdm(doa_chunk, leave=False)
                    ]

                tqdm.write(
                    "data generating at this chunk is finished, saving ..."
                )

                if not (isinstance(data_, list) and isinstance(labels_, list)):
                    raise TypeError("data_ and labels_ must be list")

                data[current_idx : current_idx + len(data_)] = data_
                labels[current_idx : current_idx + len(labels_)] = labels_

                current_idx += len(data_)

            # saving at every snr
            tqdm.write("data generating at {} dB is finished.".format(snr))
            tqdm.write("--------------------------------------")

        data_file.close()
        label_file.close()

    @abstractmethod
    def _get_doas(self, doa_config):
        raise NotImplementedError()


class SingleSignalDataGenerator(BaseDataGenerator):
    def _get_doas(self, doa_config):
        # consider azimuth only
        if doa_config["elevation"] is False:
            azimuths = np.arange(
                start=doa_config["azimuth"]["range"][0],
                stop=doa_config["azimuth"]["range"][1],
                step=doa_config["azimuth"]["grid"],
            )

            azimuths = np.reshape(azimuths, (len(azimuths), 1))
            doas = azimuths
        else:
            azimuths = np.arange(
                start=doa_config["azimuth"]["range"][0],
                stop=doa_config["azimuth"]["range"][1],
                step=doa_config["azimuth"]["grid"],
            )
            elevations = np.arange(
                start=doa_config["elevation"]["range"][0],
                stop=doa_config["elevation"]["range"][1],
                step=doa_config["elevation"]["grid"],
            )

            azimuths, elevations = np.meshgrid(azimuths, elevations)
            azimuths = azimuths.reshape(-1, 1)
            elevations = elevations.reshape(-1, 1)
            doas = np.concatenate((azimuths, elevations), axis=1)
            doas = doas.reshape(-1, 2, 1)

        return list(doas)


class FixedMultiSignalDataGenerator(BaseDataGenerator):
    def _get_doas(self, doa_config):
        num_signal = doa_config["num_signal"]
        if doa_config["elevation"] is False:
            azimuths = np.arange(
                doa_config["azimuth"]["range"][0],
                doa_config["azimuth"]["range"][1],
                doa_config["azimuth"]["grid"],
            )

            combined = it.combinations(azimuths, num_signal)
            combined = list(np.array(item) for item in combined)

            doas = combined

        else:
            azimuths = np.arange(
                doa_config["azimuth"]["range"][0],
                doa_config["azimuth"]["range"][1],
                doa_config["azimuth"]["grid"],
            )
            elevations = np.arange(
                doa_config["elevation"]["range"][0],
                doa_config["elevation"]["range"][1],
                doa_config["elevation"]["grid"],
            )

            azimuths, elevations = np.meshgrid(azimuths, elevations)
            azimuths = azimuths.reshape(-1, 1)
            elevations = elevations.reshape(-1, 1)
            azimuth_elevation = np.concatenate((azimuths, elevations), axis=1)
            azimuth_elevation = azimuth_elevation.reshape(-1, 2, 1)

            combined = it.combinations(azimuth_elevation, num_signal)
            combined = list(np.concatenate(item, axis=1) for item in combined)

            doas = combined

        return doas


class MixedMultiSignalDataGenerator(BaseDataGenerator):
    def _get_doas(self, doa_config):
        min_num_signal = doa_config["min_num_signal"]
        max_num_signal = doa_config["max_num_signal"]
        step_num_signal = doa_config["step_num_signal"]

        doas = []
        for num_signal in range(
            min_num_signal, max_num_signal + 1, step_num_signal
        ):
            if doa_config["elevation"] is False:
                azimuths = np.arange(
                    doa_config["azimuth"]["range"][0],
                    doa_config["azimuth"]["range"][1],
                    doa_config["azimuth"]["grid"],
                )

                combined = it.combinations(azimuths, num_signal)
                combined = list(np.array(item) for item in combined)

                doas.extend(combined)

            else:
                azimuths = np.arange(
                    doa_config["azimuth"]["range"][0],
                    doa_config["azimuth"]["range"][1],
                    doa_config["azimuth"]["grid"],
                )
                elevations = np.arange(
                    doa_config["elevation"]["range"][0],
                    doa_config["elevation"]["range"][1],
                    doa_config["elevation"]["grid"],
                )

                azimuths, elevations = np.meshgrid(azimuths, elevations)
                azimuths = azimuths.reshape(-1, 1)
                elevations = elevations.reshape(-1, 1)
                azimuth_elevation = np.concatenate(
                    (azimuths, elevations), axis=1
                )
                azimuth_elevation = azimuth_elevation.reshape(-1, 2, 1)

                combined = it.combinations(azimuth_elevation, num_signal)
                combined = list(
                    np.concatenate(item, axis=1) for item in combined
                )

                doas.extend(combined)

        return doas
