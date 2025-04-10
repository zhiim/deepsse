import sys
from pathlib import Path

import numpy as np

import dataset.data_generator as data_generator
import dataset.data_process as data_process_module
from utils.util import init_function, read_config, write_config

SEED = 3407
# create a random number generator with specific seed for reproducibility
rng = np.random.default_rng(seed=SEED)


def main(config):
    # save config file to the specified path
    cfg_save_path = Path("saved", config["project_name"], "data_configs")
    cfg_save_path.mkdir(parents=True, exist_ok=True)
    write_config(config, cfg_save_path / config["config_name"])

    generator = getattr(data_generator, config["generator"])(config, rng=rng)

    # save path of dataset and label
    dataset_path = Path("data", config["save_path"]["dataset"])
    label_path = Path("data", config["save_path"]["label"])

    data_process = init_function(data_process_module, config["data_process"])
    label_process = init_function(data_process_module, config["label_process"])

    num_data = config["num_data"] if "num_data" in config else None

    save_threthod = (
        config["save_threthod"] if "save_threthod" in config else None
    )

    parallelism = config["parallelism"] if "parallelism" in config else False

    n_jobs = config["n_jobs"] if "n_jobs" in config else -1

    paral_mode = config["paral_mode"] if "paral_mode" in config else "processes"

    generator.generate(
        data_path=dataset_path,
        label_path=label_path,
        num_repeat=config["num_repeat"],
        snrs=config["snrs"],
        nsamples=config["nsamples"],
        data_process=data_process,
        label_process=label_process,
        num_data=num_data,
        save_threthod=save_threthod,
        parallelism=parallelism,
        n_jobs=n_jobs,
        paral_mode=paral_mode,
    )


if __name__ == "__main__":
    # get config file path from command line arg
    cfg_name = Path(sys.argv[1])

    # read json config
    config = read_config(cfg_name)

    main(config)
