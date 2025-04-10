import logging
import logging.config
from pathlib import Path

from utils.util import read_config


def setup_logging(
    save_dir, log_config="logger/logger_config.json", default_level=logging.INFO
):
    """Setup logging configuration"""
    log_config = Path(log_config)

    if log_config.is_file():
        config = read_config(log_config)

        # set log saving path
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])

        logging.config.dictConfig(config)

    else:
        print(
            "Warning: logging configuration file is not found in {}.".format(
                log_config
            )
        )
        logging.basicConfig(level=default_level)
