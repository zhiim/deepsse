import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from logger.logger import setup_logging
from utils.util import read_config, write_config


class ConfigParser:
    def __init__(self, config, resume=None, run_id=None, use_yaml=False):
        """class to parse configuration json file. Handles hyperparameters for
        training, initializations of modules, checkpoint saving and logging
        module.

        Create dir for checkpoint and log, init logger

        Args:
            config: Dict containing configurations, hyperparameters for
                training. contents of `config.json` file for example.
            resume: String, path to the checkpoint being loaded.
            run_id: Unique Identifier for training processes. Used to save
                checkpoints and training log. Timestamp is being used as default
            use_yaml: Bool, if True, read and write config file as yaml format.
                Defaults to False.
        """
        self._config = config
        self._resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = Path(config["trainer"]["save_dir"])

        exper_name = config["name"]  # name of this experiment

        # use timestamp as default run-id
        if run_id is None:
            run_id = datetime.now().strftime(r"%m%d_%H%M%S")

        # model saving path
        self._save_dir = save_dir / exper_name / run_id / "models"
        # log saving path
        self._log_dir = save_dir / exper_name / run_id / "log"

        # make directory for saving checkpoints and log.
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # save updated config file to the run_id dir
        if use_yaml:
            write_config(
                self._config, save_dir / exper_name / run_id / "config.yaml"
            )
        else:
            write_config(
                self._config, save_dir / exper_name / run_id / "config.json"
            )

        # configure logging module
        setup_logging(self._log_dir)

        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG,
        }

    @classmethod
    def from_args(cls, args):
        """Initialize this class from some cli arguments. Used in train, test"""
        if args.resume is not None:
            resume = Path(args.resume)  # resume is the path of run id
            cfg_fname = resume / "config.yaml"
            if not cfg_fname.exists():
                cfg_fname = resume / "config.json"

        # if resume is not specified, config must be provided
        else:
            msg_no_cfg = "Configuration file need to be specified. Add \
                '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)

        if cfg_fname.suffix == ".yaml":
            use_yaml = True
        else:
            use_yaml = False

        config = read_config(cfg_fname)

        # if resume and config is provided at the same time, we can update
        # resume config file
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_config(args.config))

        if "run_id" in config:
            run_id = config["run_id"]
        else:
            run_id = None

        return cls(config, resume, run_id, use_yaml)

    def __getitem__(self, name: str) -> Any | None:
        """Access items like ordinary dict."""
        if name not in self._config:
            return None
        return self._config[name]

    def get_logger(self, name, verbosity=2):
        """Create a logger object with the specified name

        Args:
            name (str): name of logger
            verbosity (int, optional): level of logging. Defaults to 2.

        Returns:
            logger
        """
        msg_verbosity = (
            "verbosity option {} is invalid. Valid options are {}.".format(
                verbosity, self.log_levels.keys()
            )
        )
        assert verbosity in self.log_levels, msg_verbosity

        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])

        return logger

    def init_obj(self, module, module_type, **kwargs):
        """Finds a function handle with the name given as 'type' in config, and
        returns the instance initialized with corresponding arguments given.

        `object = config.init_obj(module, 'module_type', b=1)` is equivalent to
        `object = module.config['module_type']['type'](args_in_config, b=1)`
        """
        module_name = self._config[module_type]["type"]

        # use args in config
        if "args" in self._config[module_type]:
            module_args = dict(self._config[module_type]["args"])
        else:
            module_args = dict()

        # update module_args with kwargs
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)

        return getattr(module, module_name)(**module_args)

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def resume(self):
        return self._resume

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir
