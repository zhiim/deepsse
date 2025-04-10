import os
from abc import ABC, abstractmethod
from typing import Literal

import torch
from numpy import inf
from torch.utils.data import DataLoader

from utils.parse_config import ConfigParser


class BaseTrainer(ABC):
    """Base class for all trainers

    This base class defines methods like load parameter from config file, save
    checkpoints, and resume from checkpoints.

    You need to inherite it and define your training epoch and dataset spliting
    trategy.
    """

    def __init__(
        self,
        config: ConfigParser,
        model,
        device: Literal["cpu", "cuda"],
        dataset,
        criterion,
        metrics: list,
        do_validation: bool = False,
    ):
        """Mange training and checkpoint saving

        Args:
            config: json object read from config file
            model : network model
            device: GPU or CPU
            dataset: dataset used for training, which can be divided into
                training dataset and validation dataset if `do_validation` is
                `True`
            do_validation: set to `True` to divide dataset into training dataset
                and validation dataset
        """
        self.config = config

        self.model = model
        self.device = device
        self.criterion = criterion
        self.metrics = metrics
        self.do_validation = do_validation

        # ━━ initiate data loader ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if do_validation:
            # split dataset into training and validation dataset
            self.data_loader, self.valid_data_loader = self._split_validation(
                dataset
            )
        else:
            # initiate dataloader with the entire dataset
            data_loader_cfg = config["data_loader"]
            if data_loader_cfg is None:
                raise ValueError("data_loader configuration is not defined")
            self.data_loader = DataLoader(
                dataset=dataset,
                batch_size=data_loader_cfg["batch_size"],
                shuffle=data_loader_cfg["shuffle"],
            )

        # ━━ create optimizer from config file ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        optimizer_cfg = config["optimizer"]
        if optimizer_cfg is None:
            raise ValueError("optimizer configuration is incorrectly defined")
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = getattr(torch.optim, optimizer_cfg["type"])(
            trainable_params, **optimizer_cfg["args"]
        )

        # ━━ create learning rate scheduler from config file ━━━━━━━━━━━━━━━━━━━
        lr_scheduler_cfg = config["lr_scheduler"]
        if lr_scheduler_cfg is None:
            raise ValueError(
                "lr_scheduler configuration is incorrectly defined"
            )
        self.lr_scheduler = getattr(
            torch.optim.lr_scheduler, lr_scheduler_cfg["type"]
        )(self.optimizer, **lr_scheduler_cfg["args"])

        # ━━ read trainer config from config file and initiate logger ━━━━━━━━━━
        trainer_cfg = config["trainer"]
        if trainer_cfg is None:
            raise ValueError("trainer configuration is incorrectly defined")
        # read trainer config
        self.epochs = trainer_cfg["epochs"]
        self.save_period = trainer_cfg["save_period"]
        self.max_saved_num = trainer_cfg["max_saved_num"]
        self.monitor = trainer_cfg.get("monitor", "off")
        self.logger = config.get_logger("trainer", trainer_cfg["verbosity"])

        # ━━ configuration to monitor model performance and save best ━━━━━━━━━━
        if self.monitor == "off":  # do not use monitor
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            # mnt_metric is the name of metric used to evaluate whether
            # performance of model has improved.
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            # set initial monitor best record
            self.mnt_best = inf if self.mnt_mode == "min" else -inf

            self.early_stop = trainer_cfg.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.checkpoint_dir = config.save_dir

        # when init this class, start epoch at 1
        self.start_epoch = 1

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """Training logic for an epoch

        Args:
            epoch: Integer, current training epoch.

        Returns:
            A log that contains average loss and metric in this epoch.
        """
        raise NotImplementedError

    def train(self):
        """Full training logic"""
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)  # training in each epoch

            # save logged informations into log dict
            log = {"epoch": epoch}
            # result will be like {"val_loss": xxx} or {"val_acc": xxx}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("{:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric, save
            # best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not, according
                    # to specified metric(mnt_metric)
                    improved = (
                        self.mnt_mode == "min"
                        and log[self.mnt_metric] <= self.mnt_best
                    ) or (
                        self.mnt_mode == "max"
                        and log[self.mnt_metric] >= self.mnt_best
                    )
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                # if not improved count reach early stop, stop training
                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            # save checkpoint every save_period epoches
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """Saving checkpoints.

        In addition to saving every n epoches, it will also save the best
        checkpoint

        Args:
            epoch: current epoch number
            save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__  # network arch: name of model class
        optimizer_type = type(self.optimizer).__name__
        state = {
            "arch": arch,  # network name
            "epoch": epoch,  # at which epoch it paused
            "state_dict": self.model.state_dict(),  # model parameter
            "optimizer_type": optimizer_type,  # optimizer name
            # save optimizer parameter, which can be use to resume training
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
        }

        filename = str(
            self.checkpoint_dir / "checkpoint_epoch_{}.pth".format(epoch)
        )
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

        # save best checkpoint
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            # save weight only, because we use best model to inference only
            torch.save(self.model.state_dict(), best_path)
            self.logger.info("Saving current best: model_best.pth ...")

        # save only limited number of last checkpoints
        checkpoints = [
            chkpt
            for chkpt in os.listdir(self.checkpoint_dir)
            if chkpt.endswith(".pth") and chkpt != "model_best.pth"
        ]
        if len(checkpoints) > self.max_saved_num:
            checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            os.remove(os.path.join(self.checkpoint_dir, checkpoints[0]))

    def _resume_checkpoint(self, resume_path):
        """Resume from saved checkpoints

        start_epoch, mnt_best, model parameters and optimizer parameters will be
        loaded

        Args:
            resume_path: run id path to be resumed
        """
        checkpoints = [
            chkpt
            for chkpt in os.listdir(resume_path / "models")
            if chkpt.endswith(".pth") and chkpt != "model_best.pth"
        ]  # all checkpoints unfer models dir
        checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        resume_checkpoint_path = resume_path / "models" / checkpoints[-1]
        resume_checkpoint_path = str(resume_checkpoint_path)

        self.logger.info(
            "Loading checkpoint: {} ...".format(resume_checkpoint_path)
        )
        checkpoint = torch.load(resume_checkpoint_path)

        # at which epoch to resume
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["arch"] != type(self.model).__name__:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is "
                "different from that of checkpoint."
                "This may yield an exception while state_dict is being loaded."
            )
        else:
            # load model parameter
            self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint
        if checkpoint["optimizer_type"] != type(self.optimizer).__name__:
            self.logger.warning(
                "Warning: Optimizer type given in config file "
                "is different from that of checkpoint. "
                "Optimizer parameters not being resumed."
            )
        else:
            # load optimizer parameter
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(
                self.start_epoch
            )
        )

    @abstractmethod
    def _split_validation(self, dataset) -> tuple[DataLoader, DataLoader]:
        """Split dataset into training dataset and validation dataset

        Args:
            dataset : dataset which need to be split
        """
        raise NotImplementedError()
