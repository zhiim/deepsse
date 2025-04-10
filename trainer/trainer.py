import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from trainer.base_trainer import BaseTrainer
from utils.helper import MetricTracker
from utils.util import to_numpy


class Trainer(BaseTrainer):
    """Trainer class"""

    def __init__(
        self,
        config,
        model,
        device,
        dataset,
        criterion,
        metrics,
        do_validation=False,
    ):
        super().__init__(
            config, model, device, dataset, criterion, metrics, do_validation
        )

        # record loss and metrics every `log_step` batches
        if self.data_loader.batch_size is None:
            raise ValueError(
                "DataLoader batch_size must be defined for training"
            )
        self.log_step = int(np.sqrt(self.data_loader.batch_size))

        # creat tracker used to recod training and validation loss and
        # performance metrics
        self.writer = SummaryWriter(self.config.log_dir)
        self.train_tracker = MetricTracker(
            "loss", *[m.__name__ for m in metrics]
        )
        self.val_tracker = MetricTracker("loss", *[m.__name__ for m in metrics])

    def _train_epoch(self, epoch):
        """Training logic for an epoch

        Args:
            epoch: Integer, current training epoch.

        Returns:
            A log that contains average loss and metric in this epoch.
        """
        # Set the model to training mode - important for batch normalization and
        # dropout layers
        self.model.train()

        self.train_tracker.reset()  # reset record to 0
        for batch_idx, (data_, target_) in enumerate(self.data_loader):
            # move data to device
            data = data_.to(self.device)
            target = target_.to(self.device)

            self.optimizer.zero_grad()

            predict = self.model(data)  # model output
            loss = self.criterion(predict, target)
            loss.backward()  # get the gradient
            self.optimizer.step()  # update parameter in model

            # record training loss and into tracker
            self.train_tracker.update("loss", loss.item())

            # calculate metric and record into tracker
            mets = []
            for met in self.metrics:
                mets.append(
                    met(
                        to_numpy(target),
                        to_numpy(predict),
                    )
                )
                self.train_tracker.update(met.__name__, mets[-1])

            if batch_idx % self.log_step == 0 or batch_idx == 0:
                # record training loss and performance metric into tracker
                current_step = (epoch - 1) * len(self.data_loader) + batch_idx
                self.writer.add_scalar(
                    tag="train_loss",
                    scalar_value=loss.item(),
                    global_step=current_step,
                )
                training_progress = (
                    "Train Epoch: {}/{} {} Loss: {:.6f} ".format(
                        epoch,
                        self.epochs,
                        self._progress(batch_idx),
                        loss.item(),
                    )
                )
                for i, met in enumerate(self.metrics):
                    self.writer.add_scalar(
                        tag="train_" + met.__name__,
                        scalar_value=mets[i],
                        global_step=current_step,
                    )
                    training_progress += "{}: {:.6f} ".format(
                        met.__name__, mets[i]
                    )

                # record log
                self.logger.debug(training_progress)

        # put tracked loss and metric into a dict
        log = self.train_tracker.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """Validate after training an epoch

        Args:
            epoch: Integer, current training epoch.

        Returns:
            A log that contains information about validation
        """
        if self.valid_data_loader.batch_size is None:
            raise ValueError(
                "DataLoader batch_size must be defined for validation"
            )
        record_steps = int(np.sqrt(self.valid_data_loader.batch_size))
        # Set the model to evaluation mode - important for batch normalization
        # and dropout layers
        self.model.eval()

        self.val_tracker.reset()
        with torch.no_grad():
            for batch_idx, (data_, target_) in enumerate(
                tqdm(
                    self.valid_data_loader,
                    "Validation at epoch {}".format(epoch),
                )
            ):
                data = data_.to(self.device)
                target = target_.to(self.device)

                predict = self.model(data)

                loss = self.criterion(predict, target)
                self.val_tracker.update("loss", loss.item())

                mets = []
                for met in self.metrics:
                    mets.append(
                        met(
                            target.cpu().detach().numpy(),
                            predict.cpu().detach().numpy(),
                        )
                    )
                    self.val_tracker.update(met.__name__, mets[-1])

                if batch_idx % record_steps == 0 or batch_idx == 0:
                    current_step = (epoch - 1) * len(
                        self.valid_data_loader
                    ) + batch_idx
                    self.writer.add_scalar(
                        tag="val_loss",
                        scalar_value=loss.item(),
                        global_step=current_step,
                    )
                    for i, met in enumerate(self.metrics):
                        self.writer.add_scalar(
                            tag="val_" + met.__name__,
                            scalar_value=mets[i],
                            global_step=current_step,
                        )

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, global_step=epoch, bins="auto")

        return self.val_tracker.result()

    def _split_validation(self, dataset):
        """Split dataset into training and validatioin dataset"""
        dataloader_cfg = self.config["data_loader"]
        if dataloader_cfg is None:
            raise ValueError("data_loader configuration is not defined")

        split_ratio = dataloader_cfg["validation_split"]
        val_len = int(split_ratio * len(dataset))  # length of training dataset
        lengths = [len(dataset) - val_len, val_len]

        # split dataset
        trainset, valset = random_split(dataset, lengths)

        # create dataloader
        train_loader = DataLoader(
            dataset=trainset,
            batch_size=dataloader_cfg["batch_size"],
            shuffle=dataloader_cfg["shuffle"],
        )
        val_loader = DataLoader(
            dataset=valset,
            batch_size=dataloader_cfg["batch_size"],
            shuffle=dataloader_cfg["shuffle"],
        )

        return train_loader, val_loader

    def _progress(self, batch_idx):
        """Current training progress in this epoch"""
        base = "[{}/{} ({:.0f}%)]"
        current = batch_idx  # index of curren batch
        total = len(self.data_loader)
        return base.format(current, total, 100.0 * current / total)
