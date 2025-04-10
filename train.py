import argparse

import torch

import dataset as dataset_module
import model as model_module
import model.custom_module.loss as loss_module
import model.custom_module.metric as metric_module
from trainer.trainer import Trainer
from utils.parse_config import ConfigParser
from utils.util import get_device

# fix random seeds for reproducibility
SEED = 3407
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# If your model does not change and your input sizes remain the same - then you
# may benefit from setting torch.backends.cudnn.benchmark = True.
# However, if your model changes: for instance, if you have layers that are only
# "activated" when certain conditions are met, or you have layers inside a loop
# that can be iterated a different number of times, then setting
# torch.backends.cudnn.benchmark = True might stall your execution.
torch.backends.cudnn.benchmark = False


def main(config: ConfigParser):
    # create a logger object
    logger = config.get_logger("train")

    device = get_device()

    dataset = config.init_obj(dataset_module, "dataset")

    # build model architecture, then print to console
    model = config.init_obj(model_module, "model").to(device)

    logger.info(model)

    criterion = config.init_obj(loss_module, "loss")

    metrics = config["metrics"] if (config["metrics"] is not None) else []
    if not isinstance(metrics, list):
        raise ValueError("metrics must be a list")
    metrics = [getattr(metric_module, met) for met in metrics]

    trainer = Trainer(
        config=config,
        model=model,
        device=device,
        dataset=dataset,
        criterion=criterion,
        metrics=metrics,
        do_validation=True,
    )

    trainer.train()

    dataset.close()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Project")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to model run id path (default: None)",
    )
    args.add_argument(
        "-b",
        "--benchmark",
        action="store_true",
        help="set torch.backends.cudnn.benchmark to true (default: False)",
    )

    args = args.parse_args()
    if args.benchmark:
        torch.backends.cudnn.benchmark = True

    # init checkpoint dir and get the config object
    config = ConfigParser.from_args(args)
    main(config)
