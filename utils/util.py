import json
from pathlib import Path

import numpy as np
import ruamel.yaml
import torch

yaml = ruamel.yaml.YAML()
yaml.indent(mapping=2, sequence=4, offset=2)

CommentDict = ruamel.yaml.CommentedMap


def read_config(fname: Path | str) -> CommentDict | dict:
    """Read configuration file.

    Args:
        fname: Configuration file name, either `.json` or `.yaml` file

    Returns:
        dict: Configuration dictionary
    """
    fname = Path(fname)
    with fname.open("rt") as handle:
        if fname.suffix == ".yaml":
            return yaml.load(handle)
        else:
            return json.load(handle)


def write_config(content: CommentDict | dict, fname: Path | str):
    """Write configuration file.

    Args:
        content: configuration dictionary to be written
        fname: Configuration file name, either `.json` or `.yaml` file
    """
    fname = Path(fname)
    with fname.open("wt") as handle:
        if fname.suffix == ".yaml":
            yaml.dump(content, handle)
        else:
            json.dump(content, handle, indent=2, sort_keys=False)


def get_device():
    """support gpu or not"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def init_object(module, config):
    """Initialize object from module and configuration.

    Args:
        module (): module object
        config (): configuration dictionary

    Returns:
        object
    """
    module_type = config["type"]
    if "args" in config:
        module_args = dict(config["args"])
    else:
        module_args = dict()

    return getattr(module, module_type)(**module_args)


def init_function(module, config):
    """Get processing function wrapper.
    The returned processing function will be wrapped to keep only the first
    argument of the original function.

    Args:
        config (dict): configuration dictionary

    Returns:
        function: the processing function
    """
    if "args" in config:
        # define a wrapped function using parameters read from config file
        def process(x):
            # x will be data passed into process()
            return getattr(module, config["type"])(x, **config["args"])
    else:
        process = getattr(module, config["type"])

    return process


def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return (
            data.detach().cpu().numpy()
            if data.requires_grad
            else data.cpu().numpy()
        )
    raise ValueError("Unsupported type: {}".format(type(data)))


def to_torch(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(get_device())
    raise ValueError("Unsupported type: {}".format(type(data)))
