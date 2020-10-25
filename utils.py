import os
import logging
import torch.nn as nn


def create_net(n_inputs, n_outputs, n_layers=1, n_units=100, nonlinear=nn.Tanh):
    if n_layers == 0:
        return nn.Linear(n_inputs, n_outputs)
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers-1):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger


def logout(logger, s):
    if logger:
        logger.info(s)
    else:
        print(s)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


