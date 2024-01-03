import logging
import sys

import torch
import torch.nn.functional as F


class KLLossSoft(torch.nn.modules.loss._Loss):
    def forward(self, output, target, T=1.0):
        output, target = output / T, target / T
        target_prob = F.softmax(target, dim=1)
        output_log_prob = F.log_softmax(output, dim=1)
        loss = - torch.sum(target_prob * output_log_prob, dim=1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def setup_logging(save_path, mode: str = "a"):
    logging.root.handlers = []
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    print_plain_formatter = logging.Formatter(
        "[%(asctime)s]: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    fh_plain_formatter = logging.Formatter("%(message)s")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(print_plain_formatter)
    logger.addHandler(ch)
    if save_path is not None:
        fh = logging.FileHandler(save_path, mode=mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fh_plain_formatter)
        logger.addHandler(fh)
