""" Main entry point of the ONMT library """
from __future__ import division, print_function

import onmt.inputters
import onmt.decoders
from onmt.trainer import Trainer
import sys

src.onmt.utils.optimizers.Optim = src.onmt.utils.optimizers.Optimizer
sys.modules["onmt.Optim"] = src.onmt.utils.optimizers

# For Flake
__all__ = [onmt.inputters, onmt.encoders, onmt.decoders, onmt.models,
           onmt.utils, onmt.modules, "Trainer"]

__version__ = "1.1.1"
