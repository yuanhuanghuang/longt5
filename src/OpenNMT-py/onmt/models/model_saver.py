import os
import torch

from collections import deque
from onmt.utils.logging import logger

from copy import deepcopy


def build_model_saver(model_opt, opt, model, optim):
    model_saver = ModelSaver(opt.save_model, model, model_opt, optim)
    return model_saver


class ModelSaverBase(object):
    """Base class for model saving operations

    Inherited classes must implement private methods:
    * `_save`
    * `_rm_checkpoint
    """

    def __init__(self, base_path, model, model_opt, optim):
        self.base_path = base_path
        self.model = model
        self.model_opt = model_opt
        self.optim = optim

    def save(self, tp=""):
        """Main entry point for model saver

        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """

        save_model = self.model
        chkpt, chkpt_name = self._save(save_model, tp)

    def _save(self, model, tp):
        """Save a resumable checkpoint.

        Args:

        Returns:
            (object, str):

            * checkpoint: the saved object
            * checkpoint_name: name (or path) of the saved checkpoint
        """

        raise NotImplementedError()

    def _rm_checkpoint(self, name):
        """Remove a checkpoint

        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """

        raise NotImplementedError()


class ModelSaver(ModelSaverBase):
    """Simple model saver to filesystem"""

    def _save(self, model, tp):
        model_state_dict = model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator.' not in k}
        generator_state_dict = model.generator.state_dict()

        # NOTE: We need to trim the vocab to remove any unk tokens that
        # were not originally here.

        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'opt': self.model_opt,
            'optim': self.optim.state_dict(),
        }

        logger.info("Saving checkpoint %s_%s.pt" % (self.base_path, tp))
        checkpoint_path = '%s_%s.pt' % (self.base_path, tp)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        if os.path.exists(name):
            os.remove(name)
