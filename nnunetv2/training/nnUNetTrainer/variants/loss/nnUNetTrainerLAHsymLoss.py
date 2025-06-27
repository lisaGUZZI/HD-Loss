import numpy as np
import os
import logging

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.lahsym import LAHsymLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_iters_from_env():
    iters = int(os.environ.get("LAH_ITERS", "2"))
    logger.info("######### Using %d iterations for LAH Loss #########", iters)
    return iters

def get_norm_from_env():
    norm = int(os.environ.get("LAH_NORM", "0"))
    logger.info("######### Using norm=%d for LAH Loss #########", norm)
    return norm

class nnUNetTrainerLAHsymLoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.iters = get_iters_from_env()
        self.norm = get_norm_from_env()
        
    def _build_loss(self):
        loss = LAHLoss(
            do_bg=self.label_manager.has_regions,
            apply_nonlin=torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1,
            batch_lah= self.configuration_manager.batch_dice,
            iters=self.iters,
            norm = self.norm
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
