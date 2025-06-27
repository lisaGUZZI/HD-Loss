import numpy as np
from nnunetv2.training.loss.compound_losses import CE_and_LH_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from nnunetv2.utilities.helpers import softmax_helper_dim1


class nnUNetTrainerCEandLHLoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_weight_lh = 1.0
        self.weight_ce = 1.0
        self.lh_kwargs = {'iters': 2, 'connec': 26, 'alpha': 150, 'norm': 0, 'do_bg': self.label_manager.has_regions,
            'apply_nonlin': torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1,
            'batch_lah': self.configuration_manager.batch_dice}
        self.num_epochs = 250

    def _build_loss(self):
        ce_kwargs = {}
        
        loss = CE_and_LH_loss(
            ce_kwargs=ce_kwargs,
            lh_kwargs=self.lh_kwargs,
            weight_ce=self.weight_ce, 
            weight_lh=self.initial_weight_lh,
            ignore_label=self.label_manager.ignore_label
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss 