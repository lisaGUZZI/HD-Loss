import numpy as np

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.hd import HausdorffDTLoss, HausdorffERLoss
from nnunetv2.training.loss.compound_losses import DC_and_HD_loss, DC_and_HDDT_loss, DC_and_HDER_loss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1
import torch


class nnUNetTrainerHDLoss(nnUNetTrainer):
    """
    nnUNet trainer using pure Hausdorff Distance loss only (no Dice).
    Use hd_variant parameter to choose between 'dt' (Distance Transform) or 'er' (Erosion).
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), hd_variant: str = 'dt'):
        # Handle hd_variant before calling super to avoid KeyError in base class
        self.hd_variant = hd_variant.lower()
        if self.hd_variant not in ['dt', 'er']:
            raise ValueError(f"hd_variant must be 'dt' or 'er', got {hd_variant}")
            
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        
    def _build_loss(self):
        # Hausdorff loss parameters
        hd_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': self.label_manager.has_regions,
            'smooth': 1e-5,
            'alpha': 2.0,
            'ddp': self.is_ddp,
        }
        
        # Add erosion parameter for ER variant
        if self.hd_variant == 'er':
            hd_kwargs['erosions'] = 10
            loss = HausdorffERLoss(apply_nonlin=softmax_helper_dim1, **hd_kwargs)
        else:  # 'dt'
            loss = HausdorffDTLoss(apply_nonlin=softmax_helper_dim1, **hd_kwargs)

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


class nnUNetTrainerHDDTLoss(nnUNetTrainerHDLoss):
    """Pure Hausdorff Distance Transform loss (no Dice)"""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device, hd_variant='dt')


class nnUNetTrainerHDERLoss(nnUNetTrainerHDLoss):
    """Pure Hausdorff Erosion loss (no Dice)"""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device, hd_variant='er')


# ============================================================================
# COMPOUND LOSSES (Dice + Hausdorff) with Dynamic Weighting
# ============================================================================

class nnUNetTrainerDC_HD_Loss(nnUNetTrainer):
    """
    nnUNet trainer using Dice + Hausdorff Distance compound loss with dynamic weighting.
    Uses Distance Transform-based Hausdorff loss by default.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        
    def _build_loss(self):
        # Dice loss parameters
        dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': self.label_manager.has_regions,
            'smooth': 1e-5,
            'ddp': self.is_ddp,
        }
        
        # Hausdorff loss parameters
        hd_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': self.label_manager.has_regions,
            'smooth': 1e-5,
            'alpha': 2.0,
            'ddp': self.is_ddp,
        }
        
        # Create compound loss with dynamic weighting
        loss = DC_and_HDDT_loss(
            soft_dice_kwargs=dice_kwargs,
            hd_kwargs=hd_kwargs,
            weight_dice=1.0,
            weight_hd=1.0,  # Initial lambda, will be updated dynamically
            ignore_label=None,
            dynamic_weighting=True
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
        
    def on_epoch_end(self):
        """Called at the end of each epoch to update lambda for dynamic weighting"""
        super().on_epoch_end()
        
        # Update lambda for dynamic weighting if using compound HD loss
        if hasattr(self.loss, 'update_lambda'):
            # Direct compound loss
            self.loss.update_lambda()
            current_lambda = self.loss.get_current_lambda()
            self.print_to_log_file(f"Updated HD loss lambda: {current_lambda:.6f}")
        elif hasattr(self.loss, 'loss') and hasattr(self.loss.loss, 'update_lambda'):
            # Deep supervision wrapped compound loss
            self.loss.loss.update_lambda()
            current_lambda = self.loss.loss.get_current_lambda()
            self.print_to_log_file(f"Updated HD loss lambda: {current_lambda:.6f}")


class nnUNetTrainerDC_HDDT_Loss(nnUNetTrainerDC_HD_Loss):
    """
    nnUNet trainer using Dice + Hausdorff Distance Transform compound loss with dynamic weighting.
    (Same as base DC_HD_Loss, provided for explicit naming)
    """
    pass


class nnUNetTrainerDC_HDER_Loss(nnUNetTrainerDC_HD_Loss):
    """
    nnUNet trainer using Dice + Hausdorff Erosion compound loss with dynamic weighting.
    """
    def _build_loss(self):
        # Dice loss parameters
        dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': self.label_manager.has_regions,
            'smooth': 1e-5,
            'ddp': self.is_ddp,
        }
        
        # Hausdorff Erosion loss parameters
        hd_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': self.label_manager.has_regions,
            'smooth': 1e-5,
            'alpha': 2.0,
            'erosions': 10,
            'ddp': self.is_ddp,
        }
        
        # Create compound loss with dynamic weighting (Erosion-based)
        loss = DC_and_HDER_loss(
            soft_dice_kwargs=dice_kwargs,
            hd_kwargs=hd_kwargs,
            weight_dice=1.0,
            weight_hd=1.0,  # Initial lambda, will be updated dynamically
            ignore_label=None,
            dynamic_weighting=True
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


