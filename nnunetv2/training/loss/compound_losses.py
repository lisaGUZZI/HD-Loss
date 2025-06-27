import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.training.loss.lahsym import LAHsymLoss
from nnunetv2.training.loss.lah import LAHLoss
from nnunetv2.training.loss.lh import LHLoss
from nnunetv2.training.loss.hd import HausdorffDTLoss, HausdorffERLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
        
        breakpoint()

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_LAHsym_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, lah_kwargs, weight_dice=1, weight_lah=1, ignore_label=None):
        super(DC_and_LAHsym_loss, self).__init__()
        if ignore_label is not None:
            # For dice loss, we handle ignore_label via loss_mask
            pass
            
        self.weight_dice = weight_dice
        self.weight_lah = weight_lah
        self.ignore_label = ignore_label
        
        # Initialize losses
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.lah = LAHsymLoss(apply_nonlin=softmax_helper_dim1, **lah_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
            mask = target != self.ignore_label
            target_for_loss = torch.where(mask, target, 0)
        else:
            target_for_loss = target
            mask = None

        dc_loss = self.dc(net_output, target_for_loss, loss_mask=mask) if self.weight_dice != 0 else 0
        lah_loss = self.lah(net_output, target_for_loss, loss_mask=mask) if self.weight_lah != 0 else 0

        result = self.weight_dice * dc_loss + self.weight_lah * lah_loss
        return result


class CE_and_LAHsym_loss(nn.Module):
    def __init__(self, ce_kwargs, lah_kwargs, weight_ce=1, weight_lah=1, ignore_label=None):
        super(CE_and_LAHsym_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_lah = weight_lah
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.lah = LAHsymLoss(**lah_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
            mask = (target != self.ignore_label)
            target_for_loss = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_for_loss = target
            mask = None
            num_fg = 2 # dummy
        # breakpoint()

        lah_loss = self.lah(net_output, target_for_loss) \
            if self.weight_lah != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_lah * lah_loss
        return result


class CE_and_LAH_loss(nn.Module):
    def __init__(self, ce_kwargs, lah_kwargs, weight_ce=1, weight_lah=1, ignore_label=None):
        super(CE_and_LAH_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_lah = weight_lah
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.lah = LAHLoss(**lah_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
            mask = (target != self.ignore_label)
            target_for_loss = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_for_loss = target
            mask = None
            num_fg = 2 # dummy

        lah_loss = self.lah(net_output, target_for_loss) \
            if self.weight_lah != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_lah * lah_loss
        return result


class CE_and_LH_loss(nn.Module):
    def __init__(self, ce_kwargs, lh_kwargs, weight_ce=1, weight_lh=1, ignore_label=None):
        super(CE_and_LH_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_lh = weight_lh
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.lh = LHLoss(**lh_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
            mask = (target != self.ignore_label)
            target_for_loss = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_for_loss = target
            mask = None
            num_fg = 2 # dummy

        lh_loss = self.lh(net_output, target_for_loss) \
            if self.weight_lh != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_lh * lh_loss
        return result


class DC_and_HD_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, hd_kwargs, weight_dice=1.0, weight_hd=1.0, 
                 ignore_label=None, dice_class=SoftDiceLoss, hd_class=HausdorffDTLoss,
                 dynamic_weighting=True):
        """
        Dice + Hausdorff Distance compound loss with dynamic weighting.
        
        When dynamic_weighting=True, the weight_hd parameter serves as the initial lambda,
        and it will be updated after each epoch based on the ratio of mean HD loss to mean Dice loss.
        
        :param soft_dice_kwargs: kwargs for dice loss
        :param hd_kwargs: kwargs for hausdorff loss  
        :param weight_dice: weight for dice loss (kept constant at 1.0 for dynamic weighting)
        :param weight_hd: initial weight for HD loss (lambda), will be updated if dynamic_weighting=True
        :param ignore_label: label to ignore in loss computation
        :param dice_class: which dice loss class to use
        :param hd_class: which hausdorff loss class to use (HausdorffDTLoss or HausdorffERLoss)
        :param dynamic_weighting: whether to use dynamic lambda adjustment
        """
        super(DC_and_HD_loss, self).__init__()
        
        if ignore_label is not None:
            # For dice loss, we handle ignore_label via loss_mask
            pass
            
        self.weight_dice = weight_dice
        self.weight_hd = weight_hd
        self.ignore_label = ignore_label
        self.dynamic_weighting = dynamic_weighting
        
        # Initialize losses
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.hd = hd_class(apply_nonlin=softmax_helper_dim1, **hd_kwargs)
        
        # For dynamic weighting - accumulate losses during epoch
        self.reset_epoch_stats()
        
    def reset_epoch_stats(self):
        """Reset loss accumulation for new epoch"""
        self.epoch_dice_losses = []
        self.epoch_hd_losses = []
        
    def update_lambda(self):
        """
        Update lambda (weight_hd) based on ratio of mean HD loss to mean Dice loss.
        Call this at the end of each epoch.
        """
        if not self.dynamic_weighting or len(self.epoch_dice_losses) == 0 or len(self.epoch_hd_losses) == 0:
            return
            
        mean_dice_loss = torch.stack(self.epoch_dice_losses).mean().item()
        mean_hd_loss = torch.stack(self.epoch_hd_losses).mean().item()
        
        if mean_dice_loss > 0:
            # λ = mean_HD_loss / mean_DSC_loss
            self.weight_hd = mean_hd_loss / mean_dice_loss
        
        # Reset for next epoch
        self.reset_epoch_stats()
        
    def get_current_lambda(self):
        """Get current lambda value"""
        return self.weight_hd

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_HD_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        # Compute individual losses
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        hd_loss = self.hd(net_output, target_dice, loss_mask=mask) \
            if self.weight_hd != 0 else 0

        # Store losses for dynamic weighting (only during training)
        if self.training and self.dynamic_weighting:
            if isinstance(dc_loss, torch.Tensor):
                self.epoch_dice_losses.append(dc_loss.detach())
            if isinstance(hd_loss, torch.Tensor):
                self.epoch_hd_losses.append(hd_loss.detach())

        # Combine losses
        result = self.weight_dice * dc_loss + self.weight_hd * hd_loss
        return result


class DC_and_HDER_loss(DC_and_HD_loss):
    """Convenience class for Dice + Hausdorff Erosion loss"""
    def __init__(self, soft_dice_kwargs, hd_kwargs, weight_dice=1.0, weight_hd=1.0, 
                 ignore_label=None, dice_class=SoftDiceLoss, dynamic_weighting=True):
        super().__init__(
            soft_dice_kwargs=soft_dice_kwargs,
            hd_kwargs=hd_kwargs,
            weight_dice=weight_dice,
            weight_hd=weight_hd,
            ignore_label=ignore_label,
            dice_class=dice_class,
            hd_class=HausdorffERLoss,
            dynamic_weighting=dynamic_weighting
        )


class DC_and_HDDT_loss(DC_and_HD_loss):
    """Convenience class for Dice + Hausdorff Distance Transform loss"""
    def __init__(self, soft_dice_kwargs, hd_kwargs, weight_dice=1.0, weight_hd=1.0, 
                 ignore_label=None, dice_class=SoftDiceLoss, dynamic_weighting=True):
        super().__init__(
            soft_dice_kwargs=soft_dice_kwargs,
            hd_kwargs=hd_kwargs,
            weight_dice=weight_dice,
            weight_hd=weight_hd,
            ignore_label=ignore_label,
            dice_class=dice_class,
            hd_class=HausdorffDTLoss,
            dynamic_weighting=dynamic_weighting
        )
