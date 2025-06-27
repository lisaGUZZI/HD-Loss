import cv2 as cv
import numpy as np

import torch
from torch import nn
from typing import Callable

from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve
from nnunetv2.utilities.ddp_allgather import AllGatherGrad

"""
Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf
Modified for nnUNet compatibility with multi-class support
"""


class HausdorffDTLoss(nn.Module):
    """Multi-class Hausdorff loss based on distance transform, compatible with nnUNet"""

    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, 
                 do_bg: bool = True, smooth: float = 1e-5, alpha: float = 2.0, 
                 ddp: bool = True, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.alpha = alpha
        self.ddp = ddp

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        """
        Compute distance transform for multi-class segmentation
        img: (b, c, x, y, z) or (b, c, x, y)
        """
        batch_size, num_classes = img.shape[:2]
        field = np.zeros_like(img)

        for batch in range(batch_size):
            for cls in range(num_classes):
                fg_mask = img[batch, cls] > 0.5

                if fg_mask.any():
                    bg_mask = ~fg_mask
                    fg_dist = edt(fg_mask)
                    bg_dist = edt(bg_mask)
                    field[batch, cls] = fg_dist + bg_dist

        return field

    def forward(self, x, y, loss_mask=None, debug=False) -> torch.Tensor:
        """
        x: prediction (b, c, x, y, z) or (b, c, x, y)
        y: target (b, c, x, y, z) or (b, c, x, y) or (b, x, y, z) or (b, x, y)
        loss_mask: optional mask (b, 1, x, y, z) or (b, 1, x, y)
        """
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # Handle target format (convert to one-hot if needed)
        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # y is already one-hot encoded
                y_onehot = y
            else:
                # Convert label map to one-hot
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)

        # Apply loss mask if provided
        if loss_mask is not None:
            x = x * loss_mask
            y_onehot = y_onehot * loss_mask

        # Remove background class if specified
        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]

        # Compute distance transforms (no gradients needed for distance computation)
        with torch.no_grad():
            pred_dt = torch.from_numpy(self.distance_field(x.detach().cpu().numpy())).float().to(x.device)
            target_dt = torch.from_numpy(self.distance_field(y_onehot.detach().cpu().numpy())).float().to(x.device)
            distance = pred_dt ** self.alpha + target_dt ** self.alpha

        # Compute Hausdorff loss per class (maintain gradients through pred_error)
        pred_error = (x - y_onehot) ** 2
        dt_field = pred_error * distance

        if self.batch_dice:
            # Average over spatial dimensions, keep batch and class dimensions
            axes = tuple(range(2, dt_field.ndim))
            loss_per_class = dt_field.mean(dim=axes)
            
            if self.ddp:
                loss_per_class = AllGatherGrad.apply(loss_per_class).mean(0)
            
            # Average over batch dimension, then over classes
            loss = loss_per_class.mean()
        else:
            # Average over all dimensions
            loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0] if dt_field.shape[1] > 0 else None,
                    pred_error.cpu().numpy()[0, 0] if pred_error.shape[1] > 0 else None,
                    distance.cpu().numpy()[0, 0] if distance.shape[1] > 0 else None,
                    pred_dt.cpu().numpy()[0, 0] if pred_dt.shape[1] > 0 else None,
                    target_dt.cpu().numpy()[0, 0] if target_dt.shape[1] > 0 else None,
                ),
            )
        else:
            return loss


class HausdorffERLoss(nn.Module):
    """Multi-class Hausdorff loss based on morphological erosion, compatible with nnUNet"""

    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False,
                 do_bg: bool = True, smooth: float = 1e-5, alpha: float = 2.0,
                 erosions: int = 10, ddp: bool = True, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.alpha = alpha
        self.erosions = erosions
        self.ddp = ddp
        self.prepare_kernels()

    def prepare_kernels(self):
        # 2D kernel - remove extra dimension
        cross_2d = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        self.kernel2D = cross_2d * 0.2
        
        # 3D kernel - create proper 3D structure
        bound = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        cross_3d = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        
        # Stack to create 3D kernel (3, 3, 3)
        self.kernel3D = np.stack([bound, cross_3d, bound], axis=0) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(self, pred: np.ndarray, target: np.ndarray, debug: bool) -> np.ndarray:
        """
        Perform erosion for multi-class segmentation
        pred: (b, c, x, y, z) or (b, c, x, y)
        target: (b, c, x, y, z) or (b, c, x, y)
        """
        batch_size, num_classes = pred.shape[:2]
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is not supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(batch_size):
            for cls in range(num_classes):
                if debug and cls == 0:  # Only save debug info for first class
                    erosions.append(np.copy(bound[batch, cls]))

                current_bound = bound[batch, cls:cls+1]  # Keep channel dimension
                
                for k in range(self.erosions):
                    # Compute convolution with kernel
                    dilation = convolve(current_bound[0], kernel, mode="constant", cval=0.0)

                    # Apply soft thresholding at 0.5 and normalize
                    erosion = dilation - 0.5
                    erosion[erosion < 0] = 0

                    if np.ptp(erosion) != 0:
                        erosion = (erosion - erosion.min()) / np.ptp(erosion)

                    # Save erosion and add to loss
                    current_bound[0] = erosion
                    eroted[batch, cls] += erosion * (k + 1) ** self.alpha

                    if debug and cls == 0:
                        erosions.append(np.copy(erosion))

        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(self, x, y, loss_mask=None, debug=False) -> torch.Tensor:
        """
        x: prediction (b, c, x, y, z) or (b, c, x, y)
        y: target (b, c, x, y, z) or (b, c, x, y) or (b, x, y, z) or (b, x, y)
        loss_mask: optional mask (b, 1, x, y, z) or (b, 1, x, y)
        """
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # Handle target format (convert to one-hot if needed)
        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # y is already one-hot encoded
                y_onehot = y
            else:
                # Convert label map to one-hot
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)

        # Apply loss mask if provided
        if loss_mask is not None:
            x = x * loss_mask
            y_onehot = y_onehot * loss_mask

        # Remove background class if specified
        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]

        if debug:
            eroted, erosions = self.perform_erosion(
                x.detach().cpu().numpy(), y_onehot.detach().cpu().numpy(), debug
            )
            if self.batch_dice:
                axes = tuple(range(2, len(eroted.shape)))
                loss = eroted.mean(axis=axes).mean()
            else:
                loss = eroted.mean()
            return loss, erosions
        else:
            with torch.no_grad():
                eroted = torch.from_numpy(
                    self.perform_erosion(x.detach().cpu().numpy(), y_onehot.detach().cpu().numpy(), debug)
                ).float().to(x.device)

            if self.batch_dice:
                # Average over spatial dimensions, keep batch and class dimensions
                axes = tuple(range(2, eroted.ndim))
                loss_per_class = eroted.mean(dim=axes)
                
                if self.ddp:
                    loss_per_class = AllGatherGrad.apply(loss_per_class).mean(0)
                
                # Average over batch dimension, then over classes
                loss = loss_per_class.mean()
            else:
                # Average over all dimensions
                loss = eroted.mean()

            return loss


# Alias for backward compatibility
HausdorffLoss = HausdorffDTLoss