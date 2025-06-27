import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Union


class SoftErosion3D(nn.Module):
    """ Class implemented using Pytorch module to perform differentiable soft erosion on 3D input image. """
    def __init__(self, tan: bool = False, power: int = 7):
        super(SoftErosion3D, self).__init__()
        self._indices_list = [self.extract_indices(o) for o in range(1)]
        self.doyoutan = tan
        self.tan = nn.Tanh()
        self.power = power
        
    @staticmethod
    def extract_indices(o: int) -> torch.Tensor:
        """
        Function to extract ordered index list in each subdirection (North, East, South, West, Up, Down)
        """
        ind = [
            torch.tensor(
                [
                    [2, 0, 0], [2, 0, 1], [2, 0, 2], [1, 0, 2], [0, 0, 2],
                    [0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1], [2, 1, 0],
                    [2, 1, 1], [2, 1, 2], [1, 1, 2], [0, 1, 2], [0, 1, 1],
                    [0, 1, 0], [1, 1, 0], [2, 2, 0], [2, 2, 1], [2, 2, 2],
                    [1, 2, 2], [0, 2, 2], [0, 2, 1], [0, 2, 0], [1, 2, 0],
                    [1, 2, 1], [1, 1, 1]
                ],
                dtype=torch.long,
            )
        ]
        return ind[o]
        
    def allcondArithm(self, n: torch.Tensor, connectivity: int) -> torch.Tensor:
        """ Apply polynomial formula based on the boolean expression that defines an erosion on each 3x3x3 overlapping cubes of the 3D image. """
        if connectivity == 6:
            vox = [8, 10, 12, 25, 16, 14, 26]
        elif connectivity == 18:
            vox = [8, 10, 12, 25, 16, 14, 1, 3, 5, 7, 9, 11, 13, 15, 18, 20, 22, 24, 26]
        else:
            vox = [8, 10, 12, 25, 16, 14, 1, 3, 5, 7, 9, 11, 13, 15, 18, 20, 22, 24, 0, 2, 4, 6, 17, 19, 21, 23, 26]
            
        return torch.prod(n[:, :, :, vox], dim=-1)

    def forward(self, im: torch.Tensor, iterations: int = 1, connectivity: int = 26) -> torch.Tensor:
        """ Forward pass for the SoftErosion3D module. """
        temp = torch.zeros(im.size(), device=im.device)
        for _ in range(iterations):
            unfolded = F.pad(im, (1, 1, 1, 1, 1, 1), mode='constant', value=1)
            unfolded = unfolded.unfold(2, 3, 1).unfold(3, 3, 1).unfold(4, 3, 1)
            unfolded = unfolded.contiguous().view(im.shape[0], im.shape[1], (im.shape[2]*im.shape[3]*im.shape[4]), (3**3))
            
            # Apply the formula to all windows simultaneously
            unfolded = unfolded[
                :,
                :,
                :,
                (self._indices_list[0][:, 0] * 9)
                + (self._indices_list[0][:, 1] * 3)
                + self._indices_list[0][:, 2],
            ]
            output = self.allcondArithm(unfolded, connectivity)
            # Adjust the dimensions of output to match the spatial dimensions of im
            output = output.view(output.size(0), output.size(1), im.shape[2], im.shape[3], im.shape[4])
            # Element-wise multiplication
            temp += im * -1
            im = im * output
            if self.doyoutan:
                im = (self.tan((im-(1/2))*self.power)+1)/2
        return temp


class PositiveDistanceMap(nn.Module):
    def __init__(self, iters: int = 30, connec: int = 26, tan: bool = False):
        super(PositiveDistanceMap, self).__init__()
        self.iters = iters
        self.connec = connec
        self.erode = SoftErosion3D(tan=tan)
        
    def forward(self, im: torch.Tensor) -> torch.Tensor:
        # Convert boolean tensor to float before subtraction
        im_float = im.float() if im.dtype == torch.bool else im
        # Now use the float tensor for the operation
        dils = self.erode((1-im_float), iterations=self.iters, connectivity=self.connec)*-1  # Corresponds to the positive distance d+
        return dils


class LAHLoss(nn.Module):
    def __init__(
        self,
        do_bg: bool = False,
        to_onehot_y: bool = False,
        apply_nonlin: Optional[Callable] = None,
        batch_lah: bool = False,
        iters: int = 2,
        connec: int = 26,
        tan: bool = False,
        ddp: bool = False,
        norm: int = 0
    ) -> None:
        """ 
        Initializes the LAHLoss module.
        
        Args:
            do_bg: Whether to include background in the loss calculation
            to_onehot_y: Whether to convert target to one-hot encoding
            apply_nonlin: Optional non-linearity to apply to input
            batch_lah: Whether to compute loss over batch dimension
            iters: Number of iterations for distance map computation
            connec: Connectivity for distance map computation
            tan: Whether to apply tanh activation
            ddp: Whether using distributed data parallel
        """
        super(LAHLoss, self).__init__()
        if apply_nonlin is not None and not callable(apply_nonlin):
            raise TypeError(f"apply_nonlin must be None or callable but is {type(apply_nonlin).__name__}.")
        
        self.do_bg = do_bg
        self.to_onehot_y = to_onehot_y
        self.apply_nonlin = apply_nonlin
        self.ddp = ddp
        self.batch_lah = batch_lah
        self.norm = norm
        
        self.dm = PositiveDistanceMap(iters, connec, tan)
        self.eps = 1e-6

    def _create_one_hot(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Create one-hot encoding of target tensor."""
        if x.ndim != y.ndim:
            y = y.view((y.shape[0], 1, *y.shape[1:]))
        
        if x.shape == y.shape:
            # If shapes match, y is probably already one-hot encoded
            return y
        
        # Create one-hot encoding
        y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
        y_onehot.scatter_(1, y.long(), 1)
        return y_onehot
    
    def logsum2(self,x, y):
        # return torch.log(torch.exp(x/100)+torch.exp(y/100))*100
        return (x+y+torch.abs(x-y))/2

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """  Forward pass for LAH loss computation."""
        shp_x = x.shape
        # breakpoint()
        
        axes = [0] + list(range(2, len(shp_x))) if self.batch_lah else list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        with torch.no_grad():
            y_onehot = self._create_one_hot(x, y)
        
        # breakpoint()
        
        dm_hat = self.dm(x)
        dm_gt = self.dm(y_onehot)
        
        x_for_loss = x
        y_onehot_for_loss = y_onehot
        if loss_mask is not None:
            x_for_loss = x * loss_mask
            y_onehot_for_loss = y_onehot * loss_mask

        # fp = (dm_gt * x).sum(axes) / (x.sum(axes) + self.eps)
        # fn = (dm_hat * y_onehot).sum(axes) / (y_onehot.sum(axes) + self.eps)
        
        fp = (dm_gt * x_for_loss).sum(axes) 
        fn = (dm_hat * y_onehot_for_loss).sum(axes)
        
        if self.norm == 0:
            fp = fp/(x_for_loss.sum(axes) + self.eps)
            fn = fn/(y_onehot_for_loss.sum(axes) + self.eps)
        elif self.norm == 1:
            denominator = (((x_for_loss.sum(axes)) + (y_onehot_for_loss.sum(axes)))/2) + self.eps
            fp = fp/denominator
            fn = fn/denominator
        elif self.norm == 2:
            denominator = (x_for_loss * y_onehot_for_loss.float()).sum(axes) + self.eps
            fp = fp/denominator
            fn = fn/denominator
        elif self.norm == 3:
            denominator = y_onehot_for_loss.sum(axes) + self.eps
            fp = fp/denominator
            fn = fn/denominator

        
        lah_loss = self.logsum2(fp, fn)
        
        # breakpoint()
        if not self.do_bg:
            if self.batch_lah:
                lah_loss = lah_loss[1:]
            else:
                lah_loss = lah_loss[:, 1:]
        
        # breakpoint()
        return lah_loss.mean()

