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
        dils = self.erode((1-im_float), iterations=self.iters, connectivity=self.connec)*-1  
        return dils


class DistanceMap(nn.Module):
    def __init__(self, iters: int = 30, connec: int = 26, tan: bool = False):
        super(DistanceMap, self).__init__()
        self.iters = iters
        self.connec = connec
        self.erode = SoftErosion3D(tan=tan)
        
    def forward(self, im: torch.Tensor) -> torch.Tensor:
        # Convert boolean tensor to float before subtraction
        im_float = im.float() if im.dtype == torch.bool else im
        # Now use the float tensor for the operation
        bg = self.erode((1-im_float), iterations=self.iters, connectivity=self.connec)*-1  
        fg = self.erode((im_float), iterations=self.iters, connectivity=self.connec)*-1  
        return bg+fg

class SignedDistanceMap(nn.Module):
    def __init__(self, iters: int = 30, connec: int = 26, tan: bool = False):
        super(DistanceMap, self).__init__()
        self.iters = iters
        self.connec = connec
        self.erode = SoftErosion3D(tan=tan)
        
    def forward(self, im: torch.Tensor) -> torch.Tensor:
        # Convert boolean tensor to float before subtraction
        im_float = im.float() if im.dtype == torch.bool else im
        # Now use the float tensor for the operation
        bg = self.erode((1-im_float), iterations=self.iters, connectivity=self.connec)*-1  
        fg = self.erode((im_float), iterations=self.iters, connectivity=self.connec)*-1  
        return bg-fg