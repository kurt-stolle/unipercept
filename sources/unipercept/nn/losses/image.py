from typing import Final, override

import torch
from torch import nn

from unipercept.types import Tensor


class ReconstructionLoss(nn.Module):
    alpha: Final[float]

    def __init__(self, alpha=0.85):
        super().__init__()
        self.alpha = alpha
        self.ssim = SSIMLoss()
        self.l1 = nn.L1Loss(reduction="none")

    @override
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        ssim = self.ssim(output, target).mean(
            dim=1, keepdim=True
        )  # (N, 1, H, W), average channels
        l1 = self.l1(output, target).mean(
            dim=1, keepdim=True
        )  # (N, 1, H, W), average channels
        return (
            0.5 * self.alpha * ssim + (1 - self.alpha) * l1
        )  # NOTE: ssim is already (1-ssim)


class SSIMLoss(nn.Module):
    """
    Calculate the Structural Similarity Error between two vectors

    .. math::

        SSIM(x,y) = [l(x,y)]^a * [c(x,y)]^b * [s(x,y)]^y

    where l:luminance, c:contrast, s:structure

    According to [1], we set a=b=y=1, and then

    .. math::

        SSIM(x,y) = (2*m_x*m_y+c1)(2*s_xy+c2)/(m_x^2 + m_y^2 + c1)(s_x_2 _s_y^2 + c2)

    References
    ----------
    [1] Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale structural
        similarity for image quality assessment."
        Signals, Systems and Computers, 2004.
    """

    c1: Final[float]
    c2: Final[float]

    def __init__(self):
        super().__init__()

        bits_per_pixel = 8  # pixel values range between 0 and 255
        L = 2**bits_per_pixel - 1  # dynamic range of the pixel values
        k1 = 0.01  # by default
        k2 = 0.03  # by default
        self.c1 = (k1 * L) ** 2  # stabilize the division with weak denominator
        self.c2 = (k2 * L) ** 2  # purpose same as c1

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.refl_pad = nn.ReflectionPad2d(1)

    @override
    def forward(self, src: Tensor, tgt: Tensor):
        # Add 1 reflection padding in all boundaries of each output, and target.
        # NOTE: Padding because AvgPool2d will decrease the dim = (N, B, H, W) to dim = (N, B, H-1, W-1).
        # NOTE: Reflection because we use AvgPool2d to calculate the mean and (cov)variance on 3x3 blocks.
        src = self.refl_pad(src)
        tgt = self.refl_pad(tgt)

        # Calculate mean and variance for output.
        output_mu = self.avg_pool(src)  # pixel sample mean of output
        output_sigma = self.avg_pool(src**2) - output_mu  # variance of output

        # Calculate mean and variance for target.
        target_mu = self.avg_pool(tgt)  # pixel sample mean of target
        target_sigma = self.avg_pool(tgt**2) - target_mu  # variance of target

        # Calculate covvariance between output and target.
        output_target_sigma = (
            self.avg_pool(src * tgt) - output_mu * target_mu
        )  # covariance of output and target

        # Numerator of SSIM
        numerator = (2 * output_mu * target_mu + self.c1) * (
            2 * output_target_sigma + self.c2
        )
        # Denominator of SSIM
        denominator = (output_mu**2 + target_mu**2 + self.c1) * (
            output_sigma + target_sigma + self.c2
        )
        # Compute SSIM
        _ssim = numerator / denominator  # ssim \in [-1, 1]

        # Normalize SSIM
        norm_ssim = (1 - _ssim) / 2  # ssim \in [0, 1]

        # Clamp to avoid potential overflow.
        return torch.clamp(norm_ssim, min=0, max=1)
