import torch
from torch import Tensor


def draw_gaussian(fmap: torch.Tensor, center: torch.Tensor, radius: int, sigma_factor: int, k=1):
    diameter = 2 * radius + 1
    gaussian = get_gaussian((radius, radius), sigma=diameter / sigma_factor, device=fmap.device)

    x, y = int(center[0]), int(center[1])
    height, width = fmap.shape[:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_fmap = fmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
        masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
        fmap[y - top : y + bottom, x - left : x + right] = masked_fmap


def get_gaussian(radius: tuple[int, int], sigma: float, device: torch.device):
    m, n = radius
    x = torch.arange(-m, m + 1, device=device) ** 2
    x.unsqueeze_(0)

    y = torch.arange(-n, n + 1, device=device) ** 2
    y.unsqueeze_(0)

    eps = 1e-7
    gauss = torch.exp(-(x.T + y) / (2 * sigma * sigma))

    # gauss = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gauss[gauss < eps * gauss.max()] = 0

    return gauss


def get_radius(width: Tensor, height: Tensor, min_overlap: float) -> Tensor:
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return torch.min(r1, torch.min(r2, r3))
