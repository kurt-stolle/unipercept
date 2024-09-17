"""
Impelements modules that smooth a value.
"""

from __future__ import annotations

import enum as E
import typing as T

import torch
import torch.distributions
import typing_extensions as TX
from torch import Tensor, nn


class SmoothingObserverModule(T.Protocol):
    """
    Protocol for modules that can be observed.
    """

    def reset(self) -> None: ...

    def observe(self, default: Tensor | None = None) -> Tensor: ...

    def forward(self, new_value: Tensor) -> Tensor: ...

    if T.TYPE_CHECKING:
        __call__ = forward


class MissingValueHandling(E.StrEnum):
    """
    Enum for handling missing values.
    """

    ZERO = E.auto()
    MEAN = E.auto()


class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) module.
    """

    alpha: T.Final[float]
    ema: Tensor

    def __init__(self, alpha=0.1, **kwargs):
        """
        Initialize the EMA module.

        Parameters
        ----------
        alpha: float
            The smoothing factor for the EMA.
        initial_value: float
            The initial value of the EMA.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.register_buffer("ema", torch.tensor(torch.nan))

    def reset(self):
        """
        Reset the EMA to NaN.
        """
        self.ema.fill_(torch.nan)

    @TX.override
    def forward(self, new_value: Tensor) -> Tensor:
        """
        Update the EMA with a new value and return the updated EMA.

        Parameters
        ----------
        new_value: Tensor[*]
            New value to update the EMA with.

        Returns
        -------
        Tensor
            The updated EMA.
        """
        if torch.isnan(self.ema):
            self.ema = new_value
        else:
            self.ema = self.alpha * new_value + (1 - self.alpha) * self.ema
        return self.ema

    def observe(self, default: Tensor | None = None) -> Tensor:
        """
        Observe the current EMA value.

        Returns
        -------
        Tensor
            The current EMA value.
        """
        if torch.isnan(self.ema) and default is not None:
            return default
        return self.ema

    if T.TYPE_CHECKING:
        __call__ = forward


class GMA(nn.Module):
    """
    Gaussian Moving Average (GMA) module.
    """

    window_size: T.Final[int]
    std_dev: T.Final[float]
    missing_values: T.Final[MissingValueHandling]
    weights: Tensor
    data_window: Tensor

    def __init__(
        self,
        window_size=5,
        std_dev=1.0,
        missing_values: MissingValueHandling = MissingValueHandling.ZERO,
        **kwargs,
    ):
        """
        Initialize the GMA module.

        Parameters
        ----------
        window_size: int
            The size of the moving window to apply the Gaussian filter.
        std_dev: float
            The standard deviation for the Gaussian distribution.
        """
        super().__init__(**kwargs)
        self.window_size = window_size
        self.std_dev = std_dev
        self.missing_values = missing_values

        # Calculate Gaussian weights
        with torch.no_grad():
            interval = (2 * std_dev + 1.0) / window_size
            x = torch.linspace(
                -std_dev - interval / 2.0, std_dev + interval / 2.0, window_size + 1
            )
            kernel = torch.distributions.Normal(
                torch.tensor([0.0]), torch.tensor([std_dev])
            )
            kern1d = torch.diff(kernel.cdf(x))
            weights = kern1d / kern1d.sum()

        # Register buffer for the weights and the data window
        self.register_buffer("weights", weights)
        self.register_buffer(
            "data_window", torch.full((window_size,), torch.nan, requires_grad=False)
        )

    def reset(self):
        """
        Reset the data window to NaN.
        """
        self.data_window.fill_(torch.nan)

    @TX.override
    def forward(self, new_value: Tensor) -> Tensor:
        """
        Update the data window with a new value, apply the Gaussian weights,
        and return the result.

        Args:
        new_value: torch.Tensor
            New value to add to the window.

        Parameters
        ----------
        Tensor
            The result of applying the Gaussian Moving Average.
        """

        # Update data window
        self.data_window = torch.roll(self.data_window, -1)
        self.data_window[-1] = new_value

        return self.observe()

    def observe(self, default: Tensor | None = None) -> Tensor:
        """
        Observe the current GMA value.

        Returns
        -------
        Tensor
            The current GMA value.
        """
        if torch.all(torch.isnan(self.data_window)) and default is not None:
            return default

        match self.missing_values:
            case MissingValueHandling.ZERO:
                data_window = torch.nan_to_num(self.data_window)
            case MissingValueHandling.MEAN:
                data_window = torch.where(
                    ~torch.isnan(self.data_window),
                    self.data_window,
                    torch.nanmean(self.data_window),
                )
            case _:
                msg = f"Unsupported missing value handling: {self.missing_values}"
                raise ValueError(msg)

        # Apply basic mean value imputation for NaN (initial) values
        data_window = torch.where(
            ~torch.isnan(self.data_window),
            self.data_window,
            torch.nanmean(self.data_window),
        )

        # Apply Gaussian weights
        return torch.dot(self.weights, data_window)

    if T.TYPE_CHECKING:
        __call__ = forward
