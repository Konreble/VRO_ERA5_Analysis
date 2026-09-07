import numpy as np
import pandas as pd
from scipy import stats


def calculate_monthly_spi(
    precip_series: pd.Series,
    scale: int = 3,
    min_precip_threshold: float = 0.001,
) -> pd.Series:
    """
    Calculates the Standardized Precipitation Index (SPI) for a monthly precipitation series.

    Parameters
    ----------
    precip_series : pd.Series
        Monthly precipitation sums with a DatetimeIndex.
    scale : int
        Accumulation window in months (e.g., 3 for SPI-3, 12 for SPI-12).
    min_precip_threshold : float
        Threshold (in mm) below which monthly totals are treated as dry/zero.

    Returns
    -------
    pd.Series
        Standardized anomaly Z-scores aligned to the input series index.
    """

    rolling_precip = precip_series.rolling(window=scale).sum()
    spi_series = pd.Series(
        np.nan, index=precip_series.index, name=f"SPI_{scale}"
    )

    for month in range(1, 13):
        month_mask = (rolling_precip.index.month == month) & (
            rolling_precip.notna()
        )
        sub_series = rolling_precip[month_mask]

        if len(sub_series) < 10:
            continue

        # Separate zero-rain observations from positive rainfall
        is_zero = sub_series <= min_precip_threshold
        zeros_count = is_zero.sum()
        non_zeros = sub_series[~is_zero]
        total_count = len(sub_series)

        # Empirical probability of zero precipitation
        q = zeros_count / total_count

        # 3. Fit 2-parameter Gamma distribution to non-zero totals
        if len(non_zeros) > 1:
            # floc=0 fixes the location parameter at zero (standard Gamma)
            alpha, _, beta = stats.gamma.fit(non_zeros, floc=0)

            # Cumulative probability: H(x) = q + (1 - q) * G(x)
            gamma_cdf = stats.gamma.cdf(non_zeros, alpha, scale=beta)
            h_non_zero = q + (1.0 - q) * gamma_cdf

            # Clip probabilities slightly to avoid +/- inf when converting to Z-scores
            h_non_zero = np.clip(h_non_zero, 1e-6, 1.0 - 1e-6)
            spi_series.loc[non_zeros.index] = stats.norm.ppf(h_non_zero)

        # 4. Handle dry/zero periods
        if zeros_count > 0:
            # Average cumulative probability for the zero-rain mass
            h_zero = np.clip(q / 2.0 if q > 0 else 1e-6, 1e-6, 1.0 - 1e-6)
            spi_series.loc[sub_series[is_zero].index] = stats.norm.ppf(h_zero)

    return spi_series.round(2)


def calculate_annual_spi(annual_precip: pd.Series) -> pd.Series:
    """Calculates annual SPI by fitting a single Gamma distribution

    across all historical annual rainfall totals.
    """
    # Fit 2-parameter Gamma to all yearly totals
    alpha, _, beta = stats.gamma.fit(annual_precip, floc=0)

    # Compute CDF and transform to standard normal Z-scores
    cdf = stats.gamma.cdf(annual_precip, alpha, scale=beta)
    cdf = np.clip(cdf, 1e-6, 1.0 - 1e-6)

    return pd.Series(
        stats.norm.ppf(cdf), index=annual_precip.index, name="SPI_annual"
    ).round(2)