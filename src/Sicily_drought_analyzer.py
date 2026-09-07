import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from src.indices import (
    calculate_monthly_spi,
    calculate_annual_spi
)

logger = logging.getLogger(__name__)


class SicilyDroughtAnalyzer:

    def __init__(self, hourly_df: pd.DataFrame, default_spi_scales=(3, 6, 12)):
        if "timestamp" in hourly_df.columns:
            self.hourly = hourly_df.set_index("timestamp")
        else:
            self.hourly = hourly_df

        self.monthly = (
            self.hourly.resample("MS")
            .agg({"temperature_degC": "mean", "precipitation_mm": "sum"})
            .round(2)
        )
        self.yearly = (
            self.monthly.resample("YS")
            .agg({"temperature_degC": "mean", "precipitation_mm": "sum"})
            .round(2)
        )
        self.yearly["year"] = self.yearly.index.year
        self.yearly["SPI_annual"] = calculate_annual_spi(
            self.yearly["precipitation_mm"]
        )

        for scale in default_spi_scales:
            self.add_spi(scale)

    def add_spi(self, scale: int = 3):
        """Calls your standalone SPI calculation and attaches the column to monthly data."""
        col_name = f"SPI_{scale}"
        self.monthly[col_name] = calculate_monthly_spi(
            self.monthly["precipitation_mm"], scale=scale
        )

    def plot_monthly_spi(self, scale: int = 3, figsize=(12, 6), save_path: str | Path = None):
        """Concise SPI anomaly plot with optional file export."""
        s = self.monthly[f"SPI_{scale}"].dropna()
        fig, ax = plt.subplots(figsize=figsize)

        ax.fill_between(s.index, 0, s, where=(s >= 0), color="royalblue", alpha=0.8)
        ax.fill_between(s.index, 0, s, where=(s < 0), color="crimson", alpha=0.8)

        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(-1.0, color="goldenrod", ls="--", lw=1, label="Moderate (-1.0)")
        ax.axhline(-2.0, color="darkred", ls="--", lw=1, label="Extreme (-2.0)")

        ax.set(title=f"Sicily SPI-{scale} (1940–2025)", ylabel="SPI", ylim=(-3.5, 3.5))
        ax.legend(loc="lower left", framealpha=0.7)
        fig.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved monthly SPI-{scale} plot to: {save_path.resolve()}")

        return fig, ax

    def plot_annual_spi(
            self,
            extreme_threshold: float = -1.5,
            figsize=(12, 6),
            save_path: str | Path = None,
    ):
        """Annual SPI chart highlighting extreme drought years with distinct color and labels."""
        s = self.yearly["SPI_annual"]
        years = self.yearly["year"]

        fig, ax = plt.subplots(figsize=figsize)

        # 1. Map 3 distinct visual tiers
        colors = []
        for val in s:
            if val <= extreme_threshold:
                colors.append(
                    "#8b0000"
                )  # Dark/Deep Red for extreme/severe drought
            elif val < 0:
                colors.append("#d95f02")  # Warm orange/coral for mild-moderate
            else:
                colors.append("#2b83ba")  # Blue for wet/normal years

        # 2. Draw bars
        bars = ax.bar(years, s, color=colors, width=0.8, alpha=0.9)

        # 3. Add automatic text callouts for extreme years
        for year, val in zip(years, s):
            if val <= extreme_threshold:
                ax.annotate(
                    f"{year}\n({val:.1f})",
                    xy=(year, val),
                    xytext=(0, -6),  # Offset below the bar
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=7.5,
                    fontweight="bold",
                    color="#8b0000",
                )

        # 4. Reference lines & thresholds
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(
            -1.0, color="goldenrod", ls=":", lw=1, label="Moderate (-1.0)"
        )
        ax.axhline(
            extreme_threshold,
            color="#8b0000",
            ls="--",
            lw=1.2,
            label=f"Extreme Drought ({extreme_threshold})",
        )

        ax.set(
            title=f"Sicily Annual Standardized Precipitation Index (1940–2025)",
            ylabel="SPI",
            ylim=(-3.8, 3.2),  # Extra bottom headroom for year labels
            xlim=(years.min() - 1, years.max() + 1),
        )

        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", framealpha=0.85)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved annual SPI plot to: {save_path.resolve()}")

        return fig, ax

    def plot_annual_trends(self, figsize=(13, 4), save_path: str | Path = None):
        """2-panel annual trends showing yearly values and 10-year rolling averages."""
        fig, (ax_p, ax_t) = plt.subplots(1, 2, figsize=figsize)

        p = self.yearly["precipitation_mm"]
        t = self.yearly["temperature_degC"]

        # --- Precipitation Panel ---
        ax_p.plot(self.yearly["year"], p, color="#a6c8e0", marker="o", markersize=2.5, label="Annual Total")
        ax_p.plot(self.yearly["year"], p.rolling(10, center=True).mean(), color="#1f77b4", lw=2,
                  label="10-yr Moving Avg")
        ax_p.set(title="Annual Precipitation (mm)", ylabel="Rainfall (mm)")
        ax_p.grid(True, linestyle=":", alpha=0.6)
        ax_p.legend(framealpha=0.8)

        # --- Temperature Panel ---
        ax_t.plot(self.yearly["year"], t, color="#f4a582", marker="o", markersize=2.5, label="Annual Mean")
        ax_t.plot(self.yearly["year"], t.rolling(10, center=True).mean(), color="#d62728", lw=2,
                  label="10-yr Moving Avg")
        ax_t.set(title="Mean Annual Temperature (°C)", ylabel="Temperature (°C)")
        ax_t.grid(True, linestyle=":", alpha=0.6)
        ax_t.legend(framealpha=0.8)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved annual trends plot to: {save_path.resolve()}")

        return fig, (ax_p, ax_t)