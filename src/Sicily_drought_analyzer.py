import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from scipy import stats
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

    def plot_year_diagnostic(
            self, year: int = 1977, figsize=(12, 9), save_path: str | Path = None
    ):
        """Optimized diagnostic: cumulative deficit, monthly hyetograph, and multi-scale SPI."""
        df_yr = self.monthly[self.monthly.index.year == year]
        if df_yr.empty:
            raise ValueError(f"No records found for year {year}")

        # Vectorized 1940-2025 monthly baselines
        m_idx = self.monthly.index.month
        clim_p = self.monthly.groupby(m_idx)["precipitation_mm"].mean()
        clim_t = self.monthly.groupby(m_idx)["temperature_degC"].mean()

        months = np.arange(1, 13)
        month_labels = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # 1. Cumulative Deficit
        cum_clim = clim_p.cumsum()
        cum_yr = df_yr["precipitation_mm"].cumsum()
        ax1.plot(months, cum_clim, "k--", lw=1.8, label="1940–2025 Normal")
        ax1.plot(
            months,
            cum_yr,
            color="#b2182b",
            marker="o",
            lw=2,
            label=f"{year} ({cum_yr.iloc[-1]:.0f} mm)",
        )
        ax1.fill_between(
            months, cum_yr, cum_clim, color="#b2182b", alpha=0.15, label="Deficit"
        )
        ax1.set(
            title=f"Sicily Drought Anatomy: {year} (Total Deficit: {cum_clim.iloc[-1] - cum_yr.iloc[-1]:.0f} mm)",
            ylabel="Cum. Rain (mm)",
        )
        ax1.legend(loc="upper left")
        ax1.grid(True, ls=":", alpha=0.6)

        # 2. Monthly Rainfall + Temperature Anomaly (Twin Axis)
        w = 0.35
        ax2.bar(
            months - w / 2,
            clim_p,
            width=w,
            color="#cbd5e1",
            label="Climatological Normal",
        )
        ax2.bar(
            months + w / 2,
            df_yr["precipitation_mm"],
            width=w,
            color="#b2182b",
            label=f"{year} Rain",
        )
        ax2.set_ylabel("Rainfall (mm)")

        ax2_t = ax2.twinx()
        t_anom = df_yr["temperature_degC"].values - clim_t.values
        ax2_t.plot(
            months,
            t_anom,
            color="#d95f02",
            marker="s",
            lw=1.5,
            label="Temp Anomaly (°C)",
        )
        ax2_t.axhline(0, color="gray", ls="--", lw=0.8)
        ax2_t.set_ylabel("Temp Anomaly (°C)", color="#d95f02")
        ax2.legend(loc="upper left")
        ax2_t.legend(loc="upper right")

        # 3. Multi-Scale SPI Progression
        spi_cols = [c for c in ["SPI_3", "SPI_6", "SPI_12"] if c in df_yr.columns]
        for col in spi_cols:
            ax3.plot(months, df_yr[col], marker="o", lw=1.8, label=col)

        ax3.axhline(0, color="black", lw=0.8)
        ax3.axhline(-1.0, color="goldenrod", ls=":", label="Moderate (-1.0)")
        ax3.axhline(-1.5, color="#d95f02", ls="--", label="Severe (-1.5)")
        ax3.axhline(-2.0, color="#8b0000", ls="--", label="Extreme (-2.0)")
        ax3.set(
            ylabel="SPI", xticks=months, xticklabels=month_labels, ylim=(-3.2, 1.8)
        )
        ax3.legend(loc="lower left", ncol=len(spi_cols) + 3, fontsize=8)
        ax3.grid(True, ls=":", alpha=0.6)

        fig.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved year diagnostics for {year} to: {save_path.resolve()}")
        return fig, (ax1, ax2, ax3)

    def plot_temp_precip_coupling(
            self, figsize=(13, 5), save_path: str | Path = None
    ):
        """Optimized coupling diagnostic: vectorized anomalies and group correlations."""
        # 1. Vectorized anomaly computation in place
        m = self.monthly.index.month
        df = self.monthly.assign(
            temp_anom=self.monthly["temperature_degC"]
                      - self.monthly.groupby(m)["temperature_degC"].transform("mean"),
            precip_anom=self.monthly["precipitation_mm"]
                        - self.monthly.groupby(m)["precipitation_mm"].transform("mean"),
        )

        # 2. Vectorized Pearson r by calendar month (no scipy loops)
        corrs = df.groupby(m).apply(
            lambda g: g["temp_anom"].corr(g["precip_anom"])
        )

        # 3. Map seasons cleanly
        season_lookup = {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Autumn",
            10: "Autumn",
            11: "Autumn",
        }
        df["season"] = m.map(season_lookup)

        fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=figsize)

        # Panel 1: Bar chart
        months = np.arange(1, 13)
        month_labels = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
        ax_bar.bar(
            months,
            corrs,
            color=np.where(corrs < 0, "#b2182b", "#2166ac"),
            edgecolor="black",
            lw=0.5,
        )
        ax_bar.axhline(0, color="black", lw=0.8)
        ax_bar.set_xticks(months)
        ax_bar.set_xticklabels(
            month_labels, rotation=0
        )
        ax_bar.set(
            title="Monthly Correlation (Temp vs. Precip Anomaly)",
            ylabel="Pearson r",
            ylim=(-0.75, 0.25),
        )
        ax_bar.grid(axis="y", ls=":", alpha=0.6)

        # Panel 2: Seasonal Scatter + Vectorized Trend Lines
        colors = {
            "Winter": "#2b83ba",
            "Spring": "#abdda4",
            "Summer": "#d7191c",
            "Autumn": "#fdae61",
        }
        for s_name, grp in df.groupby("season", sort=False):
            x, y = grp["precip_anom"], grp["temp_anom"]
            r = x.corr(y)
            ax_scatter.scatter(
                x, y, alpha=0.25, s=16, color=colors[s_name], label=f"{s_name} (r={r:.2f})"
            )

            # Quick linear fit without scipy
            slope, intercept = np.polyfit(x, y, 1)
            x_vals = np.array([x.min(), x.max()])
            ax_scatter.plot(x_vals, slope * x_vals + intercept, color=colors[s_name], lw=2)

        ax_scatter.axhline(0, color="gray", ls="--", lw=0.7)
        ax_scatter.axvline(0, color="gray", ls="--", lw=0.7)
        ax_scatter.set(
            title="Coupling by Season (1940–2025)",
            xlabel="Precipitation Anomaly (mm)",
            ylabel="Temperature Anomaly (°C)",
        )
        ax_scatter.legend(loc="upper right", framealpha=0.85)
        ax_scatter.grid(True, ls=":", alpha=0.5)

        fig.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved temperature precipitation correlation plot to: {save_path.resolve()}")

        return fig, (ax_bar, ax_scatter)

    def plot_longterm_correlation_matrix(self, figsize=(7, 6), save_path=None):
        data = self.yearly[
            ["temperature_degC", "precipitation_mm", "SPI_annual"]
        ].dropna()
        data.insert(0, "Year", data.index)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            data.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax,
        )
        ax.set_title("Climate & Drought Correlation (1940–2025)", pad=12)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved longterm correlation matrix to: {save_path.resolve()}")
        return fig, ax

    def plot_decadal_drought_frequency(
            self,
            spi_col: str = "SPI_12",
            figsize=(9, 6),
            save_path: str | Path = None,
    ):
        """Optimized decadal drought percentage breakdown by McKee severity tiers."""
        if spi_col not in self.monthly.columns:
            self.add_spi(scale=int(spi_col.split("_")[-1]))

        s = self.monthly[spi_col].dropna()
        decades = (s.index.year // 10 * 10).astype(str) + "s"

        # 1. Classify tiers & compute decadal percentage in 2 lines
        tier = pd.cut(
            s,
            bins=[-np.inf, -2.0, -1.5, -1.0, np.inf],
            labels=["Extreme", "Severe", "Moderate", "Normal"],
        )
        pct = (
                pd.crosstab(decades, tier, normalize="index") * 100
        )[["Moderate", "Severe", "Extreme"]]

        # 2. Render stacked bar chart via Pandas
        fig, ax = plt.subplots(figsize=figsize)
        pct.plot(
            kind="bar",
            stacked=True,
            color=["#fed976", "#fd8d3c", "#bd0026"],
            ax=ax,
            edgecolor="#333",
            lw=0.7,
            rot=0,
            width=0.6,
        )

        # 3. Label total drought % above each bar
        totals = pct.sum(axis=1)
        for idx, total in enumerate(totals):
            ax.text(
                idx,
                total + 0.6,
                f"{total:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

        ax.set(
            title=f"Decadal Drought Prevalence in Sicily ({spi_col}, 1940–2025)",
            ylabel="% of Months in Drought",
            ylim=(0, totals.max() * 1.2),
        )
        ax.legend(title="Tier", loc="upper left", framealpha=0.9)
        ax.grid(axis="y", ls=":", alpha=0.5)

        fig.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved decadal drought frequency plot to: {save_path.resolve()}")

        return fig, ax, pct