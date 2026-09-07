from pathlib import Path
import logging
from src.prepare_hourly_csv import load_era5_hourly
from src.Sicily_drought_analyzer import SicilyDroughtAnalyzer


DEBUG = False
FORCE_REBUILD = False

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(levelname)s: %(message)s",
)

RAW_DIR = Path("data/raw/Sicily hourly")
PROCESSED_DIR = Path("data/processed/Sicily hourly")
LSM_PATH = RAW_DIR / "Sicily_hourly_land_sea_mask.nc"
FIGURES_DIR = Path("output/figures")

FILE_STEMS = [
    "ERA5_1940-1943",
    "ERA5_1944-1949",
    "ERA5_1950-1955",
    "ERA5_1956-1961",
    "ERA5_1962-1967",
    "ERA5_1968-1973",
    "ERA5_1974-1979",
    "ERA5_1980-1985",
    "ERA5_1986-1991",
    "ERA5_1992-1997",
    "ERA5_1998-2003",
    "ERA5_2004-2009",
    "ERA5_2010-2015",
    "ERA5_2016-2020",
    "ERA5_2021-2025",
]


def main():
    df = load_era5_hourly(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        lsm_path=LSM_PATH,
        file_stems=FILE_STEMS,
        force_rebuild=FORCE_REBUILD,
    )

    analyzer = SicilyDroughtAnalyzer(df)

    analyzer.plot_monthly_spi(scale=3, save_path=FIGURES_DIR / "spi_3_anomaly.svg")
    analyzer.plot_monthly_spi(scale=12, save_path=FIGURES_DIR / "spi_12_anomaly.svg")
    analyzer.plot_annual_spi(save_path=FIGURES_DIR / "spi_annual.svg")
    analyzer.plot_annual_trends(save_path=FIGURES_DIR / "climate_trends.svg")


if __name__ == "__main__":
    main()