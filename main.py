from pathlib import Path
from src.prepare_hourly_csv import load_era5_hourly
import logging

DEBUG = True

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(levelname)s: %(message)s",
)

RAW_DIR = Path("data/raw/Sicily hourly")
PROCESSED_DIR = Path("data/processed/Sicily hourly")
LSM_PATH = RAW_DIR / "Sicily_hourly_land_sea_mask.nc"

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
    print("Starting ERA5 Sicily Hourly Data Pipeline...")

    df = load_era5_hourly(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        lsm_path=LSM_PATH,
        file_stems=FILE_STEMS,
    )

    print(df.head())


if __name__ == "__main__":
    main()