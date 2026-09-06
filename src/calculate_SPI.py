"""
Pipeline for loading ERA5 GRIB data, applying a land-sea mask over Sicily,
and computing regional spatial means to create an hourly time series.
"""

from pathlib import Path
import pandas as pd
import xarray as xr

# ==========================================
# Configuration & Constants
# ==========================================
DEBUG = True

RAW_DIR = Path("../data/raw/Sicily hourly")
PROCESSED_DIR = Path("../data/processed/Sicily hourly")
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
    "ERA5_2021-2025"
]

# ==========================================
# Extraction & Processing Functions
# ==========================================
def load_land_mask(lsm_path: Path) -> xr.DataArray:
    """
    Loads the static NetCDF land-sea mask and returns a boolean mask
    where cells with >50% land coverage are True.
    """
    with xr.open_dataset(lsm_path) as ds:
        var_name = "lsm" if "lsm" in ds else list(ds.data_vars.keys())[0]
        land_mask = (ds[var_name] > 0.5).load()

    if DEBUG:
        print("Land mask loaded")
    return land_mask


def extract_variable_series(
    grib_path: Path,
    data_type: str,
    var_name: str,
    land_mask: xr.DataArray,
) -> pd.DataFrame:
    """
    Opens a specific ERA5 stream (analysis or forecast), masks out marine cells,
    computes the mean across all land cells, and standardizes timestamps.
    """
    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"dataType": data_type}},
    )

    if DEBUG:
        print("dataset opened")

    # Mask sea points and take spatial mean
    land_only = ds[var_name].where(land_mask)
    spatial_mean = land_only.mean(dim=["latitude", "longitude"])
    df = spatial_mean.to_dataframe().reset_index()

    if DEBUG:
        print("spatial mean calculated")

    # Reconstruct timestamps:
    # - 'an' (Analysis/t2m): 'time' is already the valid hourly timestamp
    # - 'fc' (Forecast/tp): valid time is forecast initialization ('time') + lead step ('step')
    if data_type == "fc" and "step" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"]) + pd.to_timedelta(df["step"])
    else:
        time_col = "valid_time" if "valid_time" in df.columns else "time"
        df["timestamp"] = pd.to_datetime(df[time_col])

    if DEBUG:
        print("timestamps constructed")

    df = df[["timestamp", var_name]].dropna()
    ds.close()
    return df


def process_single_chunk(
    grib_path: Path, land_mask: xr.DataArray
) -> pd.DataFrame:
    """
    Extracts both t2m and tp from a 6-year file and merges them on timestamp,
    """
    # Extract instantaneous temperature (Analysis stream)
    if DEBUG:
        print("Extracting temperature")
    df_t2m = extract_variable_series(
        grib_path, data_type="an", var_name="t2m", land_mask=land_mask
    )

    # Extract accumulated precipitation (Forecast stream)
    if DEBUG:
        print("Extracting precipitation")
    df_tp = extract_variable_series(
        grib_path, data_type="fc", var_name="tp", land_mask=land_mask
    )

    # Merge on matching hourly timestamps
    chunk_df = pd.merge(df_t2m, df_tp, on="timestamp", how="inner")
    chunk_df = chunk_df.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"]
    )

    if DEBUG:
        print(f"{grib_path} has been processed to a dataframe with shape: {chunk_df.shape}")
        print(f"Dates: {chunk_df["timestamp"].min()} - {chunk_df["timestamp"].max()}")
        print("Missing values: ", chunk_df.isna().sum())

    return chunk_df


# ==========================================
# Post-Processing & Feature Engineering
# ==========================================
def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts chronologically, removes duplicate timestamps, fills in missing values and converts units
    """
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    df["t2m"] = df["t2m"].interpolate(method="linear").bfill().ffill()
    df["tp"] = df["tp"].interpolate(method="linear").bfill().ffill()

    # convert units to more commonly used Celsius and millimeter
    df["temperature_degC"] = (df["temperature_degC"] - 273.15).round(2)
    df["tp"] = (df["precipitation_mm"] * 1000).round(4)

    # drop obsolete columns
    df = df.drop(columns=["t2m", "tp"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True)

    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["day_name"] = ts.dt.day_name()
    df["month"] = ts.dt.month
    df["date"] = ts.dt.date

    return df


# ==========================================
# Main Pipeline Runner
# ==========================================
def load_era5_hourly(*file_stems: str) -> pd.DataFrame:
    """
    Runs the end-to-end extraction, concatenation, and feature enrichment
    pipeline for all specified file stems.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load mask once into memory
    print(f"Loading Land-Sea Mask from {LSM_PATH.name}")
    land_mask = load_land_mask(LSM_PATH)

    data_chunks = []
    for stem in file_stems:
        grib_file = RAW_DIR / f"{stem}.grib"
        if DEBUG:
            print(f"Processing: {grib_file.name}")
        chunk_df = process_single_chunk(grib_file, land_mask)
        data_chunks.append(chunk_df)

    # Combine all chunks
    combined_df = pd.concat(data_chunks, ignore_index=True)

    # Clean, sort, and interpolate
    clean_df = clean_and_impute(combined_df)

    # Enrich with temporal features
    final_df = add_time_features(clean_df)

    # Export final full dataset
    final_output = PROCESSED_DIR / "ERA5_hourly_full.csv"
    final_df.to_csv(final_output, index=False)

    print(
        f"\nPipeline Complete! Final dataset saved to: {final_output}"
        f"\nRows: {len(final_df):,} | Columns: {list(final_df.columns)}"
        f"\nTime range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}"
    )

    return final_df

load_era5_hourly("ERA5_1974-1979", "ERA5_1980-1985")