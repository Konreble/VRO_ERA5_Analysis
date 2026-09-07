"""
Pipeline for loading ERA5 GRIB data, applying a land-sea mask over Sicily,
and computing regional spatial means to create an hourly time series.
"""

from pathlib import Path
import pandas as pd
import xarray as xr
import logging

logger = logging.getLogger(__name__)

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

    logger.debug("Land mask loaded")
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

    logger.debug("dataset opened")

    # Mask sea points and take spatial mean
    land_only = ds[var_name].where(land_mask)
    spatial_mean = land_only.mean(dim=["latitude", "longitude"])
    df = spatial_mean.to_dataframe().reset_index()

    logger.debug("spatial mean calculated")

    # Reconstruct timestamps:
    # - 'an' (Analysis/t2m): 'time' is already the valid hourly timestamp
    # - 'fc' (Forecast/tp): 'time' is in 12 hour chunks with 'step' specifying the hours in between
    if data_type == "fc":
        df["timestamp"] = pd.to_datetime(df["time"]) + pd.to_timedelta(df["step"])
    else:
        df["timestamp"] = pd.to_datetime(df["time"])

    logger.debug("timestamps constructed")

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
    logger.debug("Extracting temperature")
    df_t2m = extract_variable_series(
        grib_path, data_type="an", var_name="t2m", land_mask=land_mask
    )

    # Extract accumulated precipitation (Forecast stream)
    logger.debug("Extracting precipitation")
    df_tp = extract_variable_series(
        grib_path, data_type="fc", var_name="tp", land_mask=land_mask
    )

    # Merge on matching hourly timestamps
    chunk_df = pd.merge(df_t2m, df_tp, on="timestamp", how="inner")
    chunk_df = chunk_df.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"]
    )


    logger.debug(f"{grib_path} has been processed to a dataframe with shape: {chunk_df.shape}")
    logger.debug(f"Dates: {chunk_df["timestamp"].min()} - {chunk_df["timestamp"].max()}")
    logger.debug("Missing values: ", chunk_df.isna().sum())

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
    df["temperature_degC"] = (df["t2m"] - 273.15).round(2)
    df["precipitation_mm"] = (df["tp"] * 1000).round(4)

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
def load_era5_hourly(
        raw_dir: Path,
        processed_dir: Path,
        lsm_path: Path,
        file_stems: list[str],
        force_rebuild: bool = False,
) -> pd.DataFrame:

        """
        Runs the end-to-end extraction, concatenation, and feature enrichment and cleaning
        pipeline for all specified file stems.
        """
        processed_dir.mkdir(parents=True, exist_ok=True)
        final_output = processed_dir / "ERA5_hourly_full.csv"

        # check for existing csv
        if final_output.exists() and not force_rebuild:
            logger.info(f"Existing dataset {final_output.name} found. Loading data from file.")
            return pd.read_csv(final_output, parse_dates=["timestamp"])

        if force_rebuild:
            logger.info("FORCE_REBUILD is True. Regenerating dataset from raw GRIBs...")
        else:
            logger.info("No processed file found. Generating dataset from raw GRIBs...")

        # grib to csv pipeline
        logger.info(f"Loading Land-Sea Mask from {lsm_path.name}")
        land_mask = load_land_mask(lsm_path)

        data_chunks = []
        for stem in file_stems:
            grib_file = raw_dir / f"{stem}.grib"
            logger.info(f"Processing: {grib_file.name}")
            chunk_df = process_single_chunk(grib_file, land_mask)
            data_chunks.append(chunk_df)

        combined_df = pd.concat(data_chunks, ignore_index=True)
        clean_df = clean_and_impute(combined_df)
        final_df = add_time_features(clean_df)
        final_output = processed_dir / "ERA5_hourly_full.csv"
        final_df.to_csv(final_output, index=False)

        logger.info(
            f"\nPipeline Complete! Final dataset saved to: {final_output}"
            f"\nRows: {len(final_df):,} | Columns: {list(final_df.columns)}"
            f"\nTime range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}"
        )

        return final_df