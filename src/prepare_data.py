'''
Functions for loading .grib data and reducing

WORKFLOW:
1. convert .grib file to csv
2. concatenate in chronological order
3. find average of ERA5 grid data for the whole region to reduce to one value per time point
4. reduce dataset by combining hourly data to daily average value
5. rename columns
6. convert timestamp
7. handle missing values
'''

import pandas as pd
import xarray as xr

DATA_PATH = "*/data"

FILENAMES = [
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

RAW_DATA_NAMES = [name + ".grib" for name in FILENAMES]

def process_sicily_grib(
    grib_path: str, lsm_path: str, output_parquet_path: str
):
    # 1. Load the Land-Sea Mask GRIB
    lsm_ds = xr.open_dataset(lsm_path, engine="cfgrib")
    lsm_var = "lsm" if "lsm" in lsm_ds else list(lsm_ds.data_vars.keys())[0]
    land_mask = lsm_ds[lsm_var] > 0.5

    # 2. Open the multi-year ERA5 chunk
    # Using open_mfdataset handles multi-variable GRIBs safely
    try:
        ds = xr.open_dataset(grib_path, engine="cfgrib")
    except ValueError:
        ds = xr.open_mfdataset(grib_path, engine="cfgrib", combine="by_coords")

    # 3. Mask out the Mediterranean sea
    land_only = ds.where(land_mask)

    # 4. Spatial mean across Sicily land cells
    spatial_mean = land_only.mean(dim=["latitude", "longitude"])

    # 5. Convert to clean DataFrame
    df = spatial_mean.to_dataframe().reset_index()

    # Standardize time
    time_col = "valid_time" if "valid_time" in df.columns else "time"
    df = df.rename(columns={time_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Drop non-variable spatial coordinates
    drop_cols = [
        c
        for c in ["latitude", "longitude", "surface", "step", "number"]
        if c in df.columns
    ]
    df = df.drop(columns=drop_cols)

    # 6. Save intermediate file
    df.to_parquet(output_parquet_path, index=False)
    print(f"Processed: {output_parquet_path}")

    ds.close()
    lsm_ds.close()

def load_data():
    for file in RAW_DATA_NAMES:

