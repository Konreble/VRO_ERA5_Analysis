'''
Functions for loading .grib data and reducing size and converting to a single csv with daily value
'''

import pandas as pd
import xarray as xr

RAW_DATA_PATH = "../data/raw/Sicilly hourly/"

PROCESSED_DATA_PATH = "../data/processed/Sicilly hourly/"

LSM_PATH = "../data/raw/Sicilly hourly/Sicilly_hourly_land_sea_mask.nc"

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

RAW_DATA_FILENAMES = [name + ".grib" for name in FILENAMES]
def process_sicily_grib(
        grib_path: str, lsm_path: str, output_csv_path: str
) -> pd.DataFrame:
    # 1. Load the Land-Sea Mask
    lsm_ds = xr.open_dataset(lsm_path)
    lsm_var = "lsm" if "lsm" in lsm_ds else list(lsm_ds.data_vars.keys())[0]
    land_mask = lsm_ds[lsm_var] > 0.5

    # 2. Open t2m (Analysis stream)
    ds_t2m = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"dataType": "an"},
        },
    )
    print("t2m data variables:", list(ds_t2m.data_vars.keys()))
    print("t2m dimensions:", ds_t2m.dims)

    # 3. Open tp (Forecast stream)
    ds_tp = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"dataType": "fc"},
        },
    )
    print("tp data variables:", list(ds_tp.data_vars.keys()))
    print("tp dimensions:", ds_tp.dims)

    # --- 1. Process t2m ---
    t2m_land = ds_t2m["t2m"].where(land_mask)
    t2m_mean = t2m_land.mean(dim=["latitude", "longitude"])
    df_t2m = t2m_mean.to_dataframe().reset_index()

    # For analysis data, 'time' is already the hourly timestamp
    df_t2m["timestamp"] = pd.to_datetime(df_t2m["time"])
    df_t2m = df_t2m[["timestamp", "t2m"]].dropna()

    print("created df_t2m with shape: ", df_t2m.shape, " and columns: ", df_t2m.columns)
    print("Unique timestamps in t2m:", df_t2m["timestamp"].nunique())

    # 5. Mask water and compute spatial mean for tp
    tp_land = ds_tp["tp"].where(land_mask)
    tp_mean = tp_land.mean(dim=["latitude", "longitude"])
    df_tp = tp_mean.to_dataframe().reset_index()

    # For forecast data: base run time + forecast step = valid hourly timestamp
    # Ensure 'step' is a timedelta and 'time' is a datetime
    df_tp["timestamp"] = pd.to_datetime(df_tp["time"]) + pd.to_timedelta(
        df_tp["step"]
    )
    df_tp = df_tp[["timestamp", "tp"]].dropna()

    print("created df_tp with shape: ", df_tp.shape, " and columns: ", df_tp.columns)
    print("Unique timestamps in tp:", df_tp["timestamp"].nunique())

    # 6. Merge both variables on the timestamp
    combined_df = pd.merge(df_t2m, df_tp, on="timestamp", how="inner")
    combined_df = combined_df.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"]
    )
    print("completed merge. Shape: ", combined_df.shape)
    print(combined_df.head())

    # 7. Save intermediate CSV
    combined_df.to_csv(output_csv_path, index=False)
    print(
        f"Successfully created {output_csv_path} with shape {combined_df.shape}"
    )

    # Clean up file handles
    ds_t2m.close()
    ds_tp.close()
    lsm_ds.close()

    return combined_df
def load_era5_hourly():
    for name in FILENAMES:
        print("loading data from: ", RAW_DATA_PATH, name , ".grib")
        process_sicily_grib(
            RAW_DATA_PATH + name + ".grib",
            LSM_PATH,
            PROCESSED_DATA_PATH + name + "_hourly.csv")

