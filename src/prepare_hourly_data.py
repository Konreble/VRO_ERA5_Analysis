'''
Functions for loading .grib data, reducing size and converting to a single csv with hourly values
'''

from pathlib import Path
import pandas as pd
import xarray as xr

DEBUG = True    # set to True to enable more detailed information about the progress and data loaded

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

ORIGINAL_COLUMNS = ["timestamp", "t2m", "tp"]

RENAME_MAP = {
    "t2m" : "temperature_K",
    "tp"  : "precipitation"
}

def grib_to_linear_csv(
        grib_path: str,
        lsm_path: str,
        output_csv_path: str
) -> pd.DataFrame:

    lsm_ds = xr.open_dataset(lsm_path)
    land_mask = lsm_ds["lsm"] > 0.5

    ds_t2m = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"dataType": "an"},
        },
    )
    if DEBUG:
        print("t2m data variables:", list(ds_t2m.data_vars.keys()))
        print("t2m dimensions:", ds_t2m.dims)

    ds_tp = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"dataType": "fc"},
        },
    )
    if DEBUG:
        print("tp data variables:", list(ds_tp.data_vars.keys()))
        print("tp dimensions:", ds_tp.dims)

    t2m_land = ds_t2m["t2m"].where(land_mask)
    t2m_mean = t2m_land.mean(dim=["latitude", "longitude"])
    df_t2m = t2m_mean.to_dataframe().reset_index()

    df_t2m["timestamp"] = pd.to_datetime(df_t2m["time"])
    df_t2m = df_t2m[["timestamp", "t2m"]].dropna()

    if DEBUG:
        print("created df_t2m with shape: ", df_t2m.shape, " and columns: ", df_t2m.columns)
        print("Unique timestamps in t2m:", df_t2m["timestamp"].nunique())

    tp_land = ds_tp["tp"].where(land_mask)
    tp_mean = tp_land.mean(dim=["latitude", "longitude"])
    df_tp = tp_mean.to_dataframe().reset_index()

    df_tp["timestamp"] = pd.to_datetime(df_tp["time"]) + pd.to_timedelta(df_tp["step"])
    df_tp = df_tp[["timestamp", "tp"]].dropna()

    if DEBUG:
        print("created df_tp with shape: ", df_tp.shape, " and columns: ", df_tp.columns)
        print("Unique timestamps in tp:", df_tp["timestamp"].nunique())

    combined_df = pd.merge(df_t2m, df_tp, on="timestamp", how="inner")
    combined_df = combined_df.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"]
    )
    if DEBUG:
        print("completed merge. Shape: ", combined_df.shape)
        print(combined_df.head())

    combined_df.to_csv(output_csv_path, index=False)
    if DEBUG:
        print(f"Successfully created {output_csv_path} with shape {combined_df.shape}")

    ds_t2m.close()
    ds_tp.close()
    lsm_ds.close()

    return combined_df

def load_era5_hourly(*files):
    data_chunks = []
    for name in files:
        print("loading data from: ", RAW_DATA_PATH, name , ".grib")
        df = grib_to_linear_csv(
            RAW_DATA_PATH + name + ".grib",
            LSM_PATH,
            PROCESSED_DATA_PATH + name + "_hourly.csv"
        )
        print(RAW_DATA_PATH, name , ".grib loaded successfully. Shape: ", df.shape)

        data_chunks.append(df)

    full_df = pd.concat(data_chunks, ignore_index=True)
    if DEBUG:
        print("complete dataframe before cleaning created with shape: ", full_df.shape, " and size: ", full_df.size)

    full_df = full_df.rename(columns=RENAME_MAP)
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], utc=True)

    full_df["hour"] = full_df["timestamp"].dt.hour
    full_df["day_of_week"] = full_df["timestamp"].dt.dayofweek  # 0 = Monday
    full_df["day_name"] = full_df["timestamp"].dt.day_name()
    full_df["month"] = full_df["timestamp"].dt.month
    full_df["date"] = full_df["timestamp"].dt.date

    full_df["temperature_K"] = full_df["temperature_K"].interpolate().ffill().bfill()
    full_df["precipitation"] = full_df["precipitation"].interpolate().ffill().bfill()

    full_df["temperature_K"] = full_df["temperature_K"].round(2)
    full_df["precipitation"] = full_df["precipitation"].round(8)


    full_df.to_csv(PROCESSED_DATA_PATH + "ERA5_hourly_raw_csv")
    if DEBUG:
        print("saved file as: ", PROCESSED_DATA_PATH, "ERA5_hourly.csv")


load_era5_hourly("ERA5_1980-1985", "ERA5_1974-1979")