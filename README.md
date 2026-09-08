# Sicily Climate & Drought Dynamics (1940–2025)

An end-to-end Python pipeline to process multi-decadal ERA5 reanalysis data, calculate Standardized Precipitation Index (SPI) metrics, and visualize long-term drought patterns in Sicily.

---

## Quickstart

```bash
# 1. Clone and set up environment
git clone --depth 1 https://github.com/Konreble/VRO_ERA5_Analysis.git
cd VRO_ERA5_Analysis
python -m venv venv

source venv/bin/activate  # Linux/Mac
venv\Scripts\activate # Windows

pip install -r requirements.txt

# 2. Run the analysis and generate figures
python main.py
```

---

## Data & Replication

- **Processed Data (Default):** The cleaned hourly dataset (`data/processed/ERA5_hourly_full.csv`) is included in the repository. Running `python main.py` uses this file directly and finishes in under 5 seconds.
- **Raw GRIBs (Optional):** To re-run the full extraction from raw ERA5 .grib files and the land-sea mask, download "Sicily.hourly.zip" from the release on GitHub, unpack and place them in `data/raw/Sicily hourly/`, and set `FORCE_REBUILD = True` in `main.py`.

---

## Project Structure

```text
VRO_ERA5_Analysis/
  README.md
  requirements.txt
  main.py
  src/
    __init__.py
    Sicily_drought_analyzer.py
    indices.py
    prepare_hourly_csv.py
  output/
    figures/
  data/
    raw/
      Sicily hourly/
        #place raw ERA5 data here
    processed/
      Sicily hourly/
        ERA5_hourly_full_csv
```

---

## Key Outputs

Running the pipeline automatically saves figures to `output/figures/`:
* `annual_spi.png`: Yearly drought anomalies with extreme dry years (<= -1.5) highlighted.
* `spi_3_anomaly.png` & `spi_12_anomaly.png`: Multi-scale monthly SPI time series.
* `climate_trends.png`: Annual precipitation and temperature trends with 10-year rolling averages.
