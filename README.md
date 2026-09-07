# Sicily Climate & Drought Dynamics (1940–2025)

An end-to-end Python pipeline to process multi-decadal ERA5 reanalysis data, calculate Standardized Precipitation Index (SPI) metrics, and visualize long-term drought patterns in Sicily.


## Quickstart

# 1. Clone and set up environment
git clone [https://github.com/](https://github.com/)<your-username>/sicily-drought-analysis.git
cd sicily-drought-analysis
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the analysis and generate figures
python main.py


## Data & Replication

* **Processed Data (Default):** The cleaned hourly dataset (`data/processed/ERA5_hourly_full.csv`) is included in the repository. Running `python main.py` uses this file directly and finishes in under 5 seconds.
* **Raw GRIBs (Optional):** If you want to re-run the full extraction from raw ERA5 files and the land-sea mask, download them from [Link to Data / Releases / Drive], place them in `data/raw/Sicily hourly/`, and set `FORCE_REBUILD = True` in `main.py`.


## Project Structure

├── main.py                 # Pipeline execution & figure exports
├── prepare_hourly_csv.py   # GRIB extraction, land-sea masking, and CSV caching
├── drought_analyzer.py     # SicilyDroughtAnalyzer class & plotting methods
├── indices.py              # Pure SPI math (Gamma distribution fitting & Z-scores)
├── requirements.txt        # Dependencies (pandas, scipy, matplotlib, xarray)
├── data/                   # Raw and processed datasets
└── reports/figures/        # Generated figures (PNG)

## Key Outputs

Running the pipeline automatically saves figures to `reports/figures/`:

* `annual_spi.png`: Yearly drought anomalies with extreme dry years ($\le -1.5$) highlighted.
* `spi_3_anomaly.png` & `spi_12_anomaly.png`: Multi-scale monthly SPI time series.
* `climate_trends.png`: Annual precipitation and temperature trends with 10-year rolling averages.
