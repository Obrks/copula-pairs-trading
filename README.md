# Copula Pairs Trading

This project implements a mixed-copula pairs trading strategy adapted to hourly market data, inspired by the paper:

- Sabino da Silva, Ziegelmann & Caldeira (2017), *Mixed Copula Pairs Trading Strategy on the S&P 500*

The notebook and supporting code build rolling formation/trading windows, fit marginal GARCH models, estimate copula dependence, compute mispricing indices, and generate trading signals for a small set of equity pairs.

## Project structure

- `copula_pairs.ipynb` — main analysis notebook with data loading, cleaning, pair construction, GARCH filtering, copula fitting, and signal generation
- `copulas.py` — copula classes and fitting utilities (Gaussian, Student-t, Clayton, Frank, Gumbel, and mixed copulas)
- `data/` — cached parquet data downloaded from Yahoo Finance
- `sources/` — source papers and reference materials (not included in the repo)

## Strategy summary

The workflow is:

1. Download hourly OHLCV data for each ticker from Yahoo Finance
2. Align and clean each pair on local trading-session time
3. Compute log returns and flag the first open bar of each day (overnight gap)
4. Fit a GARCH(1,1) model with Student-t innovations to each series in each formation window
5. Standardize residuals and transform them to PIT values via the formation-window ECDF
6. Fit candidate copulas and select the best by log-likelihood (or AIC/BIC)
7. Compute conditional copula mispricing indices
8. Accumulate cumulative mispricing signals and generate pair trading positions

The notebook currently focuses on:

- `AAPL` / `GOOGL` (New York)
- `MC.PA` / `RMS.PA` (Paris)

## Dependencies

Install the required Python packages:

```bash
pip install yfinance pandas numpy matplotlib pyarrow arch statsmodels scipy
```

## Running the notebook

From the project directory:

```bash
jupyter notebook copula_pairs.ipynb
```

Or open the notebook in VS Code / Jupyter and run the cells in order.

## Notes

- Yahoo Finance intraday `60m` data is capped at roughly 730 days, so the project uses explicit start/end windows and caches downloaded data to `data/`.
- The overnight bar is excluded from the copula fitting process to avoid distorting the intraday dependence structure.
- The implementation is designed for research / backtesting rather than production deployment.

## Files of interest

- `copulas.py` contains the core copula estimation logic
- `copula_pairs.ipynb` contains the end-to-end pipeline and diagnostics
- `sources/` contains the academic references used for the methodology

## Typical usage

```python
from copulas import fit_copulas
from scipy import stats

# Example usage pattern inside the notebook
# fit_copulas(u, v, select_by="loglik")
```

## License

This project is for academic and research use within the course context and is not a commercial product.
