# Walmart Time Series Forecasting Project

This project forecasts:
- Daily total revenue (`Purchase_Amount` sum)
- Daily order count (transaction count)

It includes:
- Preprocessing and aggregation from transactional data
- Models: Naive, Seasonal Naive, Linear Regression, XGBoost, ARIMA, SARIMA, Holt, Holt-Winters (ETS)
- Rolling time-series evaluation with loss metrics (MAE, RMSE, MSE, MAPE)
- Visualizations and diagnostics
- Model and artifact saving

## Structure

- `data/Walmart_customer_purchases.csv`
- `notebooks/01_data_preprocessing.ipynb`
- `notebooks/02_classical_models.ipynb`
- `notebooks/03_ml_models.ipynb`
- `notebooks/04_evaluation_visualization.ipynb`
- `notebooks/05_model_saving_and_inference.ipynb`
- `models/` (saved fitted models)
- `artifacts/` (metrics, forecasts, plots)

## Quick Start

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Open and run notebooks in this order:

- `notebooks/01_data_preprocessing.ipynb`
- `notebooks/02_classical_models.ipynb`
- `notebooks/03_ml_models.ipynb`
- `notebooks/04_evaluation_visualization.ipynb`
- `notebooks/05_model_saving_and_inference.ipynb`

3. Outputs:

- `artifacts/classical_metrics_summary.csv`
- `artifacts/ml_metrics_summary.csv`
- `artifacts/all_metrics_summary.csv`
- `artifacts/best_models_table.csv`
- `artifacts/best_models_selected.json`
- `artifacts/classical_tuning_params.json`
- `artifacts/ml_tuning_params.json`
- `artifacts/final_forecasts.csv`
- `artifacts/final_model_registry.json`
- Plot files in `artifacts/`
- Serialized models in `models/`
