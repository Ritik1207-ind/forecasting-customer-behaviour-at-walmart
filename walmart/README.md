# 🛒 Walmart Time-Series Forecasting: Customer Behaviour Analysis

This high-performance project aims to forecast daily revenue and order volumes for Walmart transactional data using a hybrid approach of **Statistical Classical Models** and **Advanced Machine Learning Regressors**.

## 🌟 Key Features
- **Comprehensive Model Suite**: Evaluates 10+ models including ARIMA, SARIMA, XGBoost, LightGBM, CatBoost, and Random Forest.
- **Recursive Forecasting**: Implements multi-step recursive prediction with dynamic feature engineering (lags, rolling stats).
- **Advanced Visualizations**: Detailed model leaderboards, feature importance analysis, residual diagnostics, and multi-metric heatmaps.
- **Cross-Validation**: Robust evaluation using rolling-window splits to ensure stability across time.
- **Automated Artifacts**: Saves final forecasts, performance reports, and serialized models ready for inference.

## 🏗️ Project Structure
```text
walmart/
├── artifacts/            # Generated metrics (CSV), plots (PNG), and tuning params
├── data/                 # Raw and processed datasets
├── models/               # Serialized top-performing models (.joblib, .json)
├── notebooks/            # Step-by-step experiment pipeline
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_classical_models.ipynb
│   ├── 03_ml_models.ipynb
│   ├── 03b_extended_ml_models.ipynb   <-- Added for more robust ML
│   ├── 04_evaluation_visualization.ipynb
│   ├── 04b_extended_visualizations.ipynb <-- Interactive/Parallel Viz
│   ├── 04c_diverse_visualizations.ipynb  <-- Distribution & Analysis
│   ├── 04e_best_model_showcase.ipynb     <-- The Winner Circle
│   └── 05_model_saving_and_inference.ipynb
└── requirements.txt      # Project dependencies
```

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execution Flow
For the most current results, run the notebooks in numerical order. The project is designed to be Modular:
1. **Pre-processing**: Aggregates raw transactions into clean daily time-series.
2. **Experimentation**: Run notebooks 02, 03, and 03b to build and score models.
3. **Visualization**: Run 04b, 04c, and 04e to generate deep insights.
4. **Export**: Use 05 to save the best model for production.

## 📈 Top Performance (RMSE)
| Target          | Best Model | RMSE    |
|:----------------|:-----------|:--------|
| **Daily Revenue**| **ARIMA**  | ~3364   |
| **Daily Orders** | **ARIMA**  | ~12.29  |

## 🛠️ Built With
- **Python 3.12+**
- **Statsmodels**: ARIMA, SARIMA, ETS
- **Scikit-Learn**: Lasso, Ridge, Random Forest
- **Boosting**: XGBoost, LightGBM
- **Visualization**: Seaborn, Matplotlib

## 👥 Authors
**Ritik Sinha** - [GitHub](https://github.com/Ritik1207-ind)
**Udit Dadhich** - [Github](https://github.com/UditDadhich)
**Shivam kishore** - [Github](https://github.com/Shivwhoo)
---
*Developed as part of Advanced Agentic Coding experimentation.*
