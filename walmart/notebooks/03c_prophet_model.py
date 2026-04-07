#!/usr/bin/env python
# coding: utf-8

# # 03c - Prophet Model with Seasonal/Holiday Handling
# 
# To solve the issue of "smoothed" forecasts, we use the **Prophet** model which handles:
# - **Seasonality**: Explicitly models daily, weekly, and yearly patterns.
# - **Holidays**: Incorporates a built-in US calendar of events that cause sales spikes.
# - **Saturation/Trends**: Handles high-growth and saturation points better than purely recursive models.

# In[ ]:


from pathlib import Path
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Suppress Prophet logging for cleaner output
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

sns.set_style('whitegrid')
cwd = Path.cwd()
if cwd.name == 'notebooks':
    BASE_DIR = cwd.parent
    ARTIFACTS_DIR = BASE_DIR / 'artifacts'
else:
    BASE_DIR = cwd
    ARTIFACTS_DIR = BASE_DIR / 'walmart' / 'artifacts'

df = pd.read_csv(ARTIFACTS_DIR / 'daily_series.csv', parse_dates=['date'])
# Prophet expects columns: 'ds' (date) and 'y' (target)
df = df.rename(columns={'date': 'ds'})
print(f"Prophet ready. Using artifacts from: {ARTIFACTS_DIR}")


# In[ ]:


def run_prophet(target_col, horizon=30, apply_log=True, include_holidays=True):
    data = df[['ds', target_col]].copy().rename(columns={target_col: 'y'})

    if apply_log:
        data['y'] = np.log1p(data['y'])

    train = data.iloc[:-horizon]
    test = data.iloc[-horizon:]

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

    if include_holidays:
        m.add_country_holidays(country_name='US')

    m.fit(train)

    future = m.make_future_dataframe(periods=horizon)
    forecast = m.predict(future)

    # Extract predictions for the test horizon
    pred_y = forecast.tail(horizon)['yhat']

    if apply_log:
        pred_y = np.expm1(pred_y)
        train_y = np.expm1(train['y'])
        test_y = np.expm1(test['y'])
    else:
        train_y = train['y']
        test_y = test['y']

    # Metrics
    mae = np.mean(np.abs(test_y - pred_y))
    rmse = np.sqrt(np.mean((test_y - pred_y)**2))

    print(f"[{target_col}] Prophet MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Plotting
    plt.figure(figsize=(15, 7))
    plt.plot(train['ds'].tail(90), train_y.tail(90), label='History', color='blue', alpha=0.5)
    plt.plot(test['ds'], test_y, label='Actual Data', color='blue', linewidth=2)
    plt.plot(test['ds'], pred_y, label='Prophet Forecast', color='orange', linestyle='--', marker='o')

    plt.title(f'Prophet Forecast (with Holidays/Seasonality): {target_col}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / f'03c_prophet_forecast_{target_col}.png')
    plt.close() # Important for non-interactive execution

    return mae, rmse

mae_rev, rmse_rev = run_prophet('daily_revenue')
mae_ord, rmse_ord = run_prophet('daily_orders')


# In[ ]:


# Save component plots to Artifacts for transparency
def save_components(target_col):
    data = df[['ds', target_col]].copy().rename(columns={target_col: 'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.add_country_holidays(country_name='US')
    m.fit(data)
    fig = m.plot_components(m.predict(data))
    fig.savefig(ARTIFACTS_DIR / f'03c_prophet_components_{target_col}.png')
    plt.close('all')

save_components('daily_revenue')
save_components('daily_orders')
print("Prophet pipeline completed.")

