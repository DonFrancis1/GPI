#!/usr/bin/env python
# coding: utf-8

# ## Facebook Prophet Model
# 
# New notebook

# In[2]:


pip install prophet


# In[3]:


pip install fbprophet


# In[4]:


data = spark.sql("SELECT Country,GPI,Date(Year) as Year FROM GPI.GPI")
display(data)


# In[5]:


import pandas as pd
df = data.toPandas()
display(df)


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Preprocessing (if needed)
# Depending on the dataset, you may need to handle missing values, outliers, and perform other preprocessing steps.

# Create a separate model for each country
unique_countries = df['Country'].unique()

# Time series forecasting using Prophet
for country in unique_countries:
    # Select data for the current country
    country_data = df[df['Country'] == country].copy()

    # Create a Prophet model
    model = Prophet()

    # Rename columns for Prophet
    country_data.rename(columns={'Year': 'ds', 'GPI': 'y'}, inplace=True)

    # Fit the model with data
    model.fit(country_data[['ds', 'y']])

    # Forecasting future GPI values
    future = model.make_future_dataframe(periods=24, freq='M')  # Forecasting for the next 12 months
    forecast = model.predict(future)

    # Plot the forecast
    fig, ax = plt.subplots()
    ax.plot(country_data['ds'], country_data['y'], label='Actual')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3, color='gray')
    ax.set_xlabel('Date')
    ax.set_ylabel('GPI')
    ax.set_title(f'Forecast for {country}')
    ax.legend()
    plt.show()

