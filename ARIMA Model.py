#!/usr/bin/env python
# coding: utf-8

# ## ARIMA Model
# 
# New notebook

# In[1]:


df = spark.sql("SELECT country,GPI,Year FROM GPI.GPI")
display(df)


# In[2]:


import pandas as pd
data = df.toPandas()
display(data)


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# In[4]:


print(data.isnull().sum())


# In[5]:


# Convert 'Global Peace Index' to numeric dtype (float)
data['GPI'] = pd.to_numeric(data['GPI'], errors='coerce')


# In[6]:


# Drop rows with missing values
data.dropna(inplace=True)


# In[7]:


# One-hot encoding for 'Country' column
data = pd.get_dummies(data, columns=['country'], drop_first=True)


# In[8]:


# Convert data to numpy arrays
X = np.asarray(data[['Year'] + [col for col in data.columns if col.startswith('country_')]])
y = np.asarray(data['GPI'])


# In[9]:


data['Year'] = pd.to_datetime(data['Year'])  # Convert 'Year' column to datetime format


# In[10]:


# Set 'Year' as the index
data.set_index('Year', inplace=True)


# In[11]:


# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]


# In[12]:


# Initialize and fit the ARIMA model
order = (1, 1, 1)  # ARIMA order: (p, d, q)
model = ARIMA(train['GPI'], order=order)
fitted_model = model.fit()


# In[13]:


# Make predictions on the test set
predictions = fitted_model.forecast(steps=len(test))


# In[14]:


# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(test['GPI'], predictions)

print("Mean Squared Error:", mse)


# In[16]:


# Visualize the results
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['GPI'], label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Year')
plt.ylabel('Global Peace Index')
plt.legend()
plt.title('ARIMA Model Forecast')
plt.show()

