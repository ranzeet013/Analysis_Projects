#!/usr/bin/env python
# coding: utf-8

# # Predicting Dogecoin Price

# The goal of the Dogecoin Price Prediction project is to develop a model that can forecast the future price movements of Dogecoin, a cryptocurrency that was initially created as a meme but gained popularity in the crypto community. The project aims to leverage historical data and machine learning techniques to predict the price of Dogecoin over a given time horizon.

# # Importing Libraries

# These are just a few examples of popular Python libraries. You can import any other library using the same import statement followed by the library name or alias:
# 
# NumPy: for numerical operations and array manipulation
# 
# Pandas: for data manipulation and analysis
# 
# Matplotlib: for creating visualizations
# 
# Scikit-learn: for machine learning algorithms

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Importing Dataset

# In[3]:


dataframe = pd.read_csv('DOGE-USD.csv')


# # Data Exploration 

# Data exploration is a crucial step in the Dogecoin Price Prediction project as it allows you to gain insights into the collected data, understand its characteristics, and identify any patterns or trends that may be relevant for predicting Dogecoin prices.

# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe.shape


# In[7]:


dataframe.info()


# In[8]:


dataframe.isna().any()


# In[9]:


dataframe.dropna()
plt.figure(figsize=(10, 4))
plt.title("DogeCoin Price USD")
plt.xlabel("Date")
plt.ylabel("Close")
plt.plot(dataframe["Close"])
plt.show()


# # AutoTs

# AutoTs (Automated Time Series) is a Python library that can be used in the Dogecoin Price Prediction project to automate the process of building and evaluating time series forecasting models. AutoTs aims to simplify the task of time series forecasting by providing a streamlined workflow for model selection, hyperparameter tuning, and performance evaluation.

# In[10]:


from autots import AutoTS


# In[11]:


model = AutoTS(forecast_length=10,
               frequency='infer', 
               ensemble='simple', 
               drop_data_older_than_periods=200)


# In[13]:


model = model.fit(dataframe, date_col='Date', value_col='Close', id_col=None)


# # Prediction Close Values

# In[15]:


prediction = model.predict()


# In[16]:


forecast = prediction.forecast


# In[18]:


print(forecast)


# # Thanks !
