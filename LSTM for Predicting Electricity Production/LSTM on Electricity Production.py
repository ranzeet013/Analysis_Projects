#!/usr/bin/env python
# coding: utf-8

# # LSTM on Electricity Production

# The goal of this project is to implement a LSTM Recurrent Neural Network (RNN) model to predict the values of a electricity production. The RNN will be trained on a sequence of electricity production data points and learn to predict the next value in the sequence.

# # Importing Libraries

# These are just a few examples of popular Python libraries. You can import any other library using the same import statement followed by the library name or alias:

# NumPy: for numerical operations and array manipulation
# 
# Pandas: for data manipulation and analysis
# 
# Matplotlib: for creating visualizations
# 
# Scikit-learn: for machine learning algorithms

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading Dataset

# In[2]:


dataframe = pd.read_csv('Electric_Production.csv', index_col = 'DATE', parse_dates = True)


# # Exploratory Data Analysis

# In[3]:


dataframe.head()


# In[4]:


dataframe.tail()


# In[5]:


dataframe.columns = ['Production']


# In[6]:


dataframe.head()


# In[7]:


dataframe.plot(figsize = (12, 5))


# In[8]:


len(dataframe)


# In[9]:


test_size = 60


# In[10]:


test_index = len(dataframe) - test_size


# In[11]:


test_index


# # Splitting The Dataset

# Dataset splitting is an important step in machine learning and data analysis. It involves dividing a dataset into two or more subsets to train and evaluate a model effectively. The most common type of dataset splitting is into training and testing subsets.

# In[12]:


train_dataframe = dataframe.iloc[:test_index]
test_dataframe = dataframe.iloc[test_index:]


# In[13]:


train_dataframe


# In[14]:


test_dataframe


# # Scaling

# Scaling is a common preprocessing step in data analysis and machine learning. It involves transforming the features of a dataset to a standard scale, which can help improve the performance and stability of models

# MinMaxScaler is a popular scaling technique used in data preprocessing. It scales the features to a specified range, typically between 0 and 1.

# In[15]:


from sklearn.preprocessing import MinMaxScaler


# In[16]:


scaler = MinMaxScaler()


# In[17]:


scaler.fit(train_dataframe)


# In[18]:


scaled_train = scaler.transform(train_dataframe)
scaled_test = scaler.transform(test_dataframe)


# In[19]:


scaled_train


# In[20]:


scaled_test


# # Timeseries Generator

# In time series analysis, a common approach is to use a time series generator to generate batches of sequential data for training recurrent neural networks (RNNs) or other time-based models. This allows you to efficiently process and train models on large time series datasets. Here's an example of how you can create a time series generator using the TimeseriesGenerator calss.

# In[21]:


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[22]:


length = 12
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)


# In[23]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping


# # Building Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.

# In[24]:


n_features = 1


# In[25]:


model = Sequential()
model.add(LSTM(100, activation = 'relu', input_shape = (length, n_features)))
model.add(Dense(1))


# # Compiling Model 

# Compiling the model in deep learning involves configuring essential components that define how the model will be trained.

# In[26]:


model.compile(optimizer = 'adam', loss = 'mse')


# In[27]:


model.summary()


# In[28]:


early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2)


# In[29]:


validation_generator = TimeseriesGenerator(scaled_test,scaled_test, length=length, batch_size=1)


# # Training The Model

# Training the model in deep learning involves the process of iteratively updating the model's parameters (weights and biases) based on the provided training data to minimize the loss function and improve the model's performance

# In[30]:


model.fit(generator, 
          validation_data = validation_generator,
          epochs = 20,
          callbacks = [early_stopping])


# In[31]:


model.save('model_LSTM_production')


# In[33]:


loss = pd.DataFrame(model.history.history)


# In[34]:


loss.plot()


# # Test Prediction On Test Dataset

# In[39]:


test_predictions = []
first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))
for i in range(len(test_dataframe)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[41]:


test_prediction = scaler.inverse_transform(test_predictions)


# In[42]:


test_predictions


# In[44]:


test_dataframe['Prediction'] = test_prediction


# In[45]:


test_dataframe.head()


# In[46]:


test_dataframe.plot(figsize = (12, 5))


# # Predicting On Whole Dataset

# In[47]:


full_scaler = MinMaxScaler()


# In[49]:


full_dataset_scaled = full_scaler.fit_transform(dataframe)


# In[50]:


length = 12
generator = TimeseriesGenerator(full_dataset_scaled, full_dataset_scaled, length=length, batch_size=1)


# In[53]:


model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(length, n_features)))
model.add(Dense(1))


# In[54]:


model.compile(optimizer = 'adam', loss = 'mse')


# In[55]:


model.summary()


# In[56]:


model.fit_generator(generator,epochs=8)


# In[58]:


prediction = []
periods = 12
first_eval_batch = full_dataset_scaled[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))
for i in range(periods):
    current_pred = model.predict(current_batch)[0]
    prediction.append(current_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[136]:


model.save('model_LSTM_full_df.h5')


# In[59]:


loss = pd.DataFrame(model.history.history)


# In[62]:


prediction = scaler.inverse_transform(prediction)


# In[128]:


prediction_index = pd.date_range(start='2017-10-01',periods=periods,freq='MS')


# In[129]:


prediction_df = pd.DataFrame(data=prediction,
                             index=prediction_index,
                             columns=['Prediction'])


# In[130]:


prediction_df


# In[131]:


dataframe.plot()


# In[132]:


prediction_df.plot()


# In[133]:


ax = dataframe.plot()
prediction_df.plot(ax = ax)

