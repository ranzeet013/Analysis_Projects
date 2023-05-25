#!/usr/bin/env python
# coding: utf-8

# # LSTM For Beer Production In Australlia

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
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading Dataset

# In[2]:


dataframe = pd.read_csv('monthly-beer-production-in-austr.csv', index_col = 'Month', parse_dates = True)


# # Exploratory Data Analysis

# The process of analyzing and understanding a dataset to gain insights and identify patterns or trends. The goal of exploring the data is to become familiar with its structure, distribution, and quality, as well as to identify potential issues or anomalies that may need to be addressed before further analysis.

# In[3]:


dataframe.head()


# In[4]:


dataframe.tail()


# In[5]:


dataframe.info()


# In[6]:


dataframe.describe()


# In[7]:


dataframe.isna().sum()


# In[8]:


dataframe.plot()


# In[9]:


len(dataframe)


# In[11]:


test_size = 18


# In[12]:


test_index = len(dataframe) - test_size


# In[13]:


test_index


# # Splitting Dataset

# Dataset splitting is an important step in machine learning and data analysis. It involves dividing a dataset into two or more subsets to train and evaluate a model effectively. The most common type of dataset splitting is into training and testing subsets.

# In[14]:


train_dataframe = dataframe.iloc[:test_index]
test_dataframe = dataframe.iloc[test_index:]


# In[15]:


test_dataframe


# In[16]:


train_dataframe


# # Scaling

# Scaling is a common preprocessing step in data analysis and machine learning. It involves transforming the features of a dataset to a standard scale, which can help improve the performance and stability of models

# MinMaxScaler is a popular scaling technique used in data preprocessing. It scales the features to a specified range, typically between 0 and 1.

# In[17]:


from sklearn.preprocessing import MinMaxScaler


# In[18]:


scaler = MinMaxScaler()


# In[19]:


scaler.fit(train_dataframe)


# In[20]:


scaled_train = scaler.transform(train_dataframe)
scaled_test = scaler.transform(test_dataframe)


# In[21]:


scaled_train


# In[22]:


scaled_test


# # Timeseries Generator

# In time series analysis, a common approach is to use a time series generator to generate batches of sequential data for training recurrent neural networks (RNNs) or other time-based models. This allows you to efficiently process and train models on large time series datasets. Here's an example of how you can create a time series generator using the TimeseriesGenerator calss.

# In[23]:


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[24]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping


# In[25]:


length = 12
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)


# In[26]:


n_features = 1


# # Building Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.

# In[27]:


model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(length, n_features)))
model.add(Dense(1))


# In[28]:


model.summary()


# # Compiling Model

# Compiling the model in deep learning involves configuring essential components that define how the model will be trained.

# In[29]:


model.compile(optimizer='adam', loss='mse')


# In[30]:


early_stopping = EarlyStopping(monitor='val_loss',patience=2)


# In[31]:


validation_generator = TimeseriesGenerator(scaled_test,scaled_test, length=length, batch_size=1)


# # Training Model

# Training the model in deep learning involves the process of iteratively updating the model's parameters (weights and biases) based on the provided training data to minimize the loss function and improve the model's performance

# In[33]:


model.fit_generator(generator,epochs=20,
                    validation_data=validation_generator,
                   callbacks=[early_stopping])


# In[34]:


loss = pd.DataFrame(model.history.history)


# In[35]:


loss.head()


# In[36]:


loss.plot()


# # Predicting On Test Dataframe

# In[37]:


test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test_dataframe)):

    current_pred = model.predict(current_batch)[0]

    test_predictions.append(current_pred) 

    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[38]:


test_prediction = scaler.inverse_transform(test_predictions)


# In[39]:


test_prediction


# In[40]:


test_dataframe['Prediction'] = test_prediction


# In[41]:


test_dataframe


# # Test Predicting VS Monthly Production

# In[44]:


test_dataframe.plot(figsize = (12, 5))


# # Predicting On Full Dataframe

# In[45]:


full_scaled = MinMaxScaler()


# In[49]:


scaled_full_data = full_scaled.fit_transform(dataframe)


# In[50]:


length = 12
generator = TimeseriesGenerator(scaled_full_data, scaled_full_data, length=length, batch_size=1)


# In[51]:


model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(length, n_features)))
model.add(Dense(1))


# In[52]:


model.summary()


# In[53]:


model.compile(optimizer='adam', loss='mse')


# In[54]:


model.fit_generator(generator,epochs=10)


# In[55]:


model.save('model_beer_production.h5')


# In[56]:


loss = pd.DataFrame(model.history.history)


# In[57]:


loss.plot()


# In[58]:


predict = []
periods = 18

first_eval_batch = scaled_full_data[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(periods):

    current_pred = model.predict(current_batch)[0]

    predict.append(current_pred) 

    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[59]:


predict = scaler.inverse_transform(predict)


# In[60]:


predict


# In[76]:


predict_index = pd.date_range(start = '1995-08-01', periods = periods, freq = 'MS')


# In[77]:


predict_df = pd.DataFrame(data = predict,
                          index = predict_index, 
                          columns = ['Prediction'])


# In[78]:


predict_df


# In[79]:


dataframe.plot()


# In[80]:


predict_df.plot()


# In[83]:


ax = dataframe.plot()
predict_df.plot(ax=ax)


# # Actual Production And Predicted Preoduction

# The graph shows the actual prediction of beer along with prediction of beer production in comming 18 month.

# In[81]:


ax = dataframe.plot()
predict_df.plot(ax=ax)
plt.xlim('1994-03-01','1997-02-01')

