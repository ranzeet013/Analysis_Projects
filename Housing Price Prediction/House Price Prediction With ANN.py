#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction With ANN

# House price prediction using Artificial Neural Networks (ANN) is a popular application of machine learning and predictive analytics. ANNs are powerful models that can learn patterns and relationships from historical data to make predictions on new, unseen data. When it comes to house price prediction, an ANN can analyze various features or attributes of a house, such as its size, location, number of rooms, amenities, and more, to estimate its corresponding price.

# # Importing Python Libraries

# Importing libraries is an essential step in any data analysis or machine learning project. These libraries provide various functions and tools to manipulate, visualize, and analyze data efficiently. Here are explanations of some popular data analysis libraries:

# Pandas: Pandas is a powerful and widely used library for data manipulation and analysis. It provides data structures like DataFrames and Series, which allow you to store and manipulate tabular data. Pandas offers a wide range of functions for data cleaning, filtering, aggregation, merging, and more. It also supports reading and writing data from various file formats such as CSV, Excel, SQL databases, and more.

# NumPy: NumPy (Numerical Python) is a fundamental library for scientific computing in Python. It provides efficient data structures like arrays and matrices and a vast collection of mathematical functions. NumPy enables you to perform various numerical operations on large datasets, such as element-wise calculations, linear algebra, Fourier transforms, and random number generation. It also integrates well with other libraries for data analysis and machine learning.

# Matplotlib: Matplotlib is a popular plotting library that enables you to create a wide range of static, animated, and interactive visualizations. It provides a MATLAB-like interface and supports various types of plots, including line plots, scatter plots, bar plots, histograms, and more. Matplotlib gives you extensive control over plot customization, including labels, colors, legends, and annotations, allowing you to effectively communicate insights from your data

# Seaborn: Seaborn is a statistical data visualization library that is built on top of Matplotlib. It provides a high-level interface for creating attractive and informative statistical graphics. Seaborn simplifies the process of creating complex visualizations like heatmaps, kernel density plots, violin plots, and regression plots. It also offers additional functionalities for handling categorical data, multi-plot grids, and color palettes.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Reading Dataset

# In[3]:


dataframe = pd.read_csv('data.csv')


# # Exploratory Data Analysis

# Exploratory Data Analysis (EDA) is a crucial step in the data analysis process that involves examining and summarizing the main characteristics of a dataset. Its main objective is to gain insights, discover patterns, and understand the structure of the data before proceeding with more advanced analysis or modeling techniques

# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe.shape


# In[7]:


dataframe.info()


# In[8]:


dataframe.describe()


# In[9]:


dataframe.describe().T


# In[10]:


dataframe.isnull().sum()


# # Data Visualization

# Data visualization is the graphical representation of data using visual elements such as charts, graphs, maps, and other visual tools. It aims to present complex data in a visual format that is easy to understand, interpret, and communicate

# In[11]:


plt.figure(figsize = (12, 5))
sns.distplot(dataframe['price'])


# In[12]:


dataframe.columns


# In[13]:


dataframe['bedrooms'].value_counts()


# In[14]:


sns.boxplot(dataframe['bedrooms'])


# In[15]:


plt.figure(figsize = (12, 5))
sns.scatterplot(x = 'price', y = 'sqft_living', data = dataframe)


# In[16]:


sns.scatterplot(x = 'bedrooms', y = 'price', data = dataframe)


# In[17]:


plt.figure(figsize = (12, 5))
sns.scatterplot(x = 'sqft_above', y = 'price', data = dataframe)


# In[18]:


plt.figure(figsize = (12, 5))
sns.scatterplot(x = 'price', y = 'sqft_basement', data = dataframe)


# In[19]:


plt.figure(figsize = (12, 5))
sns.scatterplot(x = 'sqft_above', y = 'sqft_basement', data = dataframe, hue = 'price')


# In[20]:


dataframe.sort_values('price', ascending = False).head(10)


# In[21]:


len(dataframe)*(0.01)


# In[22]:


top_1_percent = dataframe.sort_values('price', ascending = False).iloc[46:]


# In[23]:


plt.figure(figsize = (12, 5))
sns.scatterplot(x = 'sqft_above', y = 'sqft_basement', data = top_1_percent, 
                hue = 'price', palette = 'RdYlGn',
                edgecolor = None, alpha = 0.2)


# In[24]:


sns.boxplot(x = 'waterfront', y = 'price', data = dataframe)


# # Feature Engineering From Date

# Feature engineering from date variables involves extracting relevant information or creating new features from date-related data

# Dates have various components such as year, month, day, day of the week, quarter, etc. Extracting these components as separate features can provide valuable information. For example, you can extract the month and create a feature indicating the season (e.g., spring, summer, fall, winter) based on the month

# In[25]:


dataframe.head()


# In[29]:


dataframe['date'] = pd.to_datetime(dataframe['date'])


# In[31]:


dataframe['month'] = dataframe['date'].apply(lambda date:date.month)


# In[33]:


dataframe['year'] = dataframe['date'].apply(lambda date:date.year)


# In[36]:


sns.boxplot(x = 'year', y = 'price', data = dataframe)


# In[37]:


sns.boxplot(x = 'month', y = 'price', data = dataframe)


# In[38]:


dataframe.groupby('month').mean()['price'].plot()


# In[39]:


dataframe.groupby('year').mean()['price'].plot()


# In[40]:


dataframe.head()


# In[41]:


dataframe = dataframe.drop('date', axis = 1)


# In[43]:


dataframe.head()


# In[44]:


dataframe.columns


# In[45]:


dataframe.info()


# In[46]:


dataframe['yr_renovated'].value_counts


# In[47]:


dataframe['yr_built'].value_counts


# # Splitting The Data

# Splitting the data refers to dividing the dataset into separate subsets for training, validation, and testing purposes. This division is essential to assess the performance of a machine learning model on unseen data and prevent overfitting. Here are the common types of data splits:

# Training Set: The training set is the largest subset of the data used to train the machine learning model. It is used to learn the underlying patterns and relationships between the input features and the target variable. Typically, around 70-80% of the data is allocated for training

# Validation Set: The validation set is used to tune and optimize the model during the training process. It helps in selecting hyperparameters, evaluating different models, and making decisions about model architecture or feature selection. The validation set aids in preventing overfitting by providing an unbiased evaluation of the model's performance. It is generally around 10-20% of the data.

# Test Set: The test set is used to assess the final performance of the trained model. It represents unseen data that the model has not been exposed to during training or validation. The test set helps estimate the model's generalization ability and provides a reliable measure of its performance on real-world data. Typically, it comprises 10-20% of the data

# In[48]:


x = dataframe.drop('price', axis = 1)
y = dataframe['price']


# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=101)


# # Scaling The Data

# Scaling the data is a preprocessing step that involves transforming the numerical features of a dataset to a similar scale or range. It ensures that all features contribute equally to the analysis and modeling process, regardless of their original units or scales. Here's a description of common scaling techniques:

# Min-Max Scaling: Min-max scaling, also known as normalization, rescales the data to a fixed range, typically between 0 and 1. It subtracts the minimum value of the feature and divides by the range (difference between the maximum and minimum values). This technique preserves the relative relationships between data points but maps them to a common range.

# In[52]:


from sklearn.preprocessing import MinMaxScaler


# In[53]:


scaler = MinMaxScaler()


# In[55]:


x_train= scaler.fit_transform(x_train)


# In[56]:


x_test = scaler.transform(x_test)


# In[58]:


x_train.shape


# In[59]:


x_test.shape


# # Importing Deep Learning Libraries

# Importing deep learning libraries allows you to leverage powerful tools and frameworks specifically designed for building and training deep neural networks. Here's a brief description of popular deep learning libraries:

# TensorFlow: TensorFlow is an open-source deep learning library developed by Google. It provides a comprehensive ecosystem for building and training various types of neural networks. TensorFlow offers high-level APIs like Keras for ease of use and abstraction, as well as low-level APIs for fine-grained control. It supports distributed computing, GPU acceleration, and deployment on different platforms.

# In[61]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# # Creating The Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.

# In[63]:


model = Sequential()
model.add(Dense(15, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(1))


# Compiling the model in deep learning involves configuring essential components that define how the model will be trained. 

# In[64]:


model.compile(optimizer = 'Adam', loss = 'mse', metrics = ['accuracy'])


# # Training The Model

# Training the model in deep learning involves the process of iteratively updating the model's parameters (weights and biases) based on the provided training data to minimize the loss function and improve the model's performance

# In[65]:


model.fit(x = x_train, 
          y = y_train.values,
          validation_data = (x_test, y_test.values),
          batch_size = 128,
          epochs = 250)


# In[66]:


loss = pd.DataFrame(model.history.history)


# In[67]:


loss.plot()


# In[70]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score


# In[71]:


x_test


# # Predicting New Data

# In[72]:


prediction = model.predict(x_test)


# In[73]:


mean_absolute_error(y_test, prediction)


# In[74]:


np.sqrt(mean_absolute_error(y_test, prediction))


# In[75]:


explained_variance_score(y_test, prediction)


# In[76]:


dataframe['price'].describe()


# In[77]:


plt.scatter(y_test, prediction)
plt.plot(y_test, y_test, 'r')


# # Prediction In Single House 

# In[79]:


single_house =  dataframe.drop('price',axis=1).iloc[0]


# In[81]:


single_house = scaler.transform(single_house.values.reshape(-1, 14))


# In[82]:


single_house


# In[83]:


model.predict(single_house)


# In[84]:


dataframe.iloc[0]


# In[ ]:




