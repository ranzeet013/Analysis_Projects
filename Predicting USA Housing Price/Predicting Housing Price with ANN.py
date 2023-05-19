#!/usr/bin/env python
# coding: utf-8

# # Predicting Housing Price with ANN

# # Housing Dataset of 5000 people staying in USA

# # About dataset

# A dataset in CSV format that describes the rows where 5000 entries of each particular house in USA .
# 
# The dataset has columns as such as :
# 
# Avg. Area Income : Average income of people residing in a particular area.
# 
# Avg. Area House Age : Average age of houses in the particular area.
# 
# Avg. Area Number of Rooms : Average number of rooms in each house .
# 
# Avg. Area Number of Bedrooms : Average number of bedrooms in a particular house.
# 
# Area Population : Population or the people residing in that area
# 
# Price : Price of house
# 
# Address : A unique column that has a unique value for each entry of data in the whole dataset (count = 5000).

# # Importing Python Libraries

# Importing libraries is an essential step in any data analysis or machine learning project. These libraries provide various functions and tools to manipulate, visualize, and analyze data efficiently. Here are explanations of some popular data analysis libraries:

# Pandas: Pandas is a powerful and widely used library for data manipulation and analysis. It provides data structures like DataFrames and Series, which allow you to store and manipulate tabular data. Pandas offers a wide range of functions for data cleaning, filtering, aggregation, merging, and more

# NumPy: NumPy (Numerical Python) is a fundamental library for scientific computing in Python. It provides efficient data structures like arrays and matrices and a vast collection of mathematical functions. NumPy enables you to perform various numerical operations on large datasets, such as element-wise calculations, linear algebra, Fourier transforms, and random number generation

# Matplotlib: Matplotlib is a popular plotting library that enables you to create a wide range of static, animated, and interactive visualizations. It provides a MATLAB-like interface and supports various types of plots, including line plots, scatter plots, bar plots, histograms, and more

# Seaborn: Seaborn is a statistical data visualization library that is built on top of Matplotlib. It provides a high-level interface for creating attractive and informative statistical graphics. Seaborn simplifies the process of creating complex visualizations like heatmaps, kernel density plots, violin plots, and regression plots.

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


dataframe = pd.read_csv('USA_Housing.csv')


# # Exploratory Data Analysis

# Exploratory Data Analysis (EDA) is a crucial step in the data analysis process that involves examining and summarizing the main characteristics of a dataset. Its main objective is to gain insights, discover patterns, and understand the structure of the data before proceeding with more advanced analysis or modeling techniques.

# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe.shape


# In[7]:


dataframe.info()


# In[8]:


dataframe.select_dtypes(include = 'float')


# In[9]:


dataframe.select_dtypes(include = 'object')


# In[10]:


dataframe.describe()


# In[11]:


dataframe['Price'].describe()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.

# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[12]:


corr_matrix = dataframe.corr()


# In[13]:


corr_matrix


# In[14]:


plt.figure(figsize = (12, 6))
sns.heatmap(corr_matrix,
            annot = True,
            cmap = 'coolwarm')


# # Checking Missing Value

# In[15]:


dataframe.isnull().sum()


# In[16]:


dataframe.columns


# # Data Visualization 

# Data visualization is the graphical representation of data using visual elements such as charts, graphs, maps, and other visual tools. It aims to present complex data in a visual format that is easy to understand, interpret, and communicate

# In[17]:


plt.figure(figsize = (12, 6))
sns.distplot(dataframe['Price'])


# In[18]:


dataframe['Avg. Area Number of Bedrooms'].value_counts()


# In[19]:


sns.boxplot(dataframe['Avg. Area Number of Bedrooms'])


# In[20]:


plt.figure(figsize = (12, 5))
sns.scatterplot(x = 'Avg. Area Number of Bedrooms', y = 'Price', data = dataframe)


# In[21]:


dataframe['Avg. Area Number of Rooms'].value_counts()


# In[22]:


sns.boxplot(dataframe['Avg. Area Number of Rooms'])


# In[23]:


plt.figure(figsize = (12, 5))
sns.scatterplot(x = 'Avg. Area Number of Rooms', y = 'Price', data = dataframe)


# In[24]:


plt.figure(figsize = (12, 5))
sns.scatterplot(x = 'Avg. Area House Age', y = 'Avg. Area Income', data = dataframe)


# In[25]:


plt.figure(figsize = (12, 5))
sns.scatterplot(x = 'Avg. Area Number of Bedrooms', 
                y = 'Avg. Area Number of Rooms',
                data = dataframe,
                hue = 'Price')


# In[26]:


dataframe.sort_values('Price', ascending = False).head(10)


# In[27]:


len(dataframe)*(0.01)


# In[28]:


top_1_percent = dataframe.sort_values('Price', ascending = False)[50:]


# In[29]:


plt.figure(figsize = (12, 5))
sns.scatterplot(x = 'Avg. Area Number of Bedrooms',
                y = 'Avg. Area Number of Rooms',
                data = top_1_percent,
                hue = 'Price',
                palette = 'RdYlGn',
                edgecolor = None,
                alpha = 0.2)


# In[30]:


dataframe.head()


# In[31]:


dataframe.select_dtypes(include = 'object')


# In[32]:


dataframe = dataframe.drop('Address', axis = 1)


# In[33]:


dataframe.head()


# # Splitting The Data

# Splitting the data refers to dividing the dataset into separate subsets for training, validation, and testing purposes. This division is essential to assess the performance of a machine learning model on unseen data and prevent overfitting. Here are the common types of data splits:

# Training Set: The training set is the largest subset of the data used to train the machine learning model. It is used to learn the underlying patterns and relationships between the input features and the target variable. Typically, around 70-80% of the data is allocated for training.

# Test Set: The test set is used to assess the final performance of the trained model. It represents unseen data that the model has not been exposed to during training or validation. The test set helps estimate the model's generalization ability and provides a reliable measure of its performance on real-world data. Typically, it comprises 10-20% of the data.

# In[34]:


x = dataframe.drop('Price', axis = 1)


# In[35]:


y = dataframe['Price']


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y, 
                                                    test_size = 0.2,
                                                    random_state = 101)


# In[38]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Scaling Data

# Scaling the data is a preprocessing step that involves transforming the numerical features of a dataset to a similar scale or range. It ensures that all features contribute equally to the analysis and modeling process, regardless of their original units or scales. Here's a description of common scaling techniques:

# Min-Max Scaling: Min-max scaling, also known as normalization, rescales the data to a fixed range, typically between 0 and 1. It subtracts the minimum value of the feature and divides by the range (difference between the maximum and minimum values). This technique preserves the relative relationships between data points but maps them to a common range.

# In[39]:


from sklearn.preprocessing import MinMaxScaler


# In[40]:


scaler = MinMaxScaler()


# In[41]:


x_train = scaler.fit_transform(x_train)


# In[42]:


x_test = scaler.transform(x_test)


# In[43]:


x_train.shape, x_test.shape


# # Deep Learning Libraries

# Deep learning libraries allows you to leverage powerful tools and frameworks specifically designed for building and training deep neural networks. Here's a brief description of popular deep learning libraries:

# TensorFlow: TensorFlow is an open-source deep learning library developed by Google. It provides a comprehensive ecosystem for building and training various types of neural networks. TensorFlow offers high-level APIs like Keras for ease of use and abstraction, as well as low-level APIs for fine-grained control. It supports distributed computing, GPU acceleration, and deployment on different platforms.

# In[44]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# # Creating Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training. 

# In[45]:


model = Sequential()
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1))


# # Compiling The Model

# Compiling the model in deep learning involves configuring essential components that define how the model will be trained.

# In[46]:


model.compile(optimizer = 'Adam', loss = 'mse', metrics = ['accuracy'])


# # Training The Model

# Training the model in deep learning involves the process of iteratively updating the model's parameters (weights and biases) based on the provided training data to minimize the loss function and improve the model's performance

# In[47]:


model.fit(x = x_train,
          y = y_train.values,
          validation_data = (x_test, y_test.values),
          epochs = 250)


# In[48]:


loss = pd.DataFrame(model.history.history)


# In[49]:


loss.plot()


# In[53]:


model.save('model_housing_prediction.h5')


# In[51]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score


# In[52]:


x_test


# # Predicting  Data

# In[54]:


prediction = model.predict(x_test)


# In[55]:


mean_absolute_error(y_test, prediction)


# In[57]:


np.sqrt(mean_absolute_error(y_test, prediction))


# In[58]:


mean_squared_error(y_test, prediction)


# In[59]:


explained_variance_score(y_test, prediction)


# In[60]:


plt.scatter(y_test, prediction)
plt.plot(y_test, y_test, 'r')


# # Predicting On Single House

# In[61]:


single_house = dataframe.drop('Price', axis = 1).iloc[0]


# In[64]:


single_house = scaler.transform(single_house.values.reshape(-1, 5))


# In[65]:


single_house


# In[66]:


model.predict(single_house)


# In[68]:


dataframe.iloc[0]


# In[ ]:




