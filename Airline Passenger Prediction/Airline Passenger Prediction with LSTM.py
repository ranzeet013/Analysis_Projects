#!/usr/bin/env python
# coding: utf-8

# # Airline Passenger Prediction with LSTM

# The airline passenger prediction project with LSTM (Long Short-Term Memory) involves using historical data on airline passenger numbers to build a predictive model. The goal is to forecast future passenger numbers based on patterns and trends in the historical data.

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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Dataset

# In[2]:


dataframe = pd.read_csv('AirPassengers.csv')


# # Data Investigation

# Data investigation, also known as exploratory data analysis (EDA), is an essential step in any data analysis project. It involves examining and exploring the data to gain insights, understand its characteristics, identify patterns or anomalies, and make informed decisions about subsequent data processing and modeling.

# In[3]:


dataframe.head()


# In[4]:


dataframe.tail()


# In[5]:


dataframe.shape


# In[6]:


dataframe.info()


# In[7]:


dataframe.isna().sum().any()


# In[8]:


dataframe.isna().sum()


# In[9]:


dataframe.describe()


# In[10]:


dataframe = dataframe['#Passengers']
dataframe = np.array(dataframe).reshape(-1,1)


# In[11]:


dataframe


# In[12]:


plt.plot(dataframe)


# # Scaling Dataset

# Scaling, also known as data normalization, is a preprocessing technique used to standardize the range and distribution of numerical variables in a dataset. It is an important step in data analysis and machine learning as it helps improve the performance and stability of models that rely on distance calculations or gradient-based optimization algorithms.
# 
# Min-Max scaling: This technique scales the data to a fixed range, typically between 0 and 1. It is calculated by subtracting the minimum value of the variable from each data point and then dividing by the range (maximum - minimum). Min-Max scaling maps the data to a specific range and is useful when you want to preserve the relationships between the variables' values

# In[13]:


from sklearn.preprocessing import MinMaxScaler


# In[14]:


scaler = MinMaxScaler()


# In[15]:


dataframe = scaler.fit_transform(dataframe)


# In[16]:


dataframe


# # Spliting Dataset

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[17]:


train_size = 100
test_size = 44


# In[18]:


train = dataframe[0:train_size,:]
test = dataframe[train_size:len(dataframe),:]


# In[19]:


train


# In[20]:


test


# In[21]:


def get_data(dataset,look_back):
    dataX , dataY =[] , []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i+look_back,0])
    return np.array(dataX),np.array(dataY)


# In[22]:


look_back = 1


# In[23]:


x_train ,y_train = get_data(train,look_back)
x_test,y_test = get_data(test,look_back)


# In[24]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[25]:


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)


# In[26]:


x_train.shape, x_test.shape


# # Building Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training

# In[27]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# In[28]:


model = Sequential()
model.add(LSTM(5,input_shape = (1,look_back)))
model.add(Dense(1))


# In[29]:


model.summary()


# # Compiling Model

# Compiling the model in deep learning involves configuring essential components that define how the model will be trained.

# In[30]:


model.compile(loss='mean_squared_error',optimizer='adam')


# # Fitting Model

# Fitting a model refers to the process of training the model on a given dataset. This involves optimizing the model's parameters or coefficients to make accurate predictions or capture patterns in the data.

# In[31]:


model.fit(x_train,y_train,epochs=50,batch_size=1)


# In[32]:


model.save('model_passanger_pred.h5')


# # Learning Curve

# The learning curve is a plot that shows how the loss and accuracy of a model change during training. It provides insights into how well the model is learning from the training data and how it generalizes to unseen data. The learning curve typically shows the training and validation loss/accuracy on the y-axis and the number of epochs on the x-axis. By analyzing the learning curve, you can identify if the model is overfitting (high training loss, low validation loss) or underfitting (high training and validation loss). It is a useful tool for monitoring and evaluating the performance of machine learning models.

# In[34]:


loss = pd.DataFrame(model.history.history)


# In[35]:


loss.head()


# In[36]:


loss.plot()


# # Prediction

# In[37]:


y_pred = model.predict(x_test)


# In[38]:


y_pred


# In[39]:


y_pred = scaler.inverse_transform(y_pred)


# In[40]:


y_pred


# In[41]:


y_test = np.array(y_test)


# In[42]:


y_test


# In[43]:


y_test = y_test.reshape(-1,1)


# In[44]:


y_test.shape


# In[45]:


y_test = scaler.inverse_transform(y_test)


# In[46]:


y_test


# # Prediction VS Actual Values

# In[47]:


plt.plot(y_test, label = 'Passanger')
plt.plot(y_pred,label='Predicted Passanger')
plt.ylabel('passengers')
plt.legend()
plt.show()

