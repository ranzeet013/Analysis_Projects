#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection

# The goal of this project is to develop a credit card fraud detection system using a Convolutional Neural Network (CNN). The project aims to identify fraudulent transactions and distinguish them from genuine transactions by analyzing various features associated with credit card transactions.

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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Importing Dataset

# In[3]:


dataframe = pd.read_csv('creditcard.csv')


# In[ ]:





# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe.shape


# In[7]:


dataframe.info()


# In[8]:


dataframe.isna().sum()


# # Statical Info Of Dataset

# In[9]:


dataframe.describe()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.

# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[10]:


corr_matrix = dataframe.corr()


# In[11]:


corr_matrix


# In[12]:


plt.figure(figsize = (30, 18))
sns.heatmap(corr_matrix,
            annot = True, 
            cmap = 'coolwarm')


# # Managing Data

# Data management refers to the process of organizing, storing, and manipulating data to ensure its availability, accuracy, and security. Effective data management is crucial in various fields, including business, research, and machine learning, as it enables efficient data access, analysis, and decision-making.

# In[13]:


dataframe.columns


# In[14]:


dataframe['Class'].value_counts()


# In[15]:


fraud = dataframe[dataframe['Class'] == 1]
not_fraud = dataframe[dataframe['Class'] == 0]


# In[16]:


fraud


# In[17]:


not_fraud


# In[18]:


fraud.shape, not_fraud.shape


# In[19]:


not_fraud_transaction = not_fraud.sample(n = 492)


# In[20]:


not_fraud_transaction.shape


# In[21]:


not_fraud_transaction.shape


# In[23]:


dataset = fraud.append(not_fraud_transaction, ignore_index = True)


# In[24]:


dataset


# In[25]:


dataset['Class'].value_counts()


# In[26]:


x = dataset.drop('Class', axis = 1)


# In[27]:


y = dataset['Class']


# In[28]:


x.shape, y.shape


# # Splitting Dataset

# Dataset splitting is an important step in machine learning and data analysis. It involves dividing a dataset into two or more subsets to train and evaluate a model effectively. The most common type of dataset splitting is into training and testing subsets.

# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)


# In[31]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Scaling 

# Scaling is a preprocessing step in machine learning that involves transforming the features or variables of your dataset to a consistent scale. It is important because many machine learning algorithms are sensitive to the scale of the input features. Scaling helps ensure that all features have a similar range and distribution, which can improve the performance and convergence of the model.

# StandardScaler is a popular scaling technique used in machine learning to standardize features by removing the mean and scaling to unit variance. It is available in the scikit-learn library, which provides a wide range of machine learning tools and preprocessing functions.

# In[32]:


from sklearn.preprocessing import StandardScaler


# In[33]:


scaler = StandardScaler()


# In[34]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[35]:


x_train


# In[36]:


x_test


# In[37]:


y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# In[38]:


y_train


# In[40]:


y_test


# In[41]:


x_train.shape, x_test.shape


# In[42]:


x_train = x_train.reshape(787, 30, 1)
x_test = x_test.reshape(197, 30, 1)


# In[43]:


x_train.shape, x_test.shape


# # Building Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.

# In[62]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# In[55]:


model = Sequential()


# In[56]:


model.add(Conv1D(filters = 32, kernel_size = 2,padding = 'same', activation = 'relu', input_shape = (30, 1)))

model.add(BatchNormalization())

model.add(MaxPooling1D(pool_size = 2))

model.add(Dropout(0.2))


# In[57]:


model.add(Conv1D(filters = 64, kernel_size = 2,padding = 'same', activation = 'relu', input_shape = (30, 1)))

model.add(BatchNormalization())

model.add(MaxPooling1D(pool_size = 2))

model.add(Dropout(0.2))


# In[58]:


model.add(Flatten())


# In[59]:


model.add(Dense(64, activation = 'relu'))

model.add(Dropout(0.3))


# In[60]:


model.add(Dense(1, activation = 'sigmoid'))


# In[61]:


model.summary()


# In[63]:


optimizer = Adam(learning_rate = 0.0001)


# # Compiling Model

# Compiling the model in deep learning involves configuring essential components that define how the model will be trained.

# In[64]:


model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])


# Early stopping is a technique used in machine learning to prevent overfitting and improve the generalization ability of a model. It involves monitoring the performance of a model during training and stopping the training process when the performance on a validation set starts to deteriorate.

# In[65]:


early_stop = EarlyStopping(monitor = 'val_loss', patience = 3)


# # Training Model

# Training the model in deep learning involves the process of iteratively updating the model's parameters (weights and biases) based on the provided training data to minimize the loss function and improve the model's performance.

# In[66]:


model.fit(x_train, y_train, 
          epochs = 25, 
          validation_data = (x_test, y_test), 
          callbacks = [early_stop])


# In[67]:


model.save('model_fraud_det.h5')


# # Learning Curve

# The learning curve is a plot that shows how the loss and accuracy of a model change during training. It provides insights into how well the model is learning from the training data and how it generalizes to unseen data. The learning curve typically shows the training and validation loss/accuracy on the y-axis and the number of epochs on the x-axis. By analyzing the learning curve, you can identify if the model is overfitting (high training loss, low validation loss) or underfitting (high training and validation loss). It is a useful tool for monitoring and evaluating the performance of machine learning models.

# In[68]:


loss = pd.DataFrame(model.history.history)


# In[70]:


loss.head()


# In[71]:


loss.plot()


# In[72]:


loss[['loss', 'val_loss']].plot()


# In[74]:


loss[['accuracy', 'val_accuracy']].plot()


# # Prediction

# In[75]:


y_pred = model.predict(x_test)


# In[77]:


print(y_test[2]), print(y_pred[2])


# # Error Analysis

# Error analysis is an important step in evaluating and improving the performance of a machine learning model. It involves analyzing the errors made by the model during prediction and understanding the patterns or characteristics of the data that lead to those errors. By conducting error analysis, you can gain insights into the model's weaknesses and identify areas for improvement.

# In[78]:


from sklearn.metrics import classification_report, confusion_matrix


# In[79]:


predict_classes = y_pred.argmax(axis = 1)


# In[81]:


confusion_matrix = confusion_matrix(y_test, predict_classes)


# In[82]:


confusion_matrix

