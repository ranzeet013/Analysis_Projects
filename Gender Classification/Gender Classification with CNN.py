#!/usr/bin/env python
# coding: utf-8

# # Gender Classification with CNN

# The goal of this project is to develop a gender classification system using Convolutional Neural Networks (CNN). The system will take an input image containing a person's face and predict the gender of that person as male or female.

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


# # Reading Dataset

# In[3]:


dataframe = pd.read_csv('gender_classification_v7.csv')


# # Exploratory Data Analysis

# The process of analyzing and understanding a dataset to gain insights and identify patterns or trends. The goal of exploring the data is to become familiar with its structure, distribution, and quality, as well as to identify potential issues or anomalies that may need to be addressed before further analysis.

# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe['gender'] = dataframe['gender'].map({'Female':0, 'Male':1})


# In[7]:


dataframe.head()


# In[8]:


dataframe['gender'].value_counts()


# In[9]:


dataframe.shape


# In[10]:


dataframe.info()


# In[11]:


dataframe.isna().sum().any()


# In[12]:


dataframe.isna().sum()


# In[14]:


dataframe['gender'].value_counts().plot(kind = 'bar',
                                         figsize = (10, 5), 
                                         title = 'Gender', 
                                         rot = 90, 
                                         cmap = 'magma')


# # Statical Info

# Statistical information refers to numerical data or metrics that describe various aspects of a dataset or population. These statistics provide quantitative measures of central tendency, dispersion, relationships, and other properties of the data

# In[15]:


dataframe.describe()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[16]:


corr_matrix = dataframe.corr()


# In[17]:


corr_matrix


# In[18]:


plt.figure(figsize = (15, 8))
sns.heatmap(corr_matrix,
            annot = True, 
            cmap = 'inferno')


# In[19]:


dataset = dataframe.drop('gender', axis = 1)


# In[20]:


dataset.head()


# In[21]:


dataset.corrwith(dataframe['gender']).plot.bar(
    figsize = (12, 5), 
    title = 'Correlation with Gender', 
    cmap = 'Wistia', 
    rot =90
)


# In[22]:


dataframe.head()


# In[23]:


dataframe.columns


# In[24]:


from pandas_profiling import ProfileReport


# In[26]:


profile_report = ProfileReport(dataframe, minimal = True)
profile_report.to_file('Gender_Profile_Report.html')
profile_report


# In[76]:


dataframe.head()


# # Splitting Dataset

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[27]:


x = dataframe.drop('gender', axis = 1)
y = dataframe['gender']


# In[28]:


x.shape, y.shape


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 101)


# In[31]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Scaling

# Scaling is a preprocessing technique used in machine learning to transform the input features to a similar scale. It is often necessary because features can have different units, ranges, or magnitudes, which can affect the performance of certain algorithms. Scaling ensures that all features contribute equally to the learning process and prevents features with larger values from dominating those with smaller values.
# 
# StandardScaler is a commonly used method for scaling numerical features in machine learning. It is part of the preprocessing module in scikit-learn, a popular machine learning library in Python.

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


x_train.shape, x_test.shape


# # Reshaping

# Reshaping a tensor involves rearranging its elements into a new shape without changing their values. The reshaping operation can be applied to tensors of different dimensions, such as converting a 1D tensor into a 2D tensor or vice versa. Reshaping is commonly performed using the reshape() function or method available in most deep learning frameworks.

# In[38]:


x_train = x_train.reshape(4000, 7, 1)
x_test = x_test.reshape(1001, 7, 1)


# In[39]:


x_train.shape, x_test.shape


# # Building Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.

# In[40]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# In[48]:


model = Sequential()
model.add(Conv1D(32, kernel_size = 2, padding = 'same', activation = 'relu', input_shape = (7, 1)))
model.add(MaxPooling1D(pool_size = 2))
model.add(BatchNormalization())
model.add(Dropout(0.))

model.add(Conv1D(64, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling1D(pool_size = 2))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation = 'sigmoid'))


# In[49]:


model.summary()


# Early stopping is a technique used during the training of machine learning models to prevent overfitting and find the optimal point at which to stop training. It involves monitoring the performance of the model on a validation dataset and stopping the training process when the model's performance on the validation dataset starts to degrade.

# In[78]:


early_stop = EarlyStopping(monitor = 'val_loss', 
                           patience = 3, 
                           restore_best_weights = True)


# # Compiling Model

# Compiling the model in deep learning involves configuring essential components that define how the model will be trained.

# In[51]:


model.compile(optimizer = Adam(0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])


# # Training Model

# Training the model in deep learning involves the process of iteratively updating the model's parameters (weights and biases) based on the provided training data to minimize the loss function and improve the model's performance.

# In[52]:


model.fit(x_train, y_train, 
          validation_data = (x_test, y_test), 
          epochs = 250, 
          callbacks = [early_stop])


# In[53]:


model.save('model_gender_classification.h5')


# # Learning Curve

# The learning curve is a plot that shows how the loss and accuracy of a model change during training. It provides insights into how well the model is learning from the training data and how it generalizes to unseen data. The learning curve typically shows the training and validation loss/accuracy on the y-axis and the number of epochs on the x-axis. By analyzing the learning curve, you can identify if the model is overfitting (high training loss, low validation loss) or underfitting (high training and validation loss). It is a useful tool for monitoring and evaluating the performance of machine learning models.

# In[54]:


loss = pd.DataFrame(model.history.history)


# In[55]:


loss.head()


# In[56]:


loss.plot()


# In[57]:


loss[['loss', 'val_loss']].plot()


# In[59]:


loss[['accuracy', 'val_accuracy']].plot()


# # Prediction :

# In[60]:


y_pred = model.predict(x_test)
predict_class = y_pred.argmax(axis = 1)


# In[63]:


print(y_test.iloc[2]), print(predict_class[2])


# In[66]:


print(y_test.iloc[4]), print(y_pred[45])


# # Error Analysis

# Error analysis is an important step in evaluating and improving the performance of a machine learning model. It involves analyzing the errors made by the model during prediction or classification tasks and gaining insights into the types of mistakes it is making. Error analysis can provide valuable information for model refinement and identifying areas for improvement

# In[67]:


from sklearn.metrics import confusion_matrix


# In[71]:


confusion_matrix = confusion_matrix(y_test, predict_class)


# In[72]:


confusion_matrix


# In[73]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix, 
            annot = True, 
            cmap = 'RdPu')

