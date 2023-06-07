#!/usr/bin/env python
# coding: utf-8

# # Milk Grade Classification with CNN

# The milk grade classification project using CNN (Convolutional Neural Network) aims to classify milk samples into different grades based on their quality. The project involves using deep learning techniques to analyze milk data and make predictions about its grade.

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
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Dataset

# In[2]:


dataframe = pd.read_csv('milknew.csv')


# # Exploratory Data Analysis

# The process of analyzing and understanding a dataset to gain insights and identify patterns or trends. The goal of exploring the data is to become familiar with its structure, distribution, and quality, as well as to identify potential issues or anomalies that may need to be addressed before further analysis.

# In[3]:


dataframe.head()


# In[4]:


dataframe.tail()


# In[5]:


dataframe.columns


# In[6]:


dataframe.info()


# In[7]:


dataframe.isna().sum()


# In[8]:


dataframe.shape


# In[9]:


dataframe['Grade'] = dataframe['Grade'].map({'low':0, 'medium':1, 'high':2})


# In[10]:


dataframe.head()


# # Statical Info

# Statistical information refers to numerical data or metrics that describe various aspects of a dataset or population. These statistics provide quantitative measures of central tendency, dispersion, relationships, and other properties of the data.

# In[11]:


dataframe.describe()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[12]:


corr_matrix = dataframe.corr()


# In[13]:


corr_matrix


# In[14]:


plt.figure(figsize = (12, 6))
sns.heatmap(
    corr_matrix, 
    annot = True, 
    cmap = 'coolwarm'
)


# In[15]:


dataframe.columns


# In[16]:


dataset = dataframe.drop('Grade', axis = 1)


# In[17]:


dataset.head()


# In[18]:


dataset.corrwith(dataframe['Grade']).plot.bar(
    figsize = (12, 5),
    title = 'Correlation wih Grade',
    rot = 90
)


# In[19]:


dataframe.columns


# In[20]:


dataframe.head()


# In[21]:


x = dataframe.drop('Grade', axis = 1)
y = dataframe['Grade']


# # Splitting Dataset

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 42)


# In[24]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Scaling

# Scaling is a preprocessing technique used in machine learning to transform the input features to a similar scale. It is often necessary because features can have different units, ranges, or magnitudes, which can affect the performance of certain algorithms. Scaling ensures that all features contribute equally to the learning process and prevents features with larger values from dominating those with smaller values.
# 
# StandardScaler is a commonly used method for scaling numerical features in machine learning. It is part of the preprocessing module in scikit-learn, a popular machine learning library in Python.
# 
# StandardScaler follows the concept of standardization, also known as Z-score normalization. It transforms the features such that they have a mean of 0 and a standard deviation of 1. This process centers the feature distribution around 0 and scales it to a standard deviation of 1.

# In[25]:


from sklearn.preprocessing import StandardScaler


# In[26]:


scaler = StandardScaler()


# In[27]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[28]:


x_train


# In[29]:


x_test


# # Reshaping 

# Reshaping a tensor involves rearranging its elements into a new shape without changing their values. The reshaping operation can be applied to tensors of different dimensions, such as converting a 1D tensor into a 2D tensor or vice versa. Reshaping is commonly performed using the reshape() function or method available in most deep learning frameworks.

# In[30]:


x_train.shape, x_test.shape


# In[31]:


x_train = x_train.reshape(847, 7, 1)
x_test = x_test.reshape(212, 7, 1)


# In[32]:


x_train.shape, x_test.shape


# # Building Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.

# In[33]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# In[34]:


model = Sequential()
model.add(Conv1D(64, kernel_size = 2, activation = 'relu', input_shape = (7, 1)))
model.add(MaxPooling1D(pool_size = 2))
model.add(BatchNormalization())
model.add(Conv1D(128, kernel_size = 2, activation = 'relu'))
model.add(MaxPooling1D(pool_size = 2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation = 'softmax'))


# In[35]:


model.summary()


# Early stopping is a technique used during the training of machine learning models to prevent overfitting and find the optimal point at which to stop training. It involves monitoring the performance of the model on a validation dataset and stopping the training process when the model's performance on the validation dataset starts to degrade.

# In[36]:


early_stop = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)


# # Compiling Model

# Compiling the model in deep learning involves configuring essential components that define how the model will be trained.

# In[37]:


model.compile(optimizer = Adam(0.0001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# # Training Model

# Training the model in deep learning involves the process of iteratively updating the model's parameters (weights and biases) based on the provided training data to minimize the loss function and improve the model's performance.

# In[38]:


model.fit(x_train, y_train, 
          epochs = 100, 
          validation_data = (x_test, y_test), 
          callbacks = [early_stop])


# In[39]:


model.save('model_milk_grade_classification.h5')


# # Learning Curve

# The learning curve is a plot that shows how the loss and accuracy of a model change during training. It provides insights into how well the model is learning from the training data and how it generalizes to unseen data. The learning curve typically shows the training and validation loss/accuracy on the y-axis and the number of epochs on the x-axis. By analyzing the learning curve, you can identify if the model is overfitting (high training loss, low validation loss) or underfitting (high training and validation loss). It is a useful tool for monitoring and evaluating the performance of machine learning models.

# In[40]:


loss = pd.DataFrame(model.history.history)


# In[41]:


loss.head()


# In[42]:


loss.plot()


# In[43]:


loss[['loss', 'val_loss']].plot()


# In[44]:


loss[['accuracy', 'val_accuracy']].plot()


# # Prediction

# In[45]:


y_pred = model.predict(x_test)
predict_class = y_pred.argmax(axis = 1)


# In[46]:


print(y_test.iloc[20]), print(predict_class[20])


# # Error Analysis

# Error analysis is an important step in evaluating and improving the performance of a machine learning model. It involves analyzing the errors made by the model during prediction or classification tasks and gaining insights into the types of mistakes it is making. Error analysis can provide valuable information for model refinement and identifying areas for improvement

# In[47]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# # Confusion Matrix

# A confusion matrix is a table that summarizes the performance of a classification model by showing the counts of true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions. It is a useful tool for evaluating the accuracy and effectiveness of a classification model.

# In[48]:


print(confusion_matrix(y_test, predict_class))


# # Classification Report

# A classification report is a summary of various evaluation metrics for a classification model. It provides a comprehensive overview of the model's performance, including metrics such as precision, recall, F1 score, and support.

# In[49]:


print(classification_report(y_test, predict_class))


# # Accuracy Score

# Accuracy score is a commonly used metric to evaluate the performance of a classification model. It measures the proportion of correct predictions made by the model out of the total number of predictions.
# 
# The accuracy score is calculated using the following formula:
# 
# Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)

# In[50]:


print(accuracy_score(y_test, predict_class))

