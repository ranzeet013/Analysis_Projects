#!/usr/bin/env python
# coding: utf-8

# # Water Classification with CNN

# Water Classification project with CNN aims to develop a model that can classify water samples into safe or unsafe categories based on their quality parameters. Convolutional Neural Networks (CNNs) are utilized for their ability to effectively learn and extract features from image-like data, making them well-suited for image classification tasks.
# 
# 1 for Safe
# 
# 0 for Unsafe

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
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Dataset

# In[2]:


dataframe = pd.read_csv('waterQuality1.csv')


# # Data Processing

# Data processing is a crucial step in any machine learning project, including the Safe Water Classification project. It involves preparing the raw data for training a model by transforming, cleaning, and organizing it in a format suitable for analysis and model training.

# In[3]:


dataframe.head()


# In[4]:


dataframe.tail()


# In[5]:


dataframe.shape


# In[6]:


dataframe.head().T


# In[7]:


dataframe.info()


# In[9]:


dataframe.select_dtypes('object').columns


# In[11]:


dataframe.select_dtypes(['int64', 'float64']).columns


# In[12]:


dataframe.drop([7551,7568,7890],axis=0,inplace=True)


# In[13]:


dataframe.head()


# In[14]:


dataframe.columns


# In[15]:


dataframe['ammonia'] = dataframe['ammonia'].astype('float64')
dataframe['is_safe'] = dataframe['is_safe'].astype('int64')


# In[16]:


dataframe.info()


# In[17]:


dataframe.isna().sum()


# In[18]:


dataframe.isna().sum().any()


# # Statical Info

# Statistical information refers to numerical data or metrics that describe various aspects of a dataset or population. These statistics provide quantitative measures of central tendency, dispersion, relationships, and other properties of the data.

# In[19]:


dataframe.describe()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[20]:


corr_matrix = dataframe.corr()


# In[21]:


corr_matrix


# In[22]:


plt.figure(figsize = (18, 8))
sns.heatmap(
    corr_matrix, 
    annot = True, 
    cmap = 'magma'
)


# In[23]:


dataset = dataframe.drop('is_safe', axis = 1)


# In[24]:


dataset.corrwith(dataframe['is_safe']).plot.bar(
    figsize = (12, 5),
    title = 'Correlation With Safe',
    rot = 90
)


# In[25]:


dataframe.columns


# In[26]:


dataframe.head()


# # Splitting Data

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[27]:


x = dataframe.drop('is_safe', axis = 1)
y = dataframe['is_safe']


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 42)


# In[30]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Scaling

# Scaling is a preprocessing technique used in machine learning to transform the input features to a similar scale. It is often necessary because features can have different units, ranges, or magnitudes, which can affect the performance of certain algorithms. Scaling ensures that all features contribute equally to the learning process and prevents features with larger values from dominating those with smaller values.
# 
# StandardScaler follows the concept of standardization, also known as Z-score normalization. It transforms the features such that they have a mean of 0 and a standard deviation of 1. This process centers the feature distribution around 0 and scales it to a standard deviation of 1.

# In[31]:


from sklearn.preprocessing import StandardScaler


# In[32]:


scaler = StandardScaler()


# In[33]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[34]:


x_train


# In[35]:


x_test


# In[36]:


x_train.shape, x_test.shape


# In[37]:


x_train = x_train.reshape(6396, 20, 1)
x_test = x_test.reshape(1600, 20, 1)


# In[38]:


x_train.shape, x_test.shape


# # Building CNN Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.

# In[40]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import MaxPooling1D, Conv1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# In[48]:


model = Sequential()
model.add(Conv1D(32, kernel_size = 3, activation = 'relu', input_shape = (20, 1)))
model.add(MaxPooling1D(pool_size = 2))
model.add(BatchNormalization())

model.add(Conv1D(64, kernel_size = 3, activation = 'relu'))
model.add(MaxPooling1D(pool_size = 2))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation = 'sigmoid'))


# In[49]:


model.summary()


# In[50]:


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


model.save('model_pure_water_classifier.h5')


# # Learning Curve

# The learning curve is a plot that shows how the loss and accuracy of a model change during training. It provides insights into how well the model is learning from the training data and how it generalizes to unseen data. The learning curve typically shows the training and validation loss/accuracy on the y-axis and the number of epochs on the x-axis. By analyzing the learning curve, you can identify if the model is overfitting (high training loss, low validation loss) or underfitting (high training and validation loss). It is a useful tool for monitoring and evaluating the performance of machine learning models.

# In[54]:


losses = pd.DataFrame(model.history.history)


# In[56]:


losses.head()


# In[57]:


losses.plot()


# In[58]:


losses[['loss', 'val_loss']].plot()


# In[59]:


losses[['accuracy', 'val_accuracy']].plot()


# # Prediction 

# In[60]:


y_pred = model.predict(x_test)
predict_class = y_pred.argmax(axis = 1)


# In[62]:


print(y_test.iloc[20]), print(y_pred[20])


# In[66]:


print(y_test.iloc[44]), print(y_pred[44])


# # Error Analysis

# Error analysis is an important step in evaluating and improving the performance of a machine learning model. It involves analyzing the errors made by the model during prediction or classification tasks and gaining insights into the types of mistakes it is making. Error analysis can provide valuable information for model refinement and identifying areas for improvement

# In[67]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score


# The accuracy score is calculated using the following formula:
# 
# Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)

# In[68]:


accuracy_score = accuracy_score(y_test, predict_class)


# In[69]:


accuracy_score


# A classification report is a summary of various evaluation metrics for a classification model. It provides a comprehensive overview of the model's performance, including metrics such as precision, recall, F1 score, and support.

# In[71]:


classification_report = classification_report(y_test, predict_class)


# In[74]:


print(classification_report)


# A confusion matrix is a table that summarizes the performance of a classification model by showing the counts of true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions. It is a useful tool for evaluating the accuracy and effectiveness of a classification model.

# In[75]:


confusion_matrix = confusion_matrix(y_test, predict_class)


# In[76]:


confusion_matrix


# In[82]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix, 
            annot = True, 
            cmap = 'RdPu')

