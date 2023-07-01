#!/usr/bin/env python
# coding: utf-8

# # Predicting Titanic Survival with CNN

# The goal of a project predicting Titanic survival with a CNN (Convolutional Neural Network) could be to develop a machine learning model that can accurately classify passengers as survivors or non-survivors based on their characteristics and features. The project may involve using the CNN architecture, which is well-suited for analyzing visual data like images, to extract meaningful patterns and features from the available data.

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


dataframe = pd.read_csv('SVMtrain.csv')


# # Exploring Dataset

# Exploring a dataset involves analyzing and understanding its structure, content, and characteristics. It helps in gaining insights into the data, identifying patterns, and making informed decisions about data preprocessing, feature engineering, and modeling.

# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe['Sex'] = dataframe['Sex'].map({'female':0, 'Male':1})


# In[7]:


dataframe.head()


# In[8]:


dataframe['Sex'].value_counts()


# In[9]:


dataframe.shape


# In[10]:


dataframe.info()


# In[11]:


dataframe.isna().sum().any()


# In[12]:


dataframe.isna().sum()


# In[13]:


dataframe['Survived'].value_counts().plot(kind = 'bar',
                                         figsize = (10, 5), 
                                         title = 'Survived', 
                                         rot = 90, 
                                         cmap = 'magma')


# # Statical Info

# Statistical information refers to numerical data or metrics that describe various aspects of a dataset or population. These statistics provide quantitative measures of central tendency, dispersion, relationships, and other properties of the data.

# In[14]:


dataframe.describe()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[15]:


corr_matrix = dataframe.corr()


# In[16]:


corr_matrix


# In[40]:


plt.figure(figsize = (12, 6))
sns.heatmap(corr_matrix,
            annot = True, 
            cmap = 'inferno')


# In[19]:


dataset = dataframe.drop('Survived', axis = 1)


# In[20]:


dataset.head()


# # Correlation Diagram

# A correlation diagram, also known as a correlation matrix or correlation heatmap, is a graphical representation that displays the correlation coefficients between variables in a dataset. It provides a visual summary of the relationships between pairs of variables, allowing you to quickly identify patterns and dependencies.

# In[21]:


dataset.corrwith(dataframe['Survived']).plot.bar(
    figsize = (12, 5), 
    title = 'Correlation with Survived', 
    cmap = 'ocean', 
    rot =90
)


# In[22]:


dataframe.head()


# In[23]:


dataframe.columns


# # Splitting Dataset

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[24]:


x = dataframe.drop('Survived', axis = 1)
y = dataframe['Survived']


# In[25]:


x.shape, y.shape


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 101)


# In[28]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Scaling

# Scaling is a preprocessing technique used in machine learning to transform the input features to a similar scale. It is often necessary because features can have different units, ranges, or magnitudes, which can affect the performance of certain algorithms. Scaling ensures that all features contribute equally to the learning process and prevents features with larger values from dominating those with smaller values.
# 
# StandardScaler is a commonly used method for scaling numerical features in machine learning. It is part of the preprocessing module in scikit-learn, a popular machine learning library in Python.

# In[29]:


from sklearn.preprocessing import StandardScaler


# In[30]:


scaler = StandardScaler()


# In[31]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[32]:


x_train


# In[33]:


x_test


# In[34]:


x_train.shape, x_test.shape


# # Reshaping

# Reshaping a tensor involves rearranging its elements into a new shape without changing their values. The reshaping operation can be applied to tensors of different dimensions, such as converting a 1D tensor into a 2D tensor or vice versa. Reshaping is commonly performed using the reshape() function or method available in most deep learning frameworks.

# In[35]:


x_train = x_train.reshape(711, 8, 1)
x_test = x_test.reshape(178, 8, 1)


# In[36]:


x_train.shape, x_test.shape


# # Building Model

# Building a CNN (Convolutional Neural Network) model involves constructing a deep learning architecture specifically designed for image processing and analysis. CNNs are highly effective in capturing spatial patterns and features from images, making them a popular choice for tasks like image classification, object detection, and image segmentation.

# In[37]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# In[44]:


model = Sequential()
model.add(Conv1D(32, kernel_size = 2, padding = 'same', activation = 'relu', input_shape = (8, 1)))
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


# In[45]:


model.summary()


# Early stopping is a technique used during the training of machine learning models to prevent overfitting and find the optimal point at which to stop training. It involves monitoring the performance of the model on a validation dataset and stopping the training process when the model's performance on the validation dataset starts to degrade.

# In[46]:


early_stop = EarlyStopping(monitor = 'val_loss', 
                           patience = 3, 
                           restore_best_weights = True)


# # Compiling Model

# Compiling a model in the context of deep learning refers to configuring its training process. It involves specifying the optimizer, the loss function, and any additional metrics that will be used during the training phase. Compiling a model is an essential step before training it on a dataset

# In[47]:


model.compile(optimizer = Adam(0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])


# # Training Model

# Training a model in the context of deep learning involves the process of fitting the model to a training dataset, allowing it to learn from the data and adjust its internal parameters (weights and biases) to make accurate predictions.

# In[48]:


model.fit(x_train, y_train, 
          validation_data = (x_test, y_test), 
          epochs = 250, 
          callbacks = [early_stop])


# In[49]:


model.save('survival_pred_model.h5')


# # Learning Curve

# The learning curve is a plot that shows how the loss and accuracy of a model change during training. It provides insights into how well the model is learning from the training data and how it generalizes to unseen data. The learning curve typically shows the training and validation loss/accuracy on the y-axis and the number of epochs on the x-axis. By analyzing the learning curve, you can identify if the model is overfitting (high training loss, low validation loss) or underfitting (high training and validation loss). It is a useful tool for monitoring and evaluating the performance of machine learning models.

# In[50]:


losses = pd.DataFrame(model.history.history)


# In[51]:


losses.head()


# In[52]:


losses.plot()


# # Accuracy Curve

# An accuracy curve is a graphical representation that depicts how the accuracy of a machine learning model changes as a specific aspect of the model or training process varies. The accuracy curve is typically plotted against the varying parameter or condition to analyze its impact on the model's accuracy.

# In[53]:


losses[['accuracy', 'val_accuracy']].plot()


# # Loss Curve

# A loss curve is a graphical representation that shows how the loss of a machine learning model changes over the course of training. Loss refers to the discrepancy between the predicted output of the model and the true or expected output. The loss curve helps in monitoring the progress of model training and assessing the convergence and performance of the model.

# In[54]:


losses[['loss', 'val_loss']].plot()


# # Prediction on x-Test Data

# In[55]:


y_pred = model.predict(x_test)
predict_class = y_pred.argmax(axis = 1)


# In[57]:


print(y_test.iloc[5]), print(predict_class[5])


# In[58]:


print(y_test.iloc[4]), print(y_pred[45])


# # Error Analysis

# Error analysis is an important step in evaluating and improving the performance of a machine learning model. It involves analyzing the errors made by the model during prediction or classification tasks and gaining insights into the types of mistakes it is making. Error analysis can provide valuable information for model refinement and identifying areas for improvement
# 
# 

# In[59]:


from sklearn.metrics import confusion_matrix


# A confusion matrix, also known as an error matrix, is a table that summarizes the performance of a classification model by comparing predicted class labels with actual class labels. It provides a comprehensive view of the model's predictions and the associated errors. The confusion matrix is typically used in supervised learning tasks, where the true class labels are known.

# In[63]:


confusion_matrix = confusion_matrix(y_test, predict_class)


# In[64]:


confusion_matrix


# In[65]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix, 
            annot = True, 
            cmap = 'RdPu')


# # Thanks !
