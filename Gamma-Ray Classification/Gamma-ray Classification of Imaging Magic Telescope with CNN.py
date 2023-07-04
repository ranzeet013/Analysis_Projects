#!/usr/bin/env python
# coding: utf-8

# # Gamma-ray Classification of Imaging Magic Telescope with CNN

# Gamma-ray classification of imaging telescopes involves the development of a system that can accurately classify gamma-ray events detected by telescopes based on their characteristics. This type of project is typically focused on data analysis and machine learning techniques.

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


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Importing Dataset

# In[3]:


dataframe = pd.read_csv('MagicTelescope.csv')


# # Exploring Dataset

# Exploring a dataset involves analyzing and understanding its structure, content, and characteristics. It helps in gaining insights into the data, identifying patterns, and making informed decisions about data preprocessing, feature engineering, and modeling.

# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe.shape


# In[7]:


dataframe['class:'] = dataframe['class:'].map({'g':0, 'h':1})


# In[8]:


dataframe.head()


# In[9]:


dataframe = dataframe.drop(['id', 'ID'], axis = 1)


# In[10]:


dataframe.head()


# In[11]:


dataframe.info()


# # Checking Null Values

# In[12]:


dataframe.isna().any()


# In[13]:


dataframe['class:'].value_counts()


# In[14]:


dataframe['class:'].value_counts().plot(kind = 'bar', 
                                        figsize = (10, 4), 
                                        cmap = 'coolwarm')


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


plt.figure(figsize = (12, 6))
sns.heatmap(corr_matrix, 
            annot = True, 
            cmap = 'inferno')


# In[19]:


dataframe.columns


# # Correlation Diagram for Ray Class

# A correlation diagram, also known as a correlation matrix or correlation heatmap, is a graphical representation that displays the correlation coefficients between variables in a dataset. It provides a visual summary of the relationships between pairs of variables, allowing you to quickly identify patterns and dependencies.

# In[20]:


dataset = dataframe.drop('class:', axis = 1)


# In[21]:


dataset.head()


# In[22]:


dataset.corrwith(dataframe['class:']).plot.bar(
    figsize = (12, 5), 
    title = 'Correlation with Glass type',
    cmap = 'ocean'
)


# In[23]:


dataframe.head()


# # Splitting Dataset

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[24]:


x = dataframe.drop('class:', axis = 1)
y = dataframe['class:']


# In[25]:


x.shape, y.shape


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state  = 101)


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


# # Reshaping

# Reshaping a tensor involves rearranging its elements into a new shape without changing their values. The reshaping operation can be applied to tensors of different dimensions, such as converting a 1D tensor into a 2D tensor or vice versa. Reshaping is commonly performed using the reshape() function or method available in most deep learning frameworks.

# In[34]:


x_train.shape, x_test.shape


# In[35]:


x_train = x_train.reshape(15216, 10, 1)
x_test = x_test.reshape(3804, 10, 1)


# In[36]:


x_train.shape, x_test.shape


# # Building Model

# Building a CNN (Convolutional Neural Network) model involves constructing a deep learning architecture specifically designed for image processing and analysis. CNNs are highly effective in capturing spatial patterns and features from images, making them a popular choice for tasks like image classification, object detection, and image segmentation.

# In[37]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# In[38]:


model = Sequential()
model.add(Conv1D(filters = 32, kernel_size = 2,activation = 'relu', input_shape = (10, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters = 64,kernel_size = 2, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


# In[39]:


model.summary()


# Early stopping is a technique used during the training of machine learning models to prevent overfitting and find the optimal point at which to stop training. It involves monitoring the performance of the model on a validation dataset and stopping the training process when the model's performance on the validation dataset starts to degrade.

# In[40]:


early_stop = EarlyStopping(
    monitor = 'val_loss',
    patience = 5,
    restore_best_weights = True
)


# # Compiling Model

# Compiling a model in the context of deep learning refers to configuring its training process. It involves specifying the optimizer, the loss function, and any additional metrics that will be used during the training phase. Compiling a model is an essential step before training it on a dataset

# In[41]:


model.compile(optimizer = Adam(0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])


# # Training Model

# Training a model in the context of deep learning involves the process of fitting the model to a training dataset, allowing it to learn from the data and adjust its internal parameters (weights and biases) to make accurate predictions.

# In[42]:


model.fit(
    x_train, y_train,
    validation_data = (x_test, y_test),
    epochs = 250, 
    callbacks = [early_stop])


# In[43]:


model.save('model_gamma_ray_classifier.h5')


# # Learning Curve

# The learning curve is a plot that shows how the loss and accuracy of a model change during training. It provides insights into how well the model is learning from the training data and how it generalizes to unseen data. The learning curve typically shows the training and validation loss/accuracy on the y-axis and the number of epochs on the x-axis. By analyzing the learning curve, you can identify if the model is overfitting (high training loss, low validation loss) or underfitting (high training and validation loss). It is a useful tool for monitoring and evaluating the performance of machine learning models.

# In[44]:


losses = pd.DataFrame(model.history.history)


# In[45]:


losses.head()


# In[46]:


losses.plot()


# # Loss Curve

# A loss curve is a graphical representation that shows how the loss of a machine learning model changes over the course of training. Loss refers to the discrepancy between the predicted output of the model and the true or expected output. The loss curve helps in monitoring the progress of model training and assessing the convergence and performance of the model.

# In[47]:


losses[['loss', 'val_loss']].plot()


# # Accuracy Curve

# An accuracy curve is a graphical representation that depicts how the accuracy of a machine learning model changes as a specific aspect of the model or training process varies. The accuracy curve is typically plotted against the varying parameter or condition to analyze its impact on the model's accuracy.

# In[48]:


losses[['accuracy', 'val_accuracy']].plot()


# # Predicting on x-Test

# In[49]:


y_pred = model.predict(x_test)
predict_class = y_pred.argmax(axis = 1)


# In[50]:


print(y_test[100]), print(y_pred[100])


# In[51]:


print(y_test.iloc[1700]), print(predict_class[1700])


# # Error Analysis

# Error analysis is an important step in evaluating and improving the performance of a machine learning model. It involves analyzing the errors made by the model during prediction or classification tasks and gaining insights into the types of mistakes it is making. Error analysis can provide valuable information for model refinement and identifying areas for improvement

# In[52]:


from sklearn.metrics import confusion_matrix


# A confusion matrix, also known as an error matrix, is a table that summarizes the performance of a classification model by comparing predicted class labels with actual class labels. It provides a comprehensive view of the model's predictions and the associated errors. The confusion matrix is typically used in supervised learning tasks, where the true class labels are known.

# In[53]:


confusion_matrix = confusion_matrix(y_test, predict_class)


# In[54]:


confusion_matrix


# In[55]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix, 
            annot = True, 
            cmap = 'RdPu')


# # Thanks
