#!/usr/bin/env python
# coding: utf-8

# # Importing Python Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Machine Learning Libraries

# Machine learning libraries are software packages that provide tools and algorithms for building and deploying machine learning models. These libraries provide pre-built functions and modules for various tasks such as data preprocessing, feature selection, model training and evaluation, and deployment. Some of the popular machine learning libraries are:

# Scikit-learn:
#             It is an open-source machine learning library for Python. It provides simple and efficient tools for data mining and data analysis, as well as building and evaluating machine learning models.

# In[29]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# In[3]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV


# In[4]:


#creating random seed 
np.random.seed(42)


# # Readiing Dataset

# In[31]:


data = pd.read_csv(r"C:\Users\DELL\Desktop\python project\Data Analysis Projects\Car Sales Prediction\Datasets\car-sales-extended-missing-data.csv")


# # Exploring The Data

# Exploring the data is an essential step in any data analysis project, including those involving machine learning. Data exploration involves examining the data to understand its characteristics and gain insights into the relationships between variables.

# In[7]:


#top 5 head data from dataset
data.head()


# In[8]:


#bottom 5 data from dataset
data.tail()


# In[10]:


#shape of the data 1000 rows ans 5 columns
data.shape


# In[11]:


#information about the categorical data and the numerical data
data.info()


# # Correlation Matrix

# A correlation matrix is a table that displays the correlation coefficients between pairs of variables in a dataset. The correlation coefficient is a measure of the strength and direction of the linear relationship between two variables, ranging from -1 to 1. A positive correlation coefficient indicates a positive relationship between the variables, while a negative correlation coefficient indicates a negative relationship.

# In[32]:


corr_matrix = data.corr()


# In[35]:


plt.figure(figsize = (10, 4))
ax = sns.heatmap(corr_matrix,
                 annot = True, 
                 cmap = 'coolwarm')


# # Missing Value Filling
# 

# Method for filling in missing values depends on the specific dataset and the requirements of the machine learning algorithm being used. It is essential to carefully consider the potential impact of missing data on the analysis and to choose a method that provides the most accurate and reliable results.

# In[12]:


#checking for missing data from the dataset
data.isna().sum()


# In[30]:


#information about the dataset
data['Make'].info()


# In[16]:


#selecting the categorical values
data.select_dtypes(include = 'object')


# In[17]:


#viewing the numerical data from dataset
data.select_dtypes(include = ['float64', 'int64'])


# In[18]:


#dropping tthe price column from the dataset
data.dropna(subset=["Price"], inplace=True)


# # Encoding The Dataset

# Encoding is the process of converting categorical variables into numerical values that can be used in machine learning algorithms. There are several encoding techniques available, including:

# One-Hot Encoding:
#                 This technique creates a new binary feature for each unique category in the categorical variable. Each binary feature represents whether the observation belongs to that category or not. One-hot encoding is useful when the categories are not ordinal and do not have a natural ordering.

# # Encoding Categorical Values

# Encoding categorical values is an important step in preparing a dataset for machine learning algorithms. Categorical values refer to variables that take on a finite number of values or categories, such as gender, education level, or type of product. These values cannot be used directly in most machine learning algorithms, which require numerical inputs.

# In[19]:


#encoding the categorical values from the dataset using one hot encoder
categorical_features = ["Make", "Colour"]
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])


# In[20]:


door_feature = ["Doors"]
door_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=4))])


# # Encoding Numerical Values
# 

# Encoding numerical values is not necessary because numerical values are already in a format that can be used by machine learning algorithms. However, sometimes it is useful to transform numerical values into a different scale or range to improve the performance of the machine learning algorithm.

# In[21]:


#encoding the numerical values from the dataset using onehot encoder
numeric_features = ["Odometer (KM)"]
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])


# ColumnTransformer is a scikit-learn transformer that allows different transformations to be applied to different columns of a dataset. It is useful when there are both numerical and categorical variables in the dataset that require different preprocessing steps.

# In[22]:


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("door", door_transformer, door_feature),
        ("num", numeric_transformer, numeric_features)])


# In[23]:


model = Pipeline(steps=[("preprocessor", preprocessor),
                        ("model", RandomForestRegressor())])


# # Splitting The Datasets

# Splitting a dataset is a crucial step in machine learning to evaluate the performance of a trained model on unseen data. The dataset is usually split into two or three subsets: a training set, a validation set, and a testing set.

# The training set is used to train the model by adjusting its parameters using various machine learning algorithms. The validation set is used to tune hyperparameters, such as the regularization parameter or learning rate, to optimize the model's performance. Finally, the testing set is used to evaluate the final performance of the model on unseen data.

# In[24]:


X = data.drop("Price", axis=1)
y = data["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# # Training The Model

# Training a machine learning model involves feeding a dataset into an algorithm and adjusting the algorithm's parameters to optimize its performance on the dataset

# In[25]:


model.fit(X_train, y_train)
model.score(X_test, y_test)


# # Hyperparameter Tuning

# Hyperparameter tuning is the process of selecting the optimal hyperparameters for a machine learning model to achieve the best performance on a given task. Hyperparameters are parameters that cannot be learned directly from the training data but must be set manually before training the model, such as learning rate, regularization strength, or number of hidden layers.

# In[26]:


#applying the gridsearchCV
pipe_grid = {
    "preprocessor__num__imputer__strategy": ["mean", "median"],
    "model__n_estimators": [100, 1000],
    "model__max_depth": [None, 5],
    "model__max_features": ["auto", "sqrt"],
    "model__min_samples_split": [2, 4]
}


# In[27]:


gs_model = GridSearchCV(model, pipe_grid, cv=5, verbose=2)
gs_model.fit(X_train, y_train)


# In[28]:


#score after hyperparameter tuning
gs_model.score(X_test, y_test)


# In[ ]:




