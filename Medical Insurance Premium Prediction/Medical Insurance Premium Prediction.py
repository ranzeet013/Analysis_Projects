#!/usr/bin/env python
# coding: utf-8

# # Medical Insurance Premium Prediction

# The Medical Insurance Premium Prediction project involves predicting the insurance premium that individuals are likely to pay based on various factors such as age, gender, BMI (Body Mass Index), smoking habits, region, and number of dependents. The goal is to develop a predictive model that can accurately estimate insurance premiums for new individuals based on their demographic and health-related attributes.

# # Importing Libraries

# These are just a few examples of popular Python libraries. You can import any other library using the same import statement followed by the library name or alias:
# 
# NumPy: for numerical operations and array manipulation
# 
# Pandas: for data manipulation and analysis
# 
# Matplotlib: for creating visualizations
# 
# Scikit-learn: for machine learning algorithms.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Dataset

# In[2]:


dataframe = pd.read_csv('insurance.csv')


# # Exploratory Data Analysis

# Exploratory Data Analysis (EDA) is a crucial initial step in the data analysis process. It involves examining and understanding the structure, patterns, and characteristics of the data before applying any specific modeling techniques. EDA helps to uncover insights, identify patterns, detect anomalies, and make informed decisions about data preprocessing and modeling strategies.

# In[3]:


dataframe.head()


# In[4]:


dataframe.tail()


# In[5]:


dataframe.shape


# In[6]:


dataframe.info()


# In[8]:


dataframe.select_dtypes(include = 'object').head()


# In[9]:


dataframe.select_dtypes(include = ['int64', 'float64']).head()


# In[11]:


dataframe.columns


# In[13]:


dataframe.groupby('sex').mean()


# In[14]:


dataframe.groupby('smoker').mean()


# In[15]:


dataframe.groupby('region').mean()


# # Handeling Missing Values

# In[16]:


dataframe.isna().sum()


# In[17]:


dataframe.isna().sum().any()


# In[22]:


dataframe.isna().values.any()


# # Statical Info

# Statistical information refers to numerical data or metrics that describe various aspects of a dataset or population. These statistics provide quantitative measures of central tendency, dispersion, relationships, and other properties of the data.

# In[23]:


dataframe.describe()


# # Encoding Categorical Values

# Encoding in the context of machine learning refers to the process of converting categorical or textual data into a numerical representation that can be used by machine learning algorithms.

# In[24]:


dataframe.select_dtypes(include = 'object').columns


# In[25]:


dataframe.head()


# In[27]:


dataframe = pd.get_dummies(data = dataframe, drop_first = True)


# In[29]:


dataframe.head()


# In[30]:


dataframe.shape


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[31]:


dataset = dataframe.drop('charges', axis = 1)


# In[35]:


dataset.corrwith(dataframe['charges']).plot.bar(
    figsize = (12, 4), 
    title = 'Correlation With Charges',
    rot = 90
)


# In[36]:


corr_matrix = dataframe.corr()


# In[37]:


corr_matrix


# In[38]:


plt.figure(figsize = (12, 5))
sns.heatmap(
    corr_matrix, 
    annot = True, 
    cmap = 'coolwarm'
)


# In[39]:


dataframe.head()


# In[40]:


x = dataframe.drop('charges', axis = 1)
y = dataframe['charges']


# # Splitting Dataset

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.

# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size = 0.2, 
    random_state = 42
)


# In[43]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Scaling

# Scaling is a preprocessing technique used in machine learning to transform the input features to a similar scale. It is often necessary because features can have different units, ranges, or magnitudes, which can affect the performance of certain algorithms. Scaling ensures that all features contribute equally to the learning process and prevents features with larger values from dominating those with smaller values.

# StandardScaler is a commonly used method for scaling numerical features in machine learning. It is part of the preprocessing module in scikit-learn, a popular machine learning library in Python.
# 
# StandardScaler follows the concept of standardization, also known as Z-score normalization. It transforms the features such that they have a mean of 0 and a standard deviation of 1. This process centers the feature distribution around 0 and scales it to a standard deviation of 1.

# In[44]:


from sklearn.preprocessing import StandardScaler


# In[45]:


scaler = StandardScaler()


# In[46]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[48]:


x_train


# In[49]:


x_test


# # Linear Regression

# Linear regression is a statistical method used to predict a continuous outcome based on one or more input variables. It assumes a linear relationship between the input variables and the target variable. The formula for linear regression can be represented as y = β0 + β1 * x, where y is the predicted outcome, x is the input variable, β0 is the y-intercept, and β1 is the slope of the line. The goal of linear regression is to find the best-fit line that minimizes the difference between the predicted values and the actual values in the training data. This line is then used to make predictions on new data points by substituting the input values into the equation.

# y = β0 + β1 * x

# In[51]:


from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)


# In[52]:


y_pred = linear_reg.predict(x_test)


# # R2 Score

# The R-squared (R2) score, also known as the coefficient of determination, is a statistical measure used to evaluate the goodness of fit of a regression model. It represents the proportion of the variance in the dependent variable that can be explained by the independent variables in the model.

# Mathematically, the R2 score is calculated as:
# 
# R2 = 1 - (SSres / SStot)

# In[53]:


from sklearn.metrics import r2_score


# In[54]:


r2_score(y_test, y_pred)


# # Random Forest Regressor

# Random Forest Regressor is a machine learning algorithm that belongs to the ensemble learning category. It is a popular and powerful algorithm used for regression tasks. Random Forest Regressor combines the principles of decision trees and bagging to create an ensemble of decision trees and make predictions.

# In[55]:


from sklearn.ensemble import RandomForestRegressor
random = RandomForestRegressor()
random.fit(x_train, y_train)


# In[56]:


y_pred = random.predict(x_test)


# In[57]:


r2_score(y_test, y_pred)


# In[62]:


dataframe.head()


# # Prediction On Dummy Data

# # 1st Prediction

# Name = Luffy
# Age = 20
# bmi = 28.88
# children = 0
# sex = male
# smoker = no
# region = southeast

# In[68]:


luffy_obs = [[20, 28.88, 0, 0, 1, 0, 0, 1]]


# In[69]:


luffy_obs


# In[70]:


random.predict(scaler.transform(luffy_obs))


# # 2nd Prediction

# Name = yamato
# Age = 24
# bmi = 25.88
# children = 2
# sex = female
# smoker = yes
# region = southeast

# In[71]:


yamato_obs = [[24, 25.88, 2, 0, 0, 0, 1, 1]]


# In[72]:


yamato_obs


# In[73]:


random.predict(scaler.transform(yamato_obs))

