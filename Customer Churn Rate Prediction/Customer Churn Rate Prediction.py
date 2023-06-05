#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Rate Prediction

# Customer churn rate prediction is a common project in the field of customer analytics and customer relationship management. It involves developing a machine learning model to predict which customers are likely to churn or discontinue their relationship with a business or service provider. The goal is to proactively identify customers at risk of churning and take appropriate actions to retain them.

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
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Dataset

# In[2]:


dataframe = pd.read_csv('Churn_Modelling.csv')


# # Exploratory Data Analysis

# Exploratory Data Analysis (EDA) is a process in data analysis where the primary objective is to gain insights and understanding of a dataset. It involves examining and summarizing the main characteristics of the data using various statistical and visualization techniques. The key steps in EDA typically include data cleaning, data transformation, and data visualization.

# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe.info()


# In[7]:


dataframe.select_dtypes(include = 'object').head()


# In[8]:


dataframe.select_dtypes(include = ['float64', 'int64']).head()


# In[9]:


dataframe.isna().sum()


# # Statical Info

# Statistical information refers to numerical data or metrics that describe various aspects of a dataset or population. These statistics provide quantitative measures of central tendency, dispersion, relationships, and other properties of the data

# In[10]:


dataframe.describe()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[11]:


corr_matrix = dataframe.corr()


# In[12]:


corr_matrix


# In[13]:


plt.figure(figsize = (12, 5))
sns.heatmap(
    corr_matrix, 
    annot = True,
    cmap = 'coolwarm'
)


# In[14]:


dataframe.columns


# In[15]:


dataframe.head()


# In[16]:


dataframe.select_dtypes(include = 'object').columns


# In[17]:


dataframe = dataframe.drop(['Surname', 'Geography', 'Gender'], axis = 1)


# In[18]:


dataframe.head()


# In[19]:


dataframe = pd.get_dummies(data = dataframe, drop_first = True)


# In[20]:


dataframe.head()


# In[21]:


x = dataframe.drop('Exited', axis = 1)
y = dataframe['Exited']


# # Splitting Dataset

# Dataset splitting is an important step in machine learning and data analysis. It involves dividing a dataset into two or more subsets to train and evaluate a model effectively. The most common type of dataset splitting is into training and testing subsets.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size = 0.2, 
    random_state = 42
)


# In[24]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Scaling

# Scaling is a preprocessing step in machine learning that involves transforming the features or variables of your dataset to a consistent scale. It is important because many machine learning algorithms are sensitive to the scale of the input features. Scaling helps ensure that all features have a similar range and distribution, which can improve the performance and convergence of the model.
# 
# StandardScaler is a popular scaling technique used in machine learning to standardize features by removing the mean and scaling to unit variance. It is available in the scikit-learn library, which provides a wide range of machine learning tools and preprocessing functions.

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


# # Logistic Regression

# Logistic regression is a statistical modeling technique used to predict binary or categorical outcomes. It is a type of regression analysis that models the relationship between a dependent variable and one or more independent variables. However, unlike linear regression, which predicts continuous numeric values, logistic regression predicts the probability of an event or the likelihood of belonging to a particular category.

# In[30]:


from sklearn.linear_model import LogisticRegression


# In[31]:


log_reg = LogisticRegression()


# In[32]:


log_reg.fit(x_train, y_train)


# # Random Forest Classifier

# The Random Forest Classifier is a machine learning algorithm that is commonly used for classification tasks. It is an ensemble learning method that combines multiple decision trees to make predictions. Random Forests are versatile and can handle both categorical and continuous input variables.

# In[42]:


from sklearn.ensemble import RandomForestClassifier


# In[43]:


clf = RandomForestClassifier()


# In[44]:


clf.fit(x_train, y_train)


# # XGBoost Classifier

# The XGBoost Classifier is an advanced and powerful machine learning algorithm known for its exceptional performance in a wide range of classification tasks. XGBoost stands for "Extreme Gradient Boosting," and it is an optimized implementation of the gradient boosting algorithm.

# In[53]:


from xgboost import XGBClassifier


# In[54]:


xgb = XGBClassifier()


# In[55]:


xgb.fit(x_train, y_train)


# # Prediction

# In[33]:


y_pred = log_reg.predict(x_test)


# In[45]:


y_pred = clf.predict(x_test)


# In[56]:


y_pred = xgb.predict(x_test)


# # Accuracy Score

# Accuracy score is a commonly used metric to evaluate the performance of classification models. It measures the proportion of correctly predicted instances (or observations) out of the total number of instances in the dataset. The accuracy score is calculated using the following formula:
# 
# Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)

# # F1 Score

# The F1 score is a widely used evaluation metric for classification models that takes into account both precision and recall. It provides a single value that balances the trade-off between these two metrics. The F1 score is the harmonic mean of precision and recall, calculated using the following formula:
# 
# F1 score = 2 * (Precision * Recall) / (Precision + Recall)

# # Precision Score

# Precision score is an evaluation metric that measures the proportion of correctly predicted positive instances out of all instances predicted as positive. It quantifies the model's ability to avoid false positives. Precision is calculated using the following formula:
# 
# Precision = (True Positives) / (True Positives + False Positives)

# # Recall Score 

# Recall, also known as sensitivity or true positive rate, is an evaluation metric that measures the proportion of correctly predicted positive instances out of all actual positive instances in the dataset. It quantifies the model's ability to capture true positives and avoid false negatives. Recall is calculated using the following formula:
# 
# Recall = (True Positives) / (True Positives + False Negatives)

# In[34]:


from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score


# Score for Logistic Regression

# In[35]:


accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


# Score for Random Forest Classifier

# In[46]:


accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


# Score for XGBClassifier

# In[57]:


accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


# # Confusion Matrix

# A confusion matrix is a table that provides a comprehensive view of the performance of a classification model by showing the counts or percentages of true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions. It is often used in evaluating the performance of machine learning models, particularly in binary classification problems.

# In[38]:


confusion_matrix = confusion_matrix(y_test, y_pred)


# In[39]:


confusion_matrix


# # Cross Validation Score 

# Cross-validation is a resampling technique used in machine learning to assess the performance and generalization ability of a model. Cross-validation score refers to the evaluation metric or performance measure obtained from cross-validation.
# 
# In cross-validation, the dataset is divided into k subsets or folds. The model is trained on a combination of k-1 folds and evaluated on the remaining fold. This process is repeated k times, with each fold serving as the validation set once. The performance of the model is then averaged across all k folds to obtain the cross-validation score.

# Cross Val Score for Logistic Regression

# In[40]:


cross_val = cross_val_score(
    estimator = log_reg,
    X = x_train,
    y = y_train,
    cv = 10
)


# In[41]:


np.mean(cross_val)


# Cross Val Score for Random Forest Classifier

# In[51]:


cross_val = cross_val_score(
    estimator = clf,
    X = x_train,
    y = y_train,
    cv = 10
)


# In[52]:


np.mean(cross_val)


# Cross Val Score for XGB Classifier

# In[62]:


cross_val = cross_val_score(
    estimator = xgb,
    X = x_train,
    y = y_train,
    cv = 10
)


# In[63]:


np.mean(cross_val)


# # Logistic Regression Result

# In[36]:


results = pd.DataFrame([['Logistic Regression', accuracy, f1, precision, recall]],
                       columns = ['Model', 'Accuracy', 'F1', 'Precision', 'Recall'])


# In[37]:


results


# # Random Forest Classifier Result

# In[47]:


model_results = pd.DataFrame([['Random Forest', accuracy, f1, precision, recall]],
                       columns = ['Model', 'Accuracy', 'F1', 'Precision', 'Recall'])


# In[48]:


result = results.append(model_results, ignore_index = True)


# In[49]:


result


# # XGBoost Classifier Result

# In[58]:


xgb_results = pd.DataFrame([['XGBoost', accuracy, f1, precision, recall]],
                       columns = ['Model', 'Accuracy', 'F1', 'Precision', 'Recall'])


# In[59]:


f_result = result.append(xgb_results, ignore_index = True)


# # Final Result

# In[60]:


f_result


# # Prediction On Single Observation

# In[82]:


dataframe.head()


# In[86]:


single_obs = [[15634602, 619, 42, 2, 0.00, 1, 1, 1, 101348.88, 1]]


# In[87]:


single_obs


# In[88]:


clf.predict(scaler.transform(single_obs))

