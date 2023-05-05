#!/usr/bin/env python
# coding: utf-8

# # Importing The Libraries

# These are just a few examples of popular Python libraries. You can import any other library using the same import statement followed by the library name or alias:

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


# # Reading The Dataset

# In[2]:


#reading the dataset from the requred path or file

dataset = pd.read_csv('CC GENERAL.csv')


# # Exploring The Dataset

# The process of analyzing and understanding a dataset to gain insights and identify patterns or trends. The goal of exploring the data is to become familiar with its structure, distribution, and quality, as well as to identify potential issues or anomalies that may need to be addressed before further analysis.

# In[3]:


#loading the head of dataset

dataset.head()


# In[4]:


#loading the tail of dataset

dataset.tail()


# In[6]:


#checking the datset

dataset.shape


# In[10]:


#checking the data columns

dataset.columns


# In[11]:


#information about overall dataset

dataset.info()


# In[12]:


#checking the object/categorical values present in the dataset

dataset.select_dtypes(include = 'object').columns


# In[16]:


#viewing the numerical values present in the dataset

dataset.select_dtypes(include = ['int64', 'float64']).columns


# In[17]:


#statical information about the dataset

dataset.describe()


# # Dealing With Missing Value

# There are several approaches for filling in missing values in a dataset:

# Mean, Median or Mode Imputation:
# 
#                                 In this approach, missing values are replaced with the mean, median, or mode value of the corresponding feature. This is a simple approach that can work well if the missing values are randomly distributed and the feature has a normal distribution.

# In[18]:


dataset.isnull().values.any()


# In[19]:


dataset.isnull().values.sum()


# In[21]:


dataset.columns[dataset.isnull().any()]


# In[22]:


#filling the missing value 

dataset['CREDIT_LIMIT'] = dataset['CREDIT_LIMIT'].fillna(dataset['CREDIT_LIMIT'].mean())
dataset['MINIMUM_PAYMENTS'] = dataset['MINIMUM_PAYMENTS'].fillna(dataset['MINIMUM_PAYMENTS'].mean())


# In[23]:


len(dataset.columns[dataset.isnull().any()])


# # Encoding The Categorical Data

#  Categorical data refers to data that represents categories or groups, such as gender, color, or location. Machine learning algorithms typically require numerical data as input, so categorical data must be converted to numerical data before it can be used.

# In[24]:


#categorical columns

dataset.select_dtypes(include = 'object')


# In[25]:


dataset = dataset.drop(columns = 'CUST_ID')


# In[27]:


dataset.head()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[28]:


corr_matrix = dataset.corr()


# In[29]:


plt.figure(figsize = (20, 10))
ax = sns.heatmap(corr_matrix,
                 annot = True, 
                 cmap = 'coolwarm')


# # Feature Scaling 

# Feature scaling is a preprocessing technique used in machine learning to normalize or standardize the range of independent variables (features) in a dataset. This is often done to improve the performance of machine learning models and make the optimization process more efficient.

# In[30]:


#making the copy of original dataset

dataframe = dataset


# In[31]:


#performing feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset = sc.fit_transform(dataset)


# In[33]:


dataset


# # Clustering The Data

# The goal of clustering is to partition the data into distinct groups or clusters, where objects within each cluster are similar to each other and dissimilar to objects in other clusters. There are several different types of clustering algorithm. We are going to use : 

# K-means clustering:
# 
# This algorithm partitions the data into k clusters, where k is a user-specified parameter. The algorithm iteratively assigns each data point to the closest centroid (mean) of a cluster, and then updates the centroid based on the mean of the data points in the cluster

# In[35]:


from sklearn.cluster import KMeans


# In[38]:


wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(dataset)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 20), wcss, 'bx-')
plt.title('The Elbo Methode')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.show()


# # Building The Model 

# Building a model refers to the process of training a machine learning algorithm on a dataset to create a predictive model that can make accurate predictions on new, unseen data. The goal of building a model is to create a system that can automatically learn patterns and relationships in the data, and use this knowledge to make predictions or decisions.

# In[45]:


kmeans = KMeans(n_clusters = 8, init = 'k-means++', random_state = 0)


# In[46]:


#dependent variable

y_kmeans = kmeans.fit_predict(dataset)


# In[47]:


y_kmeans


# # Gettings The Output

# In[49]:


y_kmeans.shape


# In[50]:


y_kmeans = y_kmeans.reshape(len(y_kmeans), 1)


# In[51]:


y_kmeans


# In[52]:


y_kmeans.shape


# In[55]:


bx = np.concatenate((y_kmeans, dataframe), axis = 1)


# In[56]:


dataframe.columns


# In[58]:


dataframe_final = pd.DataFrame(data = bx, columns = ['Cluster_numbersa','BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
       'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
       'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
       'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
       'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT',
       'TENURE'])


# In[60]:


dataframe_final.head()


# In[61]:


dataframe_final.to_csv('Segmented_Customers')


# In[ ]:




