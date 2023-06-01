#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

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
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Reading Dataset

# In[3]:


dataframe = pd.read_csv('Water Quality Testing.csv')


# # Exploring Dataset

# The process of analyzing and understanding a dataset to gain insights and identify patterns or trends. The goal of exploring the data is to become familiar with its structure, distribution, and quality, as well as to identify potential issues or anomalies that may need to be addressed before further analysis.

# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe.info()


# In[7]:


dataframe.isna().sum()


# In[8]:


dataframe.shape


# # Statical Info

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


plt.figure(figsize = (12 ,5))
sns.heatmap(corr_matrix,
            annot = True, 
            cmap = 'coolwarm')


# In[13]:


dataset = dataframe.copy()


# In[14]:


dataset


# # Feature Scaling 

# Feature scaling is a preprocessing technique used in machine learning to normalize or standardize the range of independent variables (features) in a dataset. This is often done to improve the performance of machine learning models and make the optimization process more efficient.

# Standard scaling, also known as feature scaling or standardization, is a common data preprocessing technique used in machine learning and data analysis. It involves transforming numerical features in a dataset to have a mean of zero and a standard deviation of one

# Mathematically, the standard scaling formula for a feature x is:
# 
# x_scaled = (x - μ) / σ

# In[15]:


from sklearn.preprocessing import StandardScaler


# In[16]:


scaler = StandardScaler()


# In[17]:


dataset = scaler.fit_transform(dataset)


# In[18]:


dataset


# # Clustering The Data

# The goal of clustering is to partition the data into distinct groups or clusters, where objects within each cluster are similar to each other and dissimilar to objects in other clusters. There are several different types of clustering algorithm. We are going to use :

# K-means clustering:
# 
# This algorithm partitions the data into k clusters, where k is a user-specified parameter. The algorithm iteratively assigns each data point to the closest centroid (mean) of a cluster, and then updates the centroid based on the mean of the data points in the cluster.

# In[20]:


from sklearn.cluster import KMeans


# In[21]:


wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(dataset)
    wcss.append(kmeans.inertia_)


# In[22]:


plt.plot(range(1, 20), wcss, 'bx-')
plt.title('The Elbo Methode')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.show()


# In[23]:


kmeans = KMeans(n_clusters = 8, init = 'k-means++', random_state = 0)


# In[24]:


y_kmeans = kmeans.fit_predict(dataset)


# In[ ]:





# In[25]:


y_kmeans


# In[26]:


y_kmeans.shape


# In[28]:


y_kmeans = y_kmeans.reshape(len(y_kmeans), 1)


# In[30]:


y_kmeans.shape


# In[31]:


ax = np.concatenate((y_kmeans, dataframe), axis = 1)


# In[32]:


ax


# In[33]:


dataframe.columns


# # Clustered Dataset

# In[34]:


dataframe_cluster = pd.DataFrame(data = ax, columns = ['Cluster Number', 'Sample ID', 'pH', 'Temperature (°C)', 'Turbidity (NTU)',
       'Dissolved Oxygen (mg/L)', 'Conductivity (µS/cm)'])


# In[37]:


dataframe_cluster


# In[40]:


dataframe_cluster.head()


# In[41]:


dataframe_cluster.to_csv('Cluster Dataset')

