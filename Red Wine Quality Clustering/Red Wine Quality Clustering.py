#!/usr/bin/env python
# coding: utf-8

# # Red Wine Clustering

# The goal of the Red Wine Clustering project is to analyze a dataset of red wine samples and identify distinct groups or clusters based on their chemical properties. The project aims to uncover patterns and relationships within the data that can provide insights into the characteristics and qualities of different types of red wines.

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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Importing Dataset

# In[3]:


dataframe = pd.read_csv('winequality-red.csv')


# # Exploratory Data Analysis

# The process of analyzing and understanding a dataset to gain insights and identify patterns or trends. The goal of exploring the data is to become familiar with its structure, distribution, and quality, as well as to identify potential issues or anomalies that may need to be addressed before further analysis.

# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe.shape


# In[8]:


dataframe.isna().sum()


# In[11]:


dataframe.info()


# # Statical Info

# Statistical information provides a summary and description of data using various statistical measures. It helps in understanding the distribution, central tendency, variability, and relationships within a dataset.

# In[9]:


dataframe.describe()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[12]:


corr_matrix = dataframe.corr()


# In[13]:


corr_matrix


# In[15]:


plt.figure(figsize = (15, 7))
sns.heatmap(corr_matrix, 
            annot = True, 
            cmap = 'inferno')


# # Correlation of Wine Quality with Different Attriutes 

# In[16]:


dataframe.columns


# In[17]:


dataset = dataframe.drop('quality', axis = 1)


# In[18]:


dataset.head()


# In[20]:


dataset.corrwith(dataframe['quality']).plot.bar(
    figsize = (12, 5), 
    title = 'Correlation with Quality', 
    rot = 90, 
    cmap = 'ocean'
)


# In[21]:


dataframe.head()


# # Scaling

# Scaling refers to the process of transforming numerical variables in a dataset to a specific range or distribution. It is often performed as a preprocessing step before applying machine learning algorithms or statistical analysis. 

# Standard scaling, also known as feature scaling or standardization, is a common data preprocessing technique used in machine learning and data analysis. It involves transforming numerical features in a dataset to have a mean of zero and a standard deviation of one

# Mathematically, the standard scaling formula for a feature x is:
# 
# x_scaled = (x - μ) / σ

# In[22]:


from sklearn.preprocessing import StandardScaler


# In[23]:


scaler = StandardScaler()


# In[24]:


wine_data = dataframe.copy()


# In[26]:


wine_data.head()


# In[28]:


wine_data = scaler.fit_transform(wine_data)


# In[30]:


wine_data


# # KMeans

# K-means is a popular clustering algorithm used to partition data points into K distinct clusters based on their features. The algorithm aims to minimize the within-cluster sum of squares (WCSS) by iteratively assigning data points to clusters and updating the cluster centroids.

# In[31]:


from sklearn.cluster import KMeans


# In[32]:


wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(wine_data)
    wcss.append(kmeans.inertia_)


# # Clustering Dataset

# Clustering is a data analysis technique that involves grouping similar data points together based on their inherent characteristics or patterns. The goal of clustering is to discover natural groupings within the data without any prior knowledge of the group labels. It is an unsupervised learning technique as it does not rely on labeled data.

# In[33]:


plt.plot(range(1, 20), wcss, 'bx-')
plt.title('The Elbow Methode')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.show()


# In[34]:


kmeans = KMeans(n_clusters = 8, init = 'k-means++', random_state = 0)


# In[35]:


y_kmeans = kmeans.fit_predict(dataset)


# In[36]:


y_kmeans = y_kmeans.reshape(len(y_kmeans), 1)


# In[37]:


ab = np.concatenate((y_kmeans, dataframe), axis = 1)


# In[39]:


dataframe.columns


# # Clustered Dataframe

# In[41]:


cluster_dataframe = pd.DataFrame(data = ab, columns =  ['Cluster Number', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality'])


# In[42]:


cluster_dataframe.head()


# In[43]:


cluster_dataframe.to_csv('Cluster_Red_Wine')


# # Thanks !
