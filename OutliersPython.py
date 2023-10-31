#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\0404o\Downloads\traffic.csv")
df.info()


# In[2]:


df.head()


# # How do we determine an outlier?

# There are several ways that we can find outliers using statistical techniques, domain knowledge, algorithms.

# In[3]:


px.line(data_frame=df, x="Date", y="Sessions", title="Lets view our outliers with the naked eye")


# In[4]:


px.box(data_frame=df, x="Sessions")


# In[5]:


px.violin(data_frame=df, x="Sessions")


# # Using IQR to estimate outliers

# In[6]:


def traditional_outlier(df, x):
    q1 = df[x].quantile(.25)
    q3 = df[x].quantile(.75)
    iqr = q3 - q1
    df['Traditional'] = np.where(df[[x]]< (q1-1.5*iqr), -1,
                        np.where(df[[x]]> (q3+1.5*iqr), -1,1))
    return df


# In[7]:


traditional_outlier(df, "Sessions")


# # Using algoritms to estimate outliers

# #### Isolation Forest

# Isolation Forest using a tree based system to detect anomalys in the data. This is really good for large datasets.

# In[8]:


from sklearn.ensemble import IsolationForest


# In[9]:


IsolationForest().fit(df[["Sessions"]]).predict(df[["Sessions"]])


# #### Eliptic Envelope

# An algoithm for detecting outliers in a Guassian distributed dataset

# In[10]:


from sklearn.covariance import EllipticEnvelope
EllipticEnvelope().fit(df[["Sessions"]]).predict(df[["Sessions"]])


# #### Local Outlier Factor

# It measures the localdeviation of the density of a given sample with respect to its neighbors.

# In[11]:


from sklearn.neighbors import LocalOutlierFactor
LocalOutlierFactor(n_neighbors=5, novelty=True).fit(df[["Sessions"]]).predict(df[["Sessions"]])


# In[12]:


def outliers_find(df, x):
    df["Local Outlier"] = LocalOutlierFactor(n_neighbors=5, novelty=True).fit(df[[x]]).predict(df[[x]])
    df["Isolation Forest"] = IsolationForest().fit(df[[x]]).predict(df[[x]])
    df["Elliptical"] = EllipticEnvelope().fit(df[[x]]).predict(df[[x]])
    return df


# In[13]:


outliers_find(df, 'Sessions')


# In[14]:


def outliers_find(df, x):
    q1 = df[x].quantile(.25)
    q3 = df[x].quantile(.75)
    iqr = q3 - q1
    df['Traditional'] = np.where(df[[x]]< (q1-1.5*iqr), -1,
                        np.where(df[[x]]> (q3+1.5*iqr), -1,1))
    df["Local Outlier"] = LocalOutlierFactor(n_neighbors=5, novelty=True).fit(df[[x]]).predict(df[[x]])
    df["Isolation Forest"] = IsolationForest().fit(df[[x]]).predict(df[[x]])
    df["Elliptical"] = EllipticEnvelope().fit(df[[x]]).predict(df[[x]])
    return df

