#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


# # One-Dimensional Convolution on time series
# 
# In this code, we'll see what convolution looks like on a time-series.
# 
# The time series we'll use is from [Google Trends](https://trends.google.com/trends/). It measures the popularity of the search term "machine learning" for weeks from January 25, 2015 to January 15, 2020.

# In[5]:




# Load the time series as a Pandas dataframe
machinelearning = pd.read_csv(
    './machinelearning.csv',
    parse_dates=['Week'],
    index_col='Week',
)

machinelearning.plot();


# A time-series is one-dimensional, so what should the kernel be? A 1D array! Here are some kernels sometimes used on time-series data:

# In[6]:


detrend = tf.constant([-1, 1], dtype=tf.float32)

average = tf.constant([0.2, 0.2, 0.2, 0.2, 0.2], dtype=tf.float32)

spencer = tf.constant([-3, -6, -5, 3, 21, 46, 67, 74, 67, 46, 32, 3, -5, -6, -3], dtype=tf.float32) / 320


# Convolution on a sequence works just like convolution on an image. The difference is just that a sliding window on a sequence only has one direction to travel -- left to right -- instead of the two directions on an image. And just like before, the features picked out depend on the pattern on numbers in the kernel.
# 
# choose one of the kernels below and run the cell to see!

# In[7]:


# UNCOMMENT ONE
import numpy as np
kernel = detrend
# kernel = average
# kernel = spencer

# Reformat for TensorFlow
ts_data = machinelearning.to_numpy()
ts_data = tf.expand_dims(ts_data, axis=0)
ts_data = tf.cast(ts_data, dtype=tf.float32)
kern = tf.reshape(kernel, shape=(*kernel.shape, 1, 1))


# In[13]:


kernel = detrend
# kernel = average
# kernel = spencer

# Reformat for TensorFlow
ts_data = machinelearning.to_numpy()
ts_data = tf.expand_dims(ts_data, axis=0)
ts_data = tf.cast(ts_data, dtype=tf.float32)
kern = tf.reshape(kernel, shape=(*kernel.shape, 1, 1))

ts_filter = tf.nn.conv1d(
    input=ts_data,
    filters=kern,
    stride=1,
    padding='VALID',
)

# Format as Pandas Series
machinelearning_filtered = pd.Series(tf.squeeze(ts_filter).numpy())

machinelearning_filtered.plot();


# In fact, the `detrend` kernel filters for *changes* in the series, while `average` and `spencer` are both "smoothers" that filter for low-frequency components in the series.
# 
# If you were interested in predicting the future popularity of search terms, you might train a convnet on time-series like this one. It would try to learn what features in those series are most informative for the prediction.
# 
# Though convnets are not often the best choice on their own for these kinds of problems, they are often incorporated into other models for their feature extraction capabilities.

# In[ ]:





# In[ ]:




