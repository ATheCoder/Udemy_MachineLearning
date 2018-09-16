# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:42:20 2018

@author: A
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
