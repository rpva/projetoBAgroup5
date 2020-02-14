from functions import *

import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters  # new import LAB2
from sklearn.preprocessing import Normalizer  # new import LAB2
from sklearn.preprocessing import OneHotEncoder  # new import LAB2
from imblearn.over_sampling import SMOTE, RandomOverSampler  # new import LAB2

# -------------------- LAB3 - CLASSIFICATION

data = pd.read_csv('covtype.csv', sep=';')
# we use a sample with 10% of the data
dataSample = data.sample(frac=0.1)

# TRAINING MODELS
# identify the target or class which is the variable to predict (Cover_Type)
# the type of the target variable determines the kind of operation to perform
# targets with just a few values allow for a classification task
# real-valued targets require a prediction one

# Training Strategy


# Estimators and Models
