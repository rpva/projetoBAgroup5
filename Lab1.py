from functions import *

import pandas as pd
import matplotlib.pyplot as plt

# SINGLE VARIABLE ANALYSIS

# data = pd.read_csv('covtype.csv', index_col='Cover_Type(Class)', sep=';')  # import the dataset into the dataframe, using pandas

data = pd.read_csv('covtype.csv', sep=';')
dataSample = data.sample(frac=0.1) # we use a sample with 10% of the data

print(data.head()) # prints/outputs table contents into console. By default, only the first 5 records
print(data.head(7)) # prints/outputs table contents into console. By default, only the first 5 records
print(data.tail()) # prints/outputs table contents into console. By default, only the last 5 records
# head and tails used mostly to check if loading operations were successful
print(data.shape) # returns the number of series and records of the dataframe

print(data.columns) # prints the series of the dataframe
col = data['Elevation'] # selects all the values from the series 'Elevation', but only prints the first 5 and last 5.
print(col) # print col into console
print(data['Elevation']) # the same as assigning col and printing it afterwards
print(len(col)) # gets (and then prints) the number of elements in the series 'Elevation'

print(data.values) # get the dataframe into numpy array format, to use at a later time with scikit learn package

fig = plt.figure(figsize=(10, 7)) # plot figure, specifying the size, but without specifying the number
mv = {}

# Missing values
for var in data:
    mv[var] = data[var].isna().sum()
    bar_chart(plt.gca(), mv.keys(), mv.values(), 'Number of missing values per variable', var, 'nr. missing values')
fig.tight_layout()
plt.show() # show the plotted data

print(data.dtypes) # give us the type of the variables

# Variables distribution

# boxplots for each variable on the same graph

dataSample.boxplot(figsize=(10, 6))
plt.show()

# singular boxplot for each variable


# Granularity

# MULTI-VARIABLE ANALYSIS
# Sparsity
# Correlation analysis



