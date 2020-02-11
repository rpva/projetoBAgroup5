from functions import *

import pandas as pd
import matplotlib.pyplot as plt

# SINGLE VARIABLE ANALYSIS

# data = pd.read_csv('covtype.csv', index_col='Cover_Type(Class)', sep=';')  # import the dataset into the dataframe, using pandas

data = pd.read_csv('covtype.csv', sep=';')
# we use a sample with 10% of the data
dataSample = data.sample(frac=0.1)

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

# (22) boxplots for each variable on the same graph

dataSample.boxplot(figsize=(10, 6))
plt.show()

# (23) singular boxplot for each variable

columns = dataSample.select_dtypes(include='number').columns
rows, cols = choose_grid(10) # only the first 10 elements are of interest because the other are binary
plt.figure()

fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
i, j = 0, 0
for n in range(10):
    axs[i, j].set_title('Boxplot for %s'%columns[n])
    axs[i, j].boxplot(data[columns[n]].dropna().values)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.show()

# (24) histogram for each numeric variable; without binary variables

columns = dataSample.select_dtypes(include='number').columns
rows, cols = choose_grid(10)
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*3), squeeze=False)
i, j = 0, 0
for n in range(10):
    axs[i, j].set_title('Histogram for %s'%columns[n])
    axs[i, j].set_xlabel(columns[n])
    axs[i, j].set_ylabel("probability") # we have different units for each variable. how can we change that??
    axs[i, j].hist(data[columns[n]].dropna().values, 'auto')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.show()

# (25) histograms for categorical variables - bar charts (counting the frequency of each value for each variable)
# it doesn't give anything because we don't have categorical variables, they are all numerical

columns = dataSample.select_dtypes(include='category').columns
rows, cols = choose_grid(len(columns))
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
i, j = 0, 0
for n in range(len(columns)):
    counts = data[columns[n]].dropna().value_counts(normalize=True)
    bar_chart(axs[i, j], counts.index, counts.values, 'Histogram for %s'%columns[n], columns[n], 'probability')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.show()

# Granularity

# MULTI-VARIABLE ANALYSIS
# Sparsity
# Correlation analysis



