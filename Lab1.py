from functions import *
from pandas.plotting import register_matplotlib_converters

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as _stats

# SINGLE VARIABLE ANALYSIS

data = pd.read_csv('covtype.csv', sep=';') # import the dataset into the dataframe, using pandas

# we use a sample with 10% of the data
dataSample = data.sample(frac=0.005)

# some notes regarding DSLabs0
# print(data.head()) # prints/outputs table contents into console. By default, only the first 5 records
# print(data.head(7)) # prints/outputs table contents into console. By default, only the first 5 records
# print(data.tail()) # prints/outputs table contents into console. By default, only the last 5 records
# head and tails methods used mostly to check if loading operations were successful
# print(data.shape) # returns the number of series and records of the dataframe

# print(data.columns) # prints the series of the dataframe
# col = data['Elevation'] # selects all the values from the series 'Elevation', but only prints the first 5 and last 5.
# print(col) # print col into console
# print(data['Elevation']) # the same as assigning col and printing it afterwards
# print(len(col)) # gets (and then prints) the number of elements in the series 'Elevation'

# print(data.values) # get the dataframe into numpy array format, to use at a later time with scikit learn package

# print(data.dtypes) # get the names of all the variables and the corresponding types

# Missing values
# fig = plt.figure(figsize=(10, 7)) # plot figure, specifying the size, but without specifying the figure number
# mv = {}
# for var in data:
#     mv[var] = data[var].isna().sum()
#     bar_chart(plt.gca(), mv.keys(), mv.values(), 'Number of missing values per variable', var, 'nr. missing values')
# fig.tight_layout()
# plt.show() # show the plotted data
#
# print(data.dtypes) # give us the type of the variables

# Variables distribution

# (22) boxplots for each variable on the same graph
# dataSample.boxplot(figsize=(10, 6))
# plt.show()

# (23) singular boxplot for each variable
# columns = dataSample.select_dtypes(include='number').columns
# rows, cols = choose_grid(10) # only the first 10 elements are of interest because the other are binary
# plt.figure()
#
# fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
# i, j = 0, 0
# for n in range(10):
#     axs[i, j].set_title('Boxplot for %s'%columns[n])
#     axs[i, j].boxplot(data[columns[n]].dropna().values)
#     i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
# fig.tight_layout()
# plt.show()

# (24) histogram for each numeric variable; without binary variables
# columns = dataSample.select_dtypes(include='number').columns
# rows, cols = choose_grid(10)
# plt.figure()
# fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*3), squeeze=False)
# i, j = 0, 0
# for n in range(10):
#     axs[i, j].set_title('Histogram for %s'%columns[n])
#     axs[i, j].set_xlabel(columns[n])
#     axs[i, j].set_ylabel("probability") # we have different units for each variable. how can we change that??
#     axs[i, j].hist(data[columns[n]].dropna().values, 'auto')
#     i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
# fig.tight_layout()
# plt.show()

# (25) histograms for categorical variables - bar charts (counting the frequency of each value for each variable)
# it doesn't give anything because we don't have categorical variables, they are all numerical
# columns = dataSample.select_dtypes(include='category').columns
# rows, cols = choose_grid(len(columns))
# plt.figure()
# fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
# i, j = 0, 0
# for n in range(len(columns)):
#     counts = data[columns[n]].dropna().value_counts(normalize=True)
#     bar_chart(axs[i, j], counts.index, counts.values, 'Histogram for %s'%columns[n], columns[n], 'probability')
#     i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
# fig.tight_layout()
# plt.show()

# (26) distribution that best fits the data
# data1 = dataSample.iloc[:, list(range(10))]  # This selects the first 10 columns
columns = dataSample.select_dtypes(include='number').columns
# rows, cols = choose_grid(len(columns))
rows, cols = choose_grid(10)
# plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*3), squeeze=False)
i, j = 0, 0
# for n in range(len(columns)):
for n in range(10):
    axs[i, j].set_title('Trend for %s'%columns[n])
    axs[i, j].set_ylabel("probability")
    sns.distplot(dataSample[columns[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=columns[n])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.show()


# (27)
def compute_known_distributions(x_values, n_bins) -> dict:
    distributions = dict()
    # LogNorm
    # sigma, loc, scale = _stats.lognorm.fit(x_values)
    # distributions['LogNor(%.1f,%.2f)'%(np.log(scale),sigma)] = _stats.lognorm.pdf(x_values, sigma, loc, scale)
    # Gaussian
    mean, sigma = _stats.norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)'%(mean, sigma)] = _stats.norm.pdf(x_values, mean, sigma)
    # SkewNorm
    # a, loc, scale = _stats.skewnorm.fit(x_values)
    # distributions['SkewNorm(%.2f)' % a] = _stats.skewnorm.pdf(x_values, a, loc, scale)
    # Exponential
    # loc, scale = _stats.expon.fit(x_values)
    # distributions['Exp(%.2f)'%(1/scale)] = _stats.expon.pdf(x_values, loc, scale)
    return distributions


def histogram_with_distributions(ax: plt.Axes, series: pd.Series, var: str):
    values = series.sort_values().values
    n, bins, patches = ax.hist(values, 10, density=True, edgecolor='grey')
    distributions = compute_known_distributions(values, bins)
    multiple_line_chart(ax, values, distributions, 'Best fit for %s'%var, var, 'probability')


columns = dataSample.select_dtypes(include='number').columns
rows, cols = choose_grid(len(columns))
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*3), squeeze=False)
i, j = 0, 0
# for n in range(len(columns)):
# for n in range(10):
#     histogram_with_distributions(axs[i, j], data[columns[n]].dropna(), columns[n])
#     i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)

histogram_with_distributions(axs[i, j], data[columns[1]].dropna(), columns[1])
i, j = (i + 1, 0) if (2) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.show()


# Granularity
#(28) histograms for categorical variables - bar charts (counting the frequency of each value for each variable)
# columns = dataSample.select_dtypes(include='number').columns
# rows = 4  # We only want the first 4 columns too keep them well spaced
# cols = 4  # We only want the first 4 rows too keep them well spaced
# plt.figure()
# fig, axs = plt.subplots(rows, cols, figsize=(cols*4.5, rows*2.5), squeeze=True) # 4.5 width and 2.5 length
# bins = range(5, 100, 25)  # From L to R, the number of bins increases by 25 (the granularity increases)
# for i in range(0, 4, 1):
#     for j in range(len(bins)):
#         axs[i, j].set_title('Histogram for %s' % columns[i])
#         axs[i, j].set_xlabel(columns[i])
#         axs[i, j].set_ylabel("occurrences")
#         axs[i, j].hist(dataSample[columns[i]].dropna().values, bins[j])
# fig.tight_layout()
# plt.show()

#MULTI-VARIABLE ANALYSIS

# (1) Sparsity-A dataset is said to be sparse when most of the space defined by its variables is not covered by the instances in the dataset
# register_matplotlib_converters()
# columns = dataSample.select_dtypes(include='number').columns
# rows = 4  # We only want the first 4 columns too keep them well spaced
# cols = 4  # We only want the first 4 rows too keep them well spaced
# plt.figure()
# fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*3), squeeze=True) # 5 width and 3 length
# for i in range(4):
#     var1 = columns[i]
#     for j in range(i+1, 5):
#         var2 = columns[j]
#         axs[i, j-1].set_title("%s x %s" % (var1, var2))
#         axs[i, j-1].set_xlabel(var1)
#         axs[i, j-1].set_ylabel(var2)
#         axs[i, j-1].scatter(dataSample[var1], dataSample[var2])
# fig.tight_layout()
# plt.show()

# (2) Correlation analysis-  In the presence of large dimensionality, a heatmap is easier to analyze Sparsity
# fig = plt.figure(figsize=[12, 12])
# data1 = dataSample.iloc[:, list(range(10))]  # This selects the first 10 columns
# corr_mtx = data1.corr()
# sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
# plt.title('Correlation analysis')
# plt.show()