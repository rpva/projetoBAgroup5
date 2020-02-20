import pandas as pd

data = pd.read_csv('covtype.csv', sep=';') # import the dataset into the dataframe, using pandas

# some notes regarding DSLabs0
print(data.head()) # prints/outputs table contents into console. By default, only the first 5 records
print(data.head(7)) # prints/outputs table contents into console. By default, only the first 5 records
print(data.tail()) # prints/outputs table contents into console. By default, only the last 5 records
# head and tails methods used mostly to check if loading operations were successful
print(data.shape) # returns the number of series and records of the dataframe

print(data.columns) # prints the series of the dataframe
col = data['Elevation'] # selects all the values from the series 'Elevation', but only prints the first 5 and last 5.
print(col) # print col into console
print(data['Elevation']) # the same as assigning col and printing it afterwards
print(len(col)) # gets (and then prints) the number of elements in the series 'Elevation'

print(data.values) # get the dataframe into numpy array format, to use at a later time with scikit learn package

print(data.dtypes) # get the names of all the variables and the corresponding types