import pandas as pd
from functions import *
data = pd.read_csv('covtype.csv')

print(data.shape)

col_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
             'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
             'Wilderness_Area (Rawah Wilderness Area)', 'Wilderness_Area (Neota Wilderness Area)', 'Wilderness_Area_(Comanche Peak Wilderness Area)',
             'Wilderness_Area (Cache la Poudre Wilderness Area)', 'Soil_Type (1)', 'soil_type (2)', 'soil_type (3)', 'soil_type (4)',
             'soil_type (5)', 'soil_type (6)', 'soil_type (7)', 'soil_type (8)', 'soil_type (9)', 'soil_type (10)', 'soil_type (11)', 'soil_type (12)',
             'soil_type (13)','soil_type (14)','soil_type (15)','soil_type (16)','soil_type (17)','soil_type (18)','soil_type (19)','soil_type (20)',
             'soil_type (21)','soil_type (22)','soil_type (23)','soil_type (24)','soil_type (25)','soil_type (26)','soil_type (27)','soil_type (28)',
             'soil_type (29)','soil_type (30)','soil_type (31)','soil_type (32)','soil_type (33)','soil_type (34)','soil_type (35)','soil_type (36)',
             'soil_type (37)','soil_type (38)', 'soil_type (39)', 'soil_type (40)', 'Cover_Type (7 types)']

data = pd.read_csv('covtype.cvs', names=col_names)
