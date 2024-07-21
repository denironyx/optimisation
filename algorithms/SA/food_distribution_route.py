import numpy as np
import pandas as pd
import random
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt

# Generate random coordinates for farms, storage warehouses, and supermarkets
num_farms = 5
num_warehouses = 2
num_supermarkets = 5

# Define ranges for coordinates
lat_range = (40.5, 40.9)
lon_range = (-74.2, -73.7)

# Generate coordinates
farm_coords = np.random.uniform(low=[lat_range[0], lon_range[0]], high=[lat_range[1], lon_range[1]], size=(num_farms, 2))
warehouse_coords = np.random.uniform(low=[lat_range[0], lon_range[0]], high=[lat_range[1], lon_range[1]], size=(num_warehouses, 2))
supermarket_coords = np.random.uniform(low=[lat_range[0],lon_range[0]], high=[lat_range[1], lon_range[1]], size=(num_supermarkets, 2))

# Combine coordinates into a DataFrame
locations = pd.DataFrame(np.vstack([farm_coords, warehouse_coords, supermarket_coords]),
                         columns=['Latitude', 'Longitude'],
                         index=[f'Farm{i}' for i in range(num_farms)] +
                               [f'Warehouse{i}' for i in range(num_warehouses)] +
                               [f'Supermarket{i}' for i in range(num_supermarkets)])

# Compute the distance matrix
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of Earth in Kilometers
    return c * r

distance_matrix = pd.DataFrame(index = locations.index, columns=locations.index)
for i in locations.index:
    for j in locations.index:
        if i != j:
            distance_matrix.loc[i, j] = haversine(locations.loc[i, 'Latitude'], locations.loc[i, 'Longitude'],
                                                  locations.loc[j, 'Latitude'], locations.loc[i, 'Longitude'])

## visualize the locations
plt.scatter(locations['Longitude'], locations['Latitude'], c = 'green')
for i, point in enumerate(locations.index):
    plt.annotate(point, (locations.loc[point, 'Longitude'], locations.loc[point, 'Latitude']))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Delivery points')
plt.show()

