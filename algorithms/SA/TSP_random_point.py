import numpy as np
import pandas as pd
import random
from math import radians, cos, sin, asin, sqrt

# generate random coordinates within a specific geographic area

num_points = 10
latitudes = np.random.uniform(40.5, 40.9, num_points) # NYC latitude range
longitudes = np.random.uniform(74.2, -73.7, num_points) # NYC longitude range

# combine latitudes and longitudes into a dataframe
locations = pd.DataFrame({
    'Latitude': latitudes,
    'Longitude': longitudes
}, index=[f'Point{i}' for i in range(num_points)])


# Function to calculate Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r


# Compute the distance matrix
distance_matrix = pd.DataFrame(np.zeros((num_points, num_points)), index=locations.index, columns=locations.index)

for i in locations.index:
    for j in locations.index:
        if i != j:
            distance_matrix.loc[i, j] = haversine(locations.loc[i, 'Latitude'], locations.loc[i, 'Longitude'],
                                                  locations.loc[j, 'Latitude'], locations.loc[j, 'Longitude'])

# Create an initial solution
initial_solution = list(locations.index)
random.shuffle(initial_solution)


# Compute the total travel distance for the initial solution
def calculate_total_distance(route, distance_matrix):
    distance = 0
    for i in range(len(route) - 1):
        distance += distance_matrix.loc[route[i], route[i + 1]]
    distance += distance_matrix.loc[route[-1], route[0]]  # Return to the starting point
    return distance


# Calculate the length of travel for the initial solution
length_of_travel = calculate_total_distance(initial_solution, distance_matrix)

print("Initial Solution:", initial_solution)
print("Length of Travel:", length_of_travel)

# Visualize the locations
plt.scatter(locations['Longitude'], locations['Latitude'], c='red')
for i, point in enumerate(locations.index):
    plt.annotate(point, (locations.loc[point, 'Longitude'], locations.loc[point, 'Latitude']))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Random Delivery Points in NYC')
plt.show()

