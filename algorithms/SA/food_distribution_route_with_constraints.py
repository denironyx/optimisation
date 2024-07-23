import numpy as np
import pandas as pd
import random
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt

"""
This is a travel sales man problem with no constriants as regards direction. 
Next stage would be to include constraints suchs as a driver needs to visit a farm after making a trip for 
6hrs it stops at the closest warehouse and another driver takes the goods to the supermarket. 

The only time a driver is expected to take the goods from farm to super market is if the journey is about 6hrs. 
If it's morethan that, it has to drop it at the closest warehouse. 

"""

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

## Define cost and time functions
# Parameters
vehicle_capacity = 1000 # kg
cost_per_km = 2.0 # cost per km
average_speed = 50.0 # km/h

# Define demand (example values)
demand = {f'Farm{i}': 200 + i * 50 for i in range(num_farms)}
demand.update({f'Supermarket{i}': 150 + i * 30 for i in range(num_supermarkets)})
demand.update({f'Warehouse{i}': 0 for i in range(num_warehouses)})

def calculate_total_distance(route, distance_matrix):
    distance = 0
    for i in range(len(route) - 1):
        distance += distance_matrix.loc[route[i], route[i + 1]]
    distance += distance_matrix.loc[route[-1], route[0]]  # Return to the start
    return distance

def calculate_cost(route, distance_matrix, cost_per_km):
    total_distance = calculate_total_distance(route, distance_matrix)
    return total_distance * cost_per_km

def calculate_time(route, distance_matrix, average_speed):
    total_distance = calculate_total_distance(route, distance_matrix)
    return total_distance / average_speed


## Implement simulated annealing

def initial_route(locations):
    route = locations[:]
    random.shuffle(route)
    return route


def acceptance_probability(current_cost, neighbor_cost, temperature):
    if neighbor_cost < current_cost:
        return 1.0
    else:
        return np.exp((current_cost - neighbor_cost) / temperature)


def generate_neighbor(route):
    neighbor = route[:]
    i, j = random.sample(range(len(route)), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def simulated_annealing(locations, distance_matrix, initial_temp, cooling_rate):
    current_solution = initial_route(locations)
    best_solution = current_solution
    current_temp = initial_temp

    while current_temp > 1:
        neighbor_solution = generate_neighbor(current_solution)
        current_cost = calculate_cost(current_solution, distance_matrix, cost_per_km)
        neighbor_cost = calculate_cost(neighbor_solution, distance_matrix, cost_per_km)

        if acceptance_probability(current_cost, neighbor_cost, current_temp) > random.random():
            current_solution = neighbor_solution

        if calculate_total_distance(current_solution, distance_matrix) < calculate_total_distance(best_solution,
                                                                                                  distance_matrix):
            best_solution = current_solution

        current_temp *= cooling_rate

    return best_solution


## Run the Optimization and Calculate cost
# Simulated Annealing Parameters
initial_temp = 10000
cooling_rate = 0.995

# Prepare the initial route
initial_solution = list(locations.index)

# Run Simulated Annealing
optimized_route_sa = simulated_annealing(initial_solution, distance_matrix, initial_temp, cooling_rate)

# Calculate metrics
optimized_distance_sa = calculate_total_distance(optimized_route_sa, distance_matrix)
optimized_cost_sa = calculate_cost(optimized_route_sa, distance_matrix, cost_per_km)
optimized_time_sa = calculate_time(optimized_route_sa, distance_matrix, average_speed)

print("\nSimulated Annealing Results:")
print("Optimized Route:", optimized_route_sa)
print("Optimized Distance (km):", optimized_distance_sa)
print("Optimized Cost ($):", optimized_cost_sa)
print("Optimized Time (hours):", optimized_time_sa)
