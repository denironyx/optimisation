import numpy as np
import pandas as pd
import random
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt

"""
This is a travel sales man problem with defined constraints. 

Define the Constraints:
A driver can only drive for a maximum of 6 hours before stopping at a warehouse.
If a trip from a farm to a supermarket exceeds 6 hours, the driver must stop at the nearest warehouse, and another driver will take over.

Modify the Simulated Annealing Algorithm:
Update the algorithm to include a check for the total driving time from the farm to the supermarket.
Include stops at the nearest warehouse if the total driving time exceeds 6 hours.

Plot the Route:
Visualize the route with stops at warehouses and changes in drivers.

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

## Compute the distance matrix

"""
This haversine function calculates the great-circle distance between two points on the Earth
specified by their latitude and longitude using the Haversine formula
"""
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
max_drive_time = 6 # hours

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

def find_nearest_warehouse(current_location, warehouse_coords):
    min_distance = float('inf')
    nearest_warehouse = None
    for warehouse in warehouse_coords.index:
        distance = distance_matrix.loc[current_location, warehouse]
        if distance < min_distance:
            min_distance = distance
            nearest_warehouse = warehouse
    return nearest_warehouse

def simulated_annealing_with_constraints(locations, distance_matrix, initial_temp, cooling_rate, max_drive_time, average_speed):
    current_solution = initial_route(locations)
    best_solution = current_solution
    current_temp = initial_temp

    # store the details of each segment
    trip_details = []

    while current_temp > 1:
        neighbor_solution = generate_neighbor(current_solution)

        # Apply constraints
        total_time = 0
        route_with_stops = []
        for i in range(len(neighbor_solution) - 1):
            start = neighbor_solution[i]
            end = neighbor_solution[i + 1]
            travel_time = distance_matrix.loc[start, end] / average_speed
            if total_time + travel_time > max_drive_time:
                nearest_warehouse = find_nearest_warehouse(start, warehouse_coords)
                route_with_stops.append(nearest_warehouse)
                trip_details.append({
                    'Start': start,
                    'End': nearest_warehouse,
                    'Distance': distance_matrix.loc[start, nearest_warehouse],
                    'Cost': distance_matrix.loc[start, nearest_warehouse] * cost_per_km,
                    'Time': distance_matrix.loc[start, nearest_warehouse] / average_speed
                })
                total_time = travel_time
            else:
                total_time += travel_time
            route_with_stops.append(start)
            trip_details.append({
                'Start': start,
                'End': end,
                'Distance': distance_matrix.loc[start, end],
                'Cost': distance_matrix.loc[start, end] * cost_per_km,
                'Time': distance_matrix.loc[start, end] / average_speed
            })
        route_with_stops.append(neighbor_solution[-1])

        current_cost = calculate_cost(current_solution, distance_matrix, cost_per_km)
        neighbor_cost = calculate_cost(route_with_stops, distance_matrix, cost_per_km)

        if acceptance_probability(current_cost, neighbor_cost, current_temp) > random.random():
            current_solution = route_with_stops

        if calculate_total_distance(current_solution, distance_matrix) < calculate_total_distance(best_solution, distance_matrix):
            best_solution = current_solution

        current_temp *= cooling_rate

    return best_solution, trip_details


## Run the Optimization and Calculate cost
# Simulated Annealing Parameters
initial_temp = 10000
cooling_rate = 0.995

# Prepare the initial route
initial_solution = list(locations.index)

# Run Simulated Annealing
optimized_route_sa, trip_details_sa = simulated_annealing_with_constraints(initial_solution, distance_matrix, initial_temp, cooling_rate, max_drive_time, average_speed)

# Convert trip details to Dataframe
trip_details_df_sa = pd.DataFrame(trip_details_sa)

# Calculate metrics
optimized_distance_sa = calculate_total_distance(optimized_route_sa, distance_matrix)
optimized_cost_sa = calculate_cost(optimized_route_sa, distance_matrix, cost_per_km)
optimized_time_sa = calculate_time(optimized_route_sa, distance_matrix, average_speed)

print("\nSimulated Annealing Results with Constraints:")
print("Optimized Route:", optimized_route_sa)
print("Optimized Distance (km):", optimized_distance_sa)
print("Optimized Cost ($):", optimized_cost_sa)
print("Optimized Time (hours):", optimized_time_sa)

# Display trip lifecycle details
print("\nTrip Lifecycle Details:")
print(trip_details_df_sa)

### Visualization

# List of warehouse labels
warehouse_labels = [f'Warehouse{i}' for i in range(num_warehouses)]


def plot_route_with_stops(locations, route, warehouse_labels):
    # Extract coordinates for plotting
    coords = locations.loc[route]
    latitudes = coords['Latitude'].values
    longitudes = coords['Longitude'].values

    # Plot the delivery points
    plt.figure(figsize=(12, 8))
    plt.scatter(longitudes, latitudes, color='red', marker='o')

    # Annotate the delivery points
    for i, point in enumerate(route):
        plt.annotate(point, (longitudes[i], latitudes[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # Plot the route with stops
    for i in range(len(route) - 1):
        start = route[i]
        end = route[i + 1]
        start_coords = locations.loc[start]
        end_coords = locations.loc[end]
        if start in warehouse_labels or end in warehouse_labels:
            line_style = 'dotted'
            color = 'green'
        else:
            line_style = 'solid'
            color = 'blue'
        plt.plot([start_coords['Longitude'], end_coords['Longitude']],
                 [start_coords['Latitude'], end_coords['Latitude']],
                 color=color, linestyle=line_style, linewidth=2)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Optimized Delivery Route with Stops')
    plt.grid(True)
    plt.show()


# Plot the optimized route with stops
plot_route_with_stops(locations, optimized_route_sa, warehouse_labels)
