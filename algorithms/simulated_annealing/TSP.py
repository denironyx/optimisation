import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataframe for 7-cities
Data1 = pd.DataFrame([[0,75,99,9,35,63,8],[51,0,86,46,88,29,20],[100,5,0,16,28,35,28],
                      [20,45,11,0,59,53,49],[86,63,33,65,0,76,72],[36,53,89,31,21,0,52],
                      [58,31,43,67,52,60,0]], columns=["A","B","C","D","E","F","G"],
                     index=["A","B","C","D","E","F","G"]) # Dataframe for 7-cities

# Initial solution
X0 = ["A","C","G","D","E","B","F"] # Initial solution

# The OF of the initial solution
Distances = []
t = 0

for i in range(len(X0)-1):
    X1 = Data1.loc[X0[t], X0[t+1]] # Each city and the city after it
    X11 = Data1.loc[X0[-1], X0[0]] # The last city to the first city
    Distances.append(X1) # Append the distances
    t = t+1

Distances.append(X11) # Append the distance of the last city with the first one
Length_of_Travel = sum(Distances) # Add up the distances

print("Length of Travel:", Length_of_Travel)