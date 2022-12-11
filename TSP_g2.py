"""
This script uses the tsplib95 library, which contains a lot of useful methods for 
solving TSP.

For documentation, see: https://tsplib95.readthedocs.io/en/stable/pages/usage.html#loading-problems
"""


import random
import math
from tqdm import tqdm
import tsplib95


####################### DATA FUNCTIONS ###################################### # # # # # # # # # 

# Using the tsp95 library to load the data
problem = tsplib95.load('TSP-Configurations/eil51.tsp')
all_edges = list(problem.get_edges())
cities = list(problem.get_edges())

# Coordinates of all the cities
coord_list = list(problem.node_coords.values()) 

######################## TSP FUNCTIONS #################################  # # # # # # # # # 

# Function to calculate the Euclidean distance between two cities
def distance(city1, city2):
  x1, y1 = city1
  x2, y2 = city2
  return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# Cost function that returns the total distance of a given route
def cost(route):
  total_distance = 0
  for i in range(len(route) - 1):
    total_distance += distance(route[i], route[i + 1])
  return total_distance

# Function to make a random 2-opt change to a given route
def two_opt(route):
  # Choose two edges at random and reverse the order of the cities they connect
  i, j = random.sample(range(len(route)), 2)
  if i > j:
    i, j = j, i
  route[i:j+1] = reversed(route[i:j+1])
  return route

# Function to implement the simulated annealing algorithm
def simulated_annealing(cities, temperature, cooling_rate):
  # Initialize the algorithm with a random route and the given temperature
  route = random.sample(cities, len(cities))

  with tqdm(total=10081) as pbar:
    while temperature > 10**(-42):
      # Make a random 2-opt change to the current route
      new_route = two_opt(route)
      # Calculate the cost of the new route
      cost_delta = cost(new_route) - cost(route)
      # If the new route has a lower cost, accept it as the current route
      if cost_delta < 0:
        route = new_route
      # If the new route has a higher cost, accept it with a certain probability
      elif random.random() < pow(math.e, -cost_delta / temperature):
        route = new_route
      # Decrease the temperature according to the cooling rate
      temperature *= 1 - cooling_rate

      pbar.update(1)
  # Return the final route as the solution to the TSP
  return route


# Define the list of cities
# cities = [(42.3600825, -71.0588801), (40.7128, -74.0060), (39.9526, -75.1652), (38.9072, -77.0369), (25.7617, -80.1918)]
cities = coord_list


# Solve the TSP using simulated annealing with the given parameters
solution = simulated_annealing(cities, 100, 0.01)

# Print the final solution
print(solution)