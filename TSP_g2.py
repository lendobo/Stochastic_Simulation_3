"""
This script uses the tsplib95 library, which contains a lot of useful methods for 
solving TSP.

For documentation, see: https://tsplib95.readthedocs.io/en/stable/pages/usage.html#loading-problems


Some inspiration for solving TSP with SA: 
https://raw.githubusercontent.com/goossaert/algorithms/master/simulated_annealing/annealing.py
https://medium.com/@francis.allanah/travelling-salesman-problem-using-simulated-annealing-f547a71ab3c6
"""

import matplotlib.pyplot as plt
import random
import math
from tqdm import tqdm
import tsplib95
import copy


####################### DATA FUNCTIONS ###################################### # # # # # # # # # 

# Using the tsp95 library to load the data
problem = tsplib95.load('TSP-Configurations/eil51.tsp.txt')
all_edges = list(problem.get_edges())
cities = list(problem.get_edges())

# Coordinates of all the cities
coord_list = list(problem.node_coords.values()) 

######################## TSP FUNCTIONS #################################  # # # # # # # # # 

"Function to calculate the Euclidean distance between two cities"
def distance(city1, city2):

  x1, y1 = city1
  x2, y2 = city2
  return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


"Cost function that returns the total distance of a given route"
def cost(route):

  total_distance = 0

  for i in range(-1, len(route) - 1):

    total_distance += distance(route[i], route[i + 1])

  return total_distance


# This is in case we dont want the complication of i->j
# def shift(route, i, j):
#   return route, j, i


"Treats the path as a circle, shifts list to allow reverse"
def shift(route, i, j):

  for iteration in range(len(route) - i):

    route.append(route.pop(0))

  return route, 0, j + (len(route) - i)


"Chooses which type of change to make to the route"
def get_next(route):

  next_route = copy.deepcopy(route)

  func = random.choice([0,1,2])
  if func == 0:
    next_route = reverse(next_route)
  elif func == 1:
    next_route = relocation(next_route)
  elif func == 2:
    next_route = insert_city(next_route)
  else:
    next_route = swap(next_route)

  return next_route


"Function to make a random 2-opt change to a given route"
def reverse(route):

  # Choose two edges at random and reverse the order of the cities they connect
  indices = random.sample(range(len(route)), 2)
  i, j = indices[0], indices[1]
  if i > j:
    route, i, j = shift(route, i, j)
  route_copy = copy.deepcopy(route)
  route_copy[i:j+1] = list(reversed(route[i:j+1]))
  return route_copy


"Select a subroute from a to b and insert it at another position in the route"
def relocation(route):

  indices = random.sample(range(len(route)), 2)
  i, j = indices[0], indices[1]

  if i > j:
    route, i, j = shift(route, i, j)

  subroute = route[i:j]
  del route[i:j]
  insert_pos = random.choice(range(len(route)))

  for i in subroute:

      route.insert(insert_pos, i)

  return route


def insert_city(route):
    "Insert city at node j before node i"
    node_j = random.choice(route)
    route.remove(node_j)
    node_i = random.choice(route)
    index = route.index(node_i)
    route.insert(index, node_j)
    
    return route

"Takes two indices at random and swaps the cities at said indices"
def swap(route):

  indices = random.sample(range(len(route)), 2)
  i, j = indices[0], indices[1]
  route_copy = copy.deepcopy(route)
  route_copy[i] = route[j]
  route_copy[j] = route[i]
  return route_copy


"Function to implement the simulated annealing algorithm"
def simulated_annealing(cities, temperature, cooling_rate):

  # Initialize the algorithm with a random route and the given temperature
  route = random.sample(cities, len(cities))
  costs = []

  with tqdm(total=105174) as pbar:

    while temperature > 10**(-42):

    # Make a random 2-opt change to the current route
      new_route = get_next(route)
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

      costs.append(cost(route))
      pbar.update(1)  # Return the final route as the solution to the TSP

  return route, costs


# Define the list of cities
# cities = [(42.3600825, -71.0588801), (40.7128, -74.0060), (39.9526, -75.1652), (38.9072, -77.0369), (25.7617, -80.1918)]
cities = coord_list

# Solve the TSP using simulated annealing with the given parameters
solution, costs = simulated_annealing(cities, 5000, 0.001)
solution.append(solution[0])
print(cost(solution))
plt.figure(0)
plt.plot(range(len(costs[math.floor(len(costs)/10):])), costs[math.floor(len(costs)/10):])

plt.figure(1)

xs = [item[0] for item in solution]
ys = [item[1] for item in solution]

plt.plot(xs, ys)
plt.show()

# Print the final solution
print(solution)