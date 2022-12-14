"""
This script uses the tsplib95 library, which contains a lot of useful methods for 
solving TSP.

For documentation, see: https://tsplib95.readthedocs.io/en/stable/pages/usage.html#loading-problems


Some inspiration for solving TSP with SA: 
https://raw.githubusercontent.com/goossaert/algorithms/master/simulated_annealing/annealing.py
https://medium.com/@francis.allanah/travelling-salesman-problem-using-simulated-annealing-f547a71ab3c6
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import math
from tqdm import tqdm
import tsplib95
import copy
import pickle


####################### DATA FUNCTIONS ###################################### # # # # # # # # # 

# Using the tsp95 library to load the data
problem = tsplib95.load('TSP-Configurations/a280.tsp.txt')
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

def metropolis(route_0, cities, temperature, MCLen=100, cost_fn=cost):
    costs = []
    route = route_0
    for i in range(MCLen):
        new_route = get_next(route)
        cost_delta = cost_fn(new_route) - cost_fn(route)
        if cost_delta < 0:
            route = new_route
            # print('CHANGE')
        elif random.random() < pow(math.e, - cost_delta / temperature):
            route = new_route
            # print('CHANGE')
        costs.append(cost_fn(route))
    return route, costs

"Function to implement the simulated annealing algorithm"
def simulated_annealing(cities, temperature, cooling_rate, sweep=False, num_chains=5000, MCLen=100):

  # Initialize the algorithm with a random route and the given temperature
  route = random.sample(cities, len(cities))
  lin_route = copy.deepcopy(route)

  costs_all = []
  lin_costs_all = []

  if sweep == False:
    # determining size of the progress bar
    chain = 0
    with tqdm(total=num_chains) as pbar:
      while chain < num_chains:
          # Make a random 2-opt change to the current route
          route, costs = metropolis(route, cities, temperature, MCLen=MCLen)

          # Decrease the temperature according to the cooling rate
          temperature = np.max([temperature * (1 - cooling_rate), 10**(-42)])
   
          costs_all.append(costs)

          chain += 1
          pbar.update(1)  # Return the final route as the solution to the TSP
  else:
    chain = 0
    lin_temp_0 = temperature # initial temperature for linear cooling rate
    lin_temp = lin_temp_0
    x = 0
    rate_ratio = cooling_rate / 0.01
    while chain < num_chains:
      route, costs = metropolis(route, cities, temperature, MCLen=MCLen)
      costs_all.append(costs)

      lin_route, lin_costs = metropolis(lin_route, cities, lin_temp, MCLen=MCLen)
      lin_costs_all.append(lin_costs)

      chain += 1

      temperature = np.max([temperature * (1 - cooling_rate), 10**(-42)])
      lin_temp = np.max([lin_temp_0 - rate_ratio * (lin_temp_0/num_chains) * x, 10**(-42)])

      x+=1
      print(lin_temp)
      

    return route, costs_all, lin_route, lin_costs_all

  return route, costs_all


if __name__ == '__main__':
  # Define the list of cities
  cities = coord_list

  # Solve the TSP using simulated annealing with the given parameters
  all_cost = []
  for rep in range(5):
    solution, costs = simulated_annealing(cities, 170, 0.05, sweep=False, num_chains=2000, MCLen=500)
    solution.append(solution[0])

    print(cost(solution))
    plt.figure(0)
    # print(costs)
    all_cost.append(costs)
  
  with open("Data/route_costs", "wb") as fp:   #Pickling
    pickle.dump(all_cost, fp)
  
  costs = np.mean(all_cost, axis=0)
  stds = np.std(all_cost, axis=0)

  with open("Data/route_solution", "wb") as fp:   #Pickling
    pickle.dump(solution, fp)

  with open("Data/route_solution", "rb") as fp:   # Unpickling
    solution = pickle.load(fp)

  plt.figure(1)

  xs = [item[0] for item in solution]
  ys = [item[1] for item in solution]

  plt.plot(xs, ys)
  plt.xlabel('X Coordinate')
  plt.ylabel('Y Coordinate')
  plt.show()

  # Print the final solution
  # print(solution)