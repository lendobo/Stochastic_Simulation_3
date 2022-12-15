import numpy as np
import tsplib95
import random
import copy
from TSP_homochain import *

problem = tsplib95.load('TSP-Configurations/eil51_x.tsp.txt')
cities  = list(problem.node_coords.values()) 

def init_temp_scanner(route_0, cities, temp_range, MCLen=100, cost_fn=cost):
    """
    Checks the acceptance rate of monte carlo moves for a range of initial temperatures
    """
    route = route_0
    
    accept_per_temp = []

    for t in temp_range:
        accept = 0
        total = 0
        temperature = t
        for i in range(MCLen):
            new_route = get_next(route)
            cost_delta = cost_fn(new_route) - cost_fn(route)
            if cost_delta < 0:
                route = new_route
                accept += 1
            elif random.random() < pow(math.e, - cost_delta / temperature):
                route = new_route
                accept += 1
            # costs.append(cost_fn(route))
            total += 1
        accept_rate = accept / total
        accept_per_temp.append(accept_rate)

    return accept_per_temp

def multi_run_temp_scanner(route_0, cities, temp_range, MCLen=100, cost_fn=cost, runs=50):
    """
    Runs the init_temp_scanner function multiple times and returns the average acceptance rate
    """
    accept_per_temp = []
    for i in range(runs):
        accept_per_temp.append(init_temp_scanner(route_0, cities, temp_range, MCLen, cost_fn))
    return np.mean(accept_per_temp, axis=0)

### TESTING ####

temp_range = np.linspace(10,500,50)
route = random.sample(cities, len(cities))

acceptances = multi_run_temp_scanner(route, cities, temp_range, MCLen=100, cost_fn=cost, runs=10)

plt.plot(temp_range, acceptances)
plt.show()