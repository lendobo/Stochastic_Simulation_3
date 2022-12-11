import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tsplib95
import time


# # Pandas Loader
# problem_scope = 'eil51.tsp.txt'
# cities = pd.read_csv(f'TSP-Configurations/{problem_scope}', sep=" ", skiprows=5)
# cities.columns = ['City', 'X', 'Y']

# print(cities.head())

data = tsplib95.load('TSP-Configurations/eil51.tsp.txt')


##### # # # # # # # # FUNCTIONS ###########

def distance_table(cities):
    """
    Calculates distances beetween all cities once and stores in a triangular matrix.
    """
    euc_dist = 1
    

def cost_calc():
    """
    Calculates the distance between all possible cities
    """
    dist = 0

    for i in range(len(area)):
        from_city = area[i]
        to_city = None

        # Ensure All City distances are calculated
        if i+1 < len(area):
            to_city = state[i+1]
        else:
            to_city = state[0]


def cool_schedule(t_start, t_end, cr):
    """
    
    Args
        t_start     starting temperature
        t_end       end temperature
        cr          cooling rate for simulated annealing
    """
    temp = t_start
    cost = np.infty

    # cooling schedule
    while temp > t_end:
        #calculate cost function
        cost_nu = cost_func
        diff = cost_nu - cost

        # Aacceptance conditions for new state
        if diff < 0 or np.exp(-(diff/temp)) > np.random.uniform():
            cost = cost_nu
        
        # Cool
        temp = temp * cr

def two_opt(cities):
    """
    2-opt method for introducing small perturbation into circuit
    """
    pos1 = random.choice(range(len(cities)))
    pos2 = random.choice(range(len(cities)))

    cities[pos1], cities[pos2] = cities[pos2], cities[pos1]

    return cities

class City():
    pass