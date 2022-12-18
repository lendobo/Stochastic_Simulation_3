import numpy as np
import tsplib95
import random
import copy
from TSP_homochain import *

### DATA ###

problem = tsplib95.load('TSP-Configurations/eil51_x.tsp.txt')
cities  = list(problem.node_coords.values()) 

### FUNCTIONS ###

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
    pbar = tqdm(total=runs)
    for i in range(runs):
        accept_per_temp.append(init_temp_scanner(route_0, cities, temp_range, MCLen, cost_fn))
        pbar.update(1)
    return np.mean(accept_per_temp, axis=0)

### TESTING ####

temp_range = np.linspace(50,300,100)
route = random.sample(cities, len(cities))

## UNCOMMENT if acceptance rate tester should be re-run
# acceptances_per_temp = multi_run_temp_scanner(route, cities, temp_range, MCLen=100, cost_fn=cost, runs=100)
# np.save('Data/acceptances_per_temp.npy', acceptances_per_temp)

acceptances_per_temp = np.load('Data/acceptances_per_temp.npy')
print(acceptances_per_temp[70])
print(temp_range[70])


### PLOTTING ACCEPTANCE RATE VS TEMP ###
# plt.plot(temp_range, acceptances_per_temp)
# plt.xlabel('Initial temperature')
# plt.ylabel('Acceptance rate')
# plt.title('Initial Acceptance rate vs initial temperature')
# plt.show()

# T = 230 -> accept rate ~ 0.95


###### TESTING MC LENGTH & COOLING RATE ######
# TODO: allow for multiple initial temps to be tested

# function that runs simulated_annealing for a range of MC lengths and cooling rates
def mc_len_cool_rate(cities, mc_len_range, cool_rate_range, temp=230):
    """
    Runs simulated annealing for a range of MC lengths and cooling rates
    """
    num_chains = 1000

    results = np.zeros((len(mc_len_range), len(cool_rate_range), num_chains, 2))
    
    pbar = tqdm(total=len(mc_len_range)*len(cool_rate_range))
    for mc_len in mc_len_range:
        for cool_rate in cool_rate_range:
            route, all_costs, lin_route, lin_costs_all = simulated_annealing(cities, temp, cool_rate, MCLen=mc_len, num_chains=num_chains, sweep=True)
            print('MC length: ', mc_len, 'Cooling rate: ', cool_rate, 'Cost: ', cost(route), 'Lin Cost: ', cost(lin_route))
            
            chain_optima = [np.min(costs) for costs in all_costs]
            lin_chain_optima = [np.min(costs) for costs in lin_costs_all]

            results[mc_len_range.index(mc_len), cool_rate_range.index(cool_rate), :, 0] = chain_optima
            results[mc_len_range.index(mc_len), cool_rate_range.index(cool_rate), :, 1] = lin_chain_optima

            pbar.update(1)
    
    np.save('Data/results.npy', results)

    return results

# mc_len_range = [10, 20, 50, 100, 200, 500, 1000]
mc_len_range = [5,10,20]
cool_rate_range = [0.01, 0.1, 0.2]

# # UNCOMMENT if MC length and cooling rate tester should be re-run
# results_sweep = mc_len_cool_rate(cities, mc_len_range, cool_rate_range, temp=230)
# np.save('Data/results.npy', results_sweep)

# shape of results: [3, 2, 1000] / [mc_len_range, cool_rate_range, num_chains]

# function that makes a 2x3 plot of the results of the mc_len_cool_rate function
def plot_results(results, mc_len_range, cool_rate_range):
    """
    Plots the results of the mc_len_cool_rate function
    """
    m_len = len(mc_len_range)
    c_len = len(cool_rate_range)

    fig, axs = plt.subplots(m_len, c_len, figsize=(10, 5), sharex=False)
    fig.suptitle('Cost of optimal solution per chain', fontsize=18)

    for i in range(m_len):
        for j in range(c_len):
            axs[i, j].plot(results[i,j,:,0], label='Geometric Cooling')
            axs[i, j].plot(results[i,j,:,1], label='Linear Cooling', alpha = 0.5)
            axs[i, j].set_title('MC length: ' + str(mc_len_range[i]) + ', Cooling rate: ' + str(cool_rate_range[j]))
            if i == m_len-1 and j == 1:
                axs[i, j].set_xlabel('MC Number / Cooling Step', fontsize=14)
            if j == 0 and i == 1:
                axs[i, j].set_ylabel('Cost', fontsize=14)
            axs[i, j].set_xlim(0, 500)
            if i == 0 and j == 2:
                axs[i, j].legend()
    plt.tight_layout()
    plt.show()


# plot results
results_sweep = np.load('Data/results.npy')
plot_results(results_sweep, mc_len_range, cool_rate_range)
