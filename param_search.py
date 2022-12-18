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

# UNCOMMENT if acceptance rate tester should be re-run
# acceptances_per_temp = multi_run_temp_scanner(route, cities, temp_range, MCLen=100, cost_fn=cost, runs=100)
# np.save('Data/acceptances_per_temp.npy', acceptances_per_temp)

acceptances_per_temp = np.load('Data/acceptances_per_temp.npy')

print(acceptances_per_temp[70])
print(temp_range[70])


## PLOTTING ACCEPTANCE RATE VS TEMP ###
# for i in [10, 22, 48]:
#     plt.scatter(temp_range[i], acceptances_per_temp[i], color='r')
# plt.plot(temp_range, acceptances_per_temp)
# plt.xlabel('Initial temperature')
# plt.ylabel('Acceptance rate')
# plt.title('Initial Acceptance rate vs initial temperature')
# plt.show()

# T = 230 -> accept rate ~ 0.95


###### TESTING MC LENGTH & COOLING RATE ######
# TODO: allow for multiple initial temps to be tested

# function that runs simulated_annealing for a range of MC lengths and cooling rates
def mc_len_cool_rate(cities, mc_len_range, cool_rate_range, iters = 500, temp=100, n=3):
    """
    Runs simulated annealing for a range of MC lengths and cooling rates
    """
    results = np.zeros((len(mc_len_range), len(cool_rate_range), iters, 2))
    stds = np.zeros((len(mc_len_range), len(cool_rate_range), iters, 2))
    
    pbar = tqdm(total=len(mc_len_range)*len(cool_rate_range)*n)
    for mc_len in mc_len_range:
        num_chains = iters/mc_len
        for cool_rate in cool_rate_range:
            # Initialize arrays to store the results for each run
            costs = np.zeros((iters, n))
            lin_costs = np.zeros((iters, n))
            
            for i in range(n):
                # Run simulated annealing
                route, all_costs, lin_route, lin_costs_all = simulated_annealing(cities, temp, cool_rate, MCLen=mc_len, num_chains=num_chains, sweep=True, iters=iters)
                # print('MC length: ', mc_len, 'Cooling rate: ', cool_rate, 'Cost: ', cost(route), 'Lin Cost: ', cost(lin_route))

                # Store the results for this run
                costs[:, i] = np.asarray(all_costs).flatten()
                lin_costs[:, i] = np.asarray(lin_costs_all).flatten()

                pbar.update(1)

            # Calculate the average results over all runs
            results[mc_len_range.index(mc_len), cool_rate_range.index(cool_rate), :, 0] = np.mean(costs, axis=1)
            results[mc_len_range.index(mc_len), cool_rate_range.index(cool_rate), :, 1] = np.mean(lin_costs, axis=1)

            stds[mc_len_range.index(mc_len), cool_rate_range.index(cool_rate), :, 0] = np.std(costs, axis=1)
            stds[mc_len_range.index(mc_len), cool_rate_range.index(cool_rate), :, 1] = np.std(lin_costs, axis=1)
    
    np.save('Data/results.npy', results)

    return results, stds



# shape of results: [3, 2, 1000] / [mc_len_range, cool_rate_range, num_chains, cool_type]

def plot_results(results, stds, iters, mc_len_range, cool_rate_range):
    """
    Plots the results of the mc_len_cool_rate function
    """
    m_len = len(mc_len_range)
    c_len = len(cool_rate_range)


    fig, axs = plt.subplots(2, m_len, figsize=(10, 5), sharey=True)
    fig.suptitle('Cost of optimal solution per chain', fontsize=18)

    for i in range(m_len):
        for j in range(c_len):
            axs[0, i].plot(results[i,j,:,0], alpha=0.9, label=f'rate:{cool_rate_range[j]}')
            axs[0, i].fill_between(np.arange(0, iters), results[i,j,:,0] - stds[i,j,:,0], results[i,j,:,0] + stds[i,j,:,0], alpha=0.2)
            axs[0, i].set_title('MC length: ' + str(mc_len_range[i]))
            axs[0, i].set_xlim(0, iters)
            if i ==1:
                axs[0, i].legend(loc='upper right')

            axs[1, i].plot(results[i,j,:,1], alpha=0.9)
            axs[1, i].fill_between(np.arange(0, iters), results[i,j,:,1] - stds[i,j,:,1], results[i,j,:,1] + stds[i,j,:,1], alpha=0.2)
            axs[1, i].set_xlim(0, iters)
            # axs[1, i].legend(cool_rate_range, loc='upper right')

    # Add y label for left side of plot
    fig.text(0.01, 0.5, 'Cost', ha='left',va='center', rotation='vertical', fontsize=14)
    

    # Add subplot titles
    axs[0, m_len-1].set_ylabel('Geometric Cooling', fontsize=14, ha='left')
    axs[1, m_len-1].set_ylabel('Linear Cooling', fontsize=14, ha='right')
    axs[1, m_len//2].set_xlabel('MC Number / Cooling Step', fontsize=14)
    plt.tight_layout()
    plt.show()


##### EXPERIMENTS #####
# SETUP: FIRST PLOT: SHOW THAT THERE IS NO DIFFERENCE BETWEEN GEOMETRIC AND LINEAR COOLING
# MAKE A PLLOT SHOWING HOW DIFFERENT MC LENS DON'T CONVERGE FOR DIFFERENT COOLING RATES
# EACH PLOT WILL HAVE 3 LINES FOR 3 DIFFERENT INITIAL TEMPS

# mc_len_range = [10, 20, 50, 100, 200, 500, 1000]
mc_len_range = [10, 50, 100] # [10,100,250]
cool_rate_range = [0.01, 0.05, 0.1, 0.2]
iters=30000
n=5
temps=[76, 106, 171]

# UNCOMMENT if MC length and cooling rate tester should be re-run
# for t in temps:
#     results_sweep, stds_sweep = mc_len_cool_rate(cities, mc_len_range, cool_rate_range, temp=t, iters=iters, n=n)
#     f_mean = f'Data/means_temp_' + str(t) + '.npy'
#     f_std = f'Data/stds_temp_' + str(t) + '.npy'
#     np.save(f_mean, results_sweep)
#     np.save(f_std, stds_sweep)


# # plot results
results_sweep = np.load('Data/means_temp_' + str(106) + '.npy')
stds_sweep = np.load('Data/stds_temp_' + str(106) + '.npy')

plot_results(results_sweep, stds_sweep, iters, mc_len_range, cool_rate_range)

print(results_sweep[0,1,-1,0])

# EXPERIMENTAL DESIGN || SWEEP ||

# Cool_rates = [0.01, 0.05, 0.2]
# N_runs = 50
# N_chains = 600


# EXPERIMENTAL DESIGN || ROUTEFINDER ||
# Cool_rate = [0.001]
# N_runs = 100



### PREVIOUS WORKING ONE, UNCOMMENT TO COMPARE PLOTTING STYLES ###
# def plot_results(results, mc_len_range, cool_rate_range):
#     """
#     Plots the results of the mc_len_cool_rate function
#     """
#     m_len = len(mc_len_range)
#     c_len = len(cool_rate_range)

#     fig, axs = plt.subplots(m_len, c_len, figsize=(10, 5), sharex=False)
#     fig.suptitle('Cost of optimal solution per chain', fontsize=18)

#     for i in range(m_len):
#         for j in range(c_len):
#             axs[i, j].plot(results[i,j,:,0], label='Geometric Cooling')
#             axs[i, j].plot(results[i,j,:,1], label='Linear Cooling', alpha = 0.5)
#             axs[i, j].set_title('MC length: ' + str(mc_len_range[i]) + ', Cooling rate: ' + str(cool_rate_range[j]))
#             if i == m_len-1 and j == 1:
#                 axs[i, j].set_xlabel('MC Number / Cooling Step', fontsize=14)
#             if j == 0 and i == 1:
#                 axs[i, j].set_ylabel('Cost', fontsize=14)
#             axs[i, j].set_xlim(0, 500)
#             if i == 0 and j == 2:
#                 axs[i, j].legend()
#     plt.tight_layout()
#     plt.show()