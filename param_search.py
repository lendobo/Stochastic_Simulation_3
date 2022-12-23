import numpy as np
import tsplib95
import random
import copy
from TSP_homochain import *

### DATA ###

problem = tsplib95.load('TSP-Configurations/a280.tsp.txt')
cities  = list(problem.node_coords.values()) 

### FUNCTIONS FOR ACCEPTANCE RATE TESTING ###

def init_temp_scanner(route_0, cities, temp_range, MCLen=500, cost_fn=cost):
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
            total += 1
        accept_rate = accept / total
        accept_per_temp.append(accept_rate)

    return accept_per_temp

def multi_run_temp_scanner(route_0, cities, temp_range, MCLen=500, cost_fn=cost, runs=50):
    """
    Runs the init_temp_scanner function multiple times and returns the average acceptance rate
    """
    accept_per_temp = []
    pbar = tqdm(total=runs)
    for i in range(runs):
        accept_per_temp.append(init_temp_scanner(route_0, cities, temp_range, MCLen, cost_fn))
        pbar.update(1)
    return np.mean(accept_per_temp, axis=0), np.std(accept_per_temp, axis=0)

### TESTING ####

temp_range = np.linspace(50,300,100)
route = random.sample(cities, len(cities))

# # UNCOMMENT if acceptance rate tester should be re-run
# acceptances_per_temp, acc_stds = multi_run_temp_scanner(route, cities, temp_range, MCLen=100, cost_fn=cost, runs=100)
# np.save('Data/acceptances_per_temp.npy', acceptances_per_temp)
# np.save('Data/acc_stds.npy', acc_stds)


# # ## PLOTTING ACCEPTANCE RATE VS TEMP ###
# acceptances_per_temp = np.load('Data/acceptances_per_temp.npy')
# acc_stds = np.load('Data/acc_stds.npy')


# for i in [8, 48, -4]:
#     plt.scatter(temp_range[i], acceptances_per_temp[i], color='r')
# plt.plot(temp_range, acceptances_per_temp)
# plt.fill_between(temp_range, acceptances_per_temp-acc_stds, acceptances_per_temp+acc_stds, alpha=0.2)
# plt.xlabel('Initial temperature', fontsize=15)
# plt.ylabel('Acceptance rate', fontsize=15)
# plt.title('Initial Acceptance rate vs initial temperature', fontsize=15)
# plt.xlim(65,300)
# plt.show()










###### FUNCTIONS FOR TESTING MC LENGTH & COOLING RATE ########

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
                route, all_costs, lin_route, lin_costs_all = simulated_annealing(cities, temp, cool_rate, MCLen=mc_len, num_chains=num_chains, sweep=True)
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

# THIS ONE PLOT GEO AND LIN AND OPTIMAL COST PLOT
def plot_results(results, stds, iters, mc_len_range, cool_rate_range):
    """
    Plots the results of the mc_len_cool_rate functiion, comparing linear and cooling rates, 
    as well as the final cost of the optimal solution.
    """
    m_len = len(mc_len_range)
    c_len = len(cool_rate_range)

    if m_len > 1:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        i = 0
        for j in range(c_len):
            axs[0, 0].plot(results[i,j,:,0], alpha=0.9, label=f'rate:{cool_rate_range[j]}')
            axs[0, 0].fill_between(np.arange(0, iters), results[i,j,:,0] - stds[i,j,:,0], results[i,j,:,0] + stds[i,j,:,0], alpha=0.2)
            axs[0, 0].set_title('MC length: ' + str(mc_len_range[i]))
            axs[0, 0].set_xlim(0, iters)
            if i == 0:
                axs[0, i].legend(loc='upper right')

            axs[0, 1].plot(results[1,j,:,1], alpha=0.9)
            axs[0, 1].fill_between(np.arange(0, iters), results[i,j,:,1] - stds[i,j,:,1], results[i,j,:,1] + stds[i,j,:,1], alpha=0.2)
            axs[0, 1].set_xlim(0, iters)

            
            axs[0, 1].set_facecolor('0.9')
            # axs[0, i].set_facecolor('0.9')
            fig.suptitle('Comparing Linear and Geometric Cooling', fontsize=18)



        axs[1, m_len//2].set_xlabel('MC Number / Cooling Step', fontsize=14)

    else:
        fig, axs = plt.subplots(1, m_len, figsize=(10, 5), sharey=True)
        axs[0].plot(results[0,0,:,0], alpha=0.9, label=f'rate:{cool_rate_range[0]}')
        axs[0].fill_between(np.arange(0, iters), results[0,0,:,0] - stds[0,0,:,0], results[0,0,:,0] + stds[0,0,:,0], alpha=0.2)
        axs[0].set_title('MC length: ' + str(mc_len_range[0]))
        axs[0].set_xlim(0, iters)
        axs[0].legend(loc='upper right')

        

    # Add y label for left side of plot
    fig.text(0.01, 0.5, 'Cost', ha='left',va='center', rotation='vertical', fontsize=14)
    
    plt.show()

# THIS ONE ONLY PLOT THE SWEEPS
def plot_sweep_results(all_results, stds, iters, mc_len_range, cool_rate_range):
    """
    Plots the results of the mc_len_cool_rate function
    """
    m_len = len(mc_len_range)
    c_len = len(cool_rate_range)

    # use ggplot style for more sophisticated visuals
    plt.style.use('ggplot')

    fig, axs = plt.subplots(3, m_len, figsize=(10,4), sharey=True, sharex=True)
    # fig.suptitle('Cost per Chain', fontsize=18, y=1)

    colrs = ['dodgerblue', 'darkgoldenrod', 'firebrick']
    xlabels = [i*10 for i in range(6)]

    for t, results in enumerate(all_results):
        for i in range(m_len):
            for j in range(c_len):
                axs[t, i].plot(results[i,j,:,0], alpha=0.9, label=f'rate:{cool_rate_range[j]}', color = colrs[j])
                axs[t, i].fill_between(np.arange(0, iters), results[i,j,:,0] - stds[t][i,j,:,0], results[i,j,:,0] + stds[t][i,j,:,0], alpha=0.2, color = colrs[j])
            axs[t, i].set_xlim(0, iters)
            if t == 0:
                axs[t, i].set_title(f'MC length: {mc_len_range[i]}')
                if i == 0:
                    axs[t, i].legend(loc='upper right')
            axs[t, i].tick_params(axis='y', labelrotation=45)
            axs[t, i].set_xticks([0, 10, 20, 30, 40, 50])


    axs[2,m_len//2].set_xlabel('Iterations (Thousands)', fontsize=14, color='black')


    # Add y label for left side of plot, with padding
    fig.text(0.03, 0.5, 'Cost', ha='left',va='center', rotation='vertical', fontsize=14)
    fig.text(0.92, 0.5, rf'$T_0:170$', ha='left',va='center', rotation='vertical', fontsize=14)
    fig.text(0.92, 0.23, rf'$T_0:300$', ha='left',va='center', rotation='vertical', fontsize=14)
    fig.text(0.92, 0.77, rf'$T_0:75$', ha='left',va='center', rotation='vertical', fontsize=14)



    # plt.tight_layout()
    plt.show()









##### EXPERIMENTS #####

mc_len_range = [50,250, 1000]
cool_rate_range = [0.01, 0.05, 0.2]
iters=50000
n=10
temps= [75, 170, 300]

# # # # UNCOMMENT if MC length and cooling rate tester should be re-run
# for t in temps:
#     results_sweep, stds_sweep = mc_len_cool_rate(cities, mc_len_range, cool_rate_range, temp=t, iters=iters, n=n)
#     f_mean = f'Data/means_temp_' + str(t) + str(mc_len_range[0]) + '_' + str(mc_len_range[-1]) + '_iters_' + str(iters) + '.npy'
#     f_std = f'Data/stds_temp_' + str(t) + str(mc_len_range[0]) + '_' + str(mc_len_range[-1]) + '_iters_' + str(iters) + '.npy'
#     np.save(f_mean, results_sweep)
#     np.save(f_std, stds_sweep)


results_sweep = []
stds_sweep = []
# # plot results
for t in temps:
    results_sweep.append(np.load(f'Data/means_temp_' + str(t) + str(mc_len_range[0]) + '_' + str(mc_len_range[-1]) + '_iters_' + str(iters) + '.npy'))
    stds_sweep.append(np.load(f'Data/stds_temp_' + str(t) + str(mc_len_range[0]) + '_' + str(mc_len_range[-1]) + '_iters_' + str(iters) + '.npy'))

    # geo_lin_res = np.load(f'Data/means_temp_' + str(t) + str(mc_len_range[0]) + '_' + str(mc_len_range[-1]) + '_iters_' + str(iters) + '.npy')
    # geo_lin_std = np.load(f'Data/stds_temp_' + str(t) + str(mc_len_range[0]) + '_' + str(mc_len_range[-1]) + '_iters_' + str(iters) + '.npy')


plot_sweep_results(results_sweep, stds_sweep, iters, mc_len_range, cool_rate_range)