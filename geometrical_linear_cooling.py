import numpy as np
import matplotlib.pyplot as plt

def geo_cool(t, cool_rate, n):
    """
    Geometric cooling function
    """
    temps = np.zeros(n)
    for i in range(n):
        temps[i] = t * (1-cool_rate)**i
    return temps

def lin_cool(t, cool_rate, n):
    """
    Linear cooling function
    """
    temps = np.zeros(n)
    for i in range(n):
        tempy_temp = t - cool_rate* (t/n)*i
        temps[i] = np.max([tempy_temp, 0])
    return temps

t_max = 230
n = 500

cooling_rates = [0.01, 0.1, 0.2]
rate_ratios = [cooling_rates[i]/cooling_rates[0] for i in range(len(cooling_rates))]


# function that creates 3 subplots for each cooling rate
def plot_cooling_rates(cooling_rates, t_max, n):
    """
    Plots the cooling rates for the geometric and linear cooling functions
    """
    fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    for i in range(len(cooling_rates)):
        axs[i].plot(lin_cool(t_max, rate_ratios[i], n), label='Linear')
        axs[i].plot(geo_cool(t_max, cooling_rates[i], n), label='Geometric')
        axs[i].set_title('Cooling rate: ' + str(cooling_rates[i]))
        axs[i].set_xlabel('MC length')
        axs[i].set_ylabel('Temperature')
        axs[i].legend()
    plt.show()

plot_cooling_rates(cooling_rates, t_max, n)
