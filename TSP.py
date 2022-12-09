import numpy as np
import matplotlib.pyplot as plt




##### # # # # # # # # FUNCTIONS ###########

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