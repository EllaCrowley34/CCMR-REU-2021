import numpy as np
import random as rand
import sys
import time

import matplotlib as mpl
from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

t_steps = 100
r_steps = 50
P = 0.5     # probability that xi = 1
Q = 0.5     # probability that gamma = 1
iterations = 100     # number of times we populate a grid
                    # (number of grids we average over)


def initialize(t_max, r_max):
    """
        t_max is row number
        r_max is column number

        middle value of first row (first time step) is alive
        all other sites dead
    """
    grid = np.zeros((t_max,r_max), dtype=int)
    grid[0, r_max//2] = 1
    return grid

def populate(t_max, r_max):

    grid = initialize(t_max, r_max)

    for t in range(t_max-1):

        # Rules for first, last row sites with null boundary conditions
        grid[t+1, 0] = (Q <= rand.random()) if grid[t][0]\
        else (P > rand.random()) and grid[t][1]             # FASTER TO USE '*' OR 'and' HERE?

        grid[t+1, r_max-1] = (Q <= rand.random()) if grid[t][r_max-1]\
        else (P > rand.random()) and grid[t][r_max-2]

        # Rule for all sites with 2 neighbors
        for i in range(1,r_max-1):
            grid[t+1, i] = (Q <= rand.random()) if grid[t][i]\
            else (P > rand.random()) and (grid[t][i+1] + grid[t][i-1])

    return(grid)

def aggregate(t_max, r_max, iters):

    master = np.zeros((t_max, r_max))

    for i in range(iters):
        master += populate(t_max,r_max)

    master /= iters

    return master

def map_grid(grid, iters):

    #cmap = mpl.colors.ListedColormap(['blue','black'])
    c = plt.pcolormesh(grid)
    plt.colorbar(c)
    plt.title(f'CA Averaged over {iters} Runs')
    plt.show()

start_time = time.time()
map_grid(aggregate(t_steps, r_steps, iterations))
end_time = time.time()



