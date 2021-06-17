""" sri_CA.py

    Implements a probabilistic CA
    Repeats run for a specified number of iterations
    Aggregates an 'average' grid
    Generates average and smoothed average lists for origin values

    Language: python 3

    Ella Crowley
    Cornell University

    Written for CCMR REU, Summer 2021.
    6/17/21 on Mac.

"""

import numpy as np
import random as rand
import sys
import time

import matplotlib as mpl
from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)


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

def populate(t_max, r_max, p, q):
    """
        t_max is row number
        r_max is column number

        Fills grid according to CA rules.
        Null boundary conditions.
        Fills rows in order (except last site -- fills second)
    """

    grid = initialize(t_max, r_max)

    for t in range(t_max-1):

        # Rules for first, last row sites with null boundary conditions
        grid[t+1, 0] = (q <= rand.random()) if grid[t][0]\
        else (p > rand.random()) and grid[t][1]

        grid[t+1, r_max-1] = (q <= rand.random()) if grid[t][r_max-1]\
        else (p > rand.random()) and grid[t][r_max-2]

        # Rule for all sites with 2 neighbors
        for i in range(1,r_max-1):
            grid[t+1, i] = (q <= rand.random()) if grid[t][i]\
            else (p > rand.random()) and (grid[t][i+1] or grid[t][i-1])

    return(grid)

def C_avg(grid):
    """
        grid is t_max by r_max array
        generates a list of the origin values over time
    """
    return( [row[len(row)//2] for row in grid] )

def weighted_avg(grid):
    origin = grid.shape[1]//2
    return( [ ((row[origin - 1] + row[origin] + row[origin + 1]) / 3) for row in grid ] )

def aggregate(t_max, r_max, iters, p, q):

    master = np.zeros((t_max, r_max))

    for i in range(iters):
        master += populate(t_max,r_max, p, q)

    master /= iters

    return master


def late_central_avg(t_max, r_max, iters, slices, lateness, q):
    """
        slices (int): how many values of P we will try
            (for slices = 100, P will have values 1/100 to 100/100)
        lateness (float, 0 - 1): time step origin value sampled at
            (for lateness = 0.9, origin values from t = t_max * 0.9)
        q (float): probability that gamma = 1
    """
    c_list = []
    t_final = int(t_max*lateness)
    origin = r_max // 2

    for i in reversed(range(slices)):
        p = 1 / (i+1)
        c_list.append(aggregate(t_max, r_max, iters, p, q)[t_final,origin])

    return(c_list)

def smoothed_c_list(c_list):
    sc_list = [ (c_list[i] + c_list[i-1]) / 2 for i in range(1, len(c_list)) ]
    sc_list.insert(0, c_list[0])
    return(sc_list)

def map_grid(grid, iters):

    c = plt.pcolormesh(grid, cmap='Greys', norm=mpl.colors.LogNorm())   # can also get rid of log norm maybe
    plt.colorbar(c)
    plt.title(f'CA Averaged over {iters} Runs')
    plt.savefig(f'CA_{iters}_{len(grid)}t')
    #plt.show()
    plt.clf()

def plot_avg(avg_list, iters, weighted=False):

    plt.plot(avg_list)
    plt.xlabel('Time Step')

    if weighted:
        plt.ylabel('Average of 3 Central Values')
        plt.title(f'Mean of 3 Central Values, Mean over {iters} Runs')
        plt.savefig(f'OriginAvgW_{iters}_{len(avg_list)}t.png')

    else:
        plt.ylabel('Average Origin Value')
        plt.title(f'Origin Value, Mean over {iters} Runs')
        plt.savefig(f'OriginAvg_{iters}_{len(avg_list)}t.png')

    #plt.show()
    plt.clf()

def plot_clist(c_list, iters, slices, lateness, weighted=False):
    plt.plot( np.linspace(0,1,len(c_list)), c_list )
    plt.xlabel('P(Xi = 1)')

    if weighted:
        plt.ylabel(f'Average of Origin Values at {lateness}t_max, {lateness}t_max - 1')
        plt.title(f'Mean of Origin Values at {lateness}t_max, Mean over {iters} Runs')
        plt.savefig(f'LateOriginAvgW_{iters}_{len(c_list)}p, {lateness}t.png')

    else:
        plt.ylabel('Average Origin Value')
        plt.title(f'Origin Value at {lateness}t_max, Mean over {iters} Runs')
        plt.savefig(f'LateOriginAvg_{iters}_{len(c_list)}p, {lateness}t.png')

    #plt.show()
    plt.clf()

def main(timer=False):

    t_steps = 50
    r_steps = 50
    P = 0.5     # probability that xi = 1
    Q = 0.2     # probability that gamma = 1
    iterations = 100    # number of times we populate a grid
                    # (number of grids we average over)
    slices = 100
    lateness = 0.9

    start_time = time.time()

    #grid = aggregate(t_steps, r_steps, iterations, P, Q)
    #map_grid(grid, iterations)
    #plot_avg(C_avg(grid), iterations)
    #plot_avg(weighted_avg(grid), iterations, weighted=True)
    clist = late_central_avg(t_steps, r_steps, iterations, slices, lateness, Q)
    sclist = smoothed_c_list(clist)
    plot_clist(clist, iterations, slices, lateness)
    plot_clist(sclist, iterations, slices, lateness, weighted=True)

    end_time = time.time()

    if timer:
        print(f"Runtime: {end_time - start_time}")


if __name__ == "__main__":
    main()



