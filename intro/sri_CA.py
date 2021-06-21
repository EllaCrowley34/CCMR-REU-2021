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
import argparse
import os
import matplotlib as mpl
from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def usage(exit_code=0):
    progname = os.path.basename(sys.argv[0])
    print(f'''Usage: {progname} [-a ALPHABET -c CORES -l LENGTH -p PATH -s HASHES]
    -t TIME         Number of time steps                (1000 default)
    -r SITES        Number of sites per time steps      (2000)
    -p P(Xi = 1)    Probability xi = 1                  (0.5)
    -q P(Gamma = 1) Probability gamma = 1               (0.5)
    -i ITERATIONS   Number of CA runs in average        (1)
    -s SLICES       Number of dial points when tuning P (100)
    -l LATENESS     % of max time to sample central vals(0.9)
    -c CLOCK        Prints times of calcs in main''')
    sys.exit(exit_code)



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
        qrand = np.random.rand(r_max)
        prand = np.random.rand(r_max)

        grid[t+1] = [ (q <= qrand[i]) if grid[t][i]\
        else (grid[t][i+1] or grid[t][i-1]) and (p > prand[i])\
        for i in range(r_max-1) ] + [0]

    return(grid)



def aggregate(t_max, r_max, iters, p, q):

    master = np.zeros((t_max, r_max))

    for i in range(iters):
        master += populate(t_max,r_max, p, q)

    master /= iters

    return master



def C_avg(grid):
    """
        grid is t_max by r_max array
        generates a list of the origin values over time
    """
    return( [row[len(row)//2] for row in grid] )



def weighted_avg(grid):
    origin = grid.shape[1]//2
    return( [ ((row[origin - 1] + row[origin] + row[origin + 1]) / 3) for row in grid ] )



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

    # range of probabilities from 0 to 1 for p
    for p in range(slices):
        c_list.append(aggregate(t_max, r_max, iters, round((p+1)/slices,2), q)[t_final-1,origin])

    return(c_list)



def smoothed_c_list(c_list):
    sc_list = [ (c_list[i] + c_list[i-1]) / 2 for i in range(1, len(c_list)) ]
    sc_list.insert(0, c_list[0])
    return(sc_list)



#####################################
# Plots
#####################################

def map_grid(grid, iters, p, q):

    c = plt.pcolormesh(grid, cmap='Greys', norm=mpl.colors.LogNorm())   # can also get rid of log norm maybe
    plt.colorbar(c)
    plt.title(f'CA Averaged over {iters} Runs (p = {p}, q = {q})')
    plt.savefig(f'CA_{iters}_{len(grid)}t_{int(p*100)}p_{int(q*100)}q.png')
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



def plot_clist(c_list, iters, slices, lateness, q, weighted=False):
    plt.plot( [round((p+1)/slices,2) for p in range(slices)], c_list )
    plt.xlabel('P(Xi = 1)')

    if weighted:
        plt.ylabel(f'Average of Origin Values at {lateness}t_max, {lateness}t_max - 1')
        plt.title(f'Mean of Origin Values at {lateness}t_max, Mean over {iters} Runs, q = {q}')
        plt.savefig(f'LateOriginAvgW_{iters}_{len(c_list)}r_{lateness}t_{int(q*100)}q.png')

    else:
        plt.ylabel('Average Origin Value')
        plt.title(f'Origin Value at {lateness}t_max, Mean over {iters} Runs, q = {q}')
        plt.savefig(f'LateOriginAvg_{iters}_{len(c_list)}r_{lateness}t_{int(q*100)}q.png')

    #plt.show()
    plt.clf()



#####################################
# Main
#####################################

def main():

    args = sys.argv[1:]
    for i in range(len(args)):
        if args[i] == '-h':
            usage()

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', action="store", dest="t_steps", default=1000, type=int)
    parser.add_argument('-r',action="store", dest="r_steps", default=2000, type=int)
    parser.add_argument('-p', action="store", dest="P", default=0.5, type=float)
    parser.add_argument('-q', action="store", dest="Q", default=0.5, type=float)
    parser.add_argument('-i', action="store", dest="iterations", default=1, type=int)
    parser.add_argument('-s', action="store", dest="slices", default=20, type=int)
    parser.add_argument('-l', action="store", dest="lateness", default=1, type=float)
    parser.add_argument('-c', action='store_true', default=False)

    res = parser.parse_args()

    start_time = time.time()
    grid = aggregate(res.t_steps, res.r_steps, res.iterations, res.P, res.Q)
    #map_grid(grid, res.iterations, res.P, res.Q)

    clist = late_central_avg(res.t_steps, res.r_steps, res.iterations, res.slices, res.lateness, res.Q)
    sclist = smoothed_c_list(clist)
    plot_clist(clist, res.iterations, res.slices, res.lateness, res.Q)
    plot_clist(sclist, res.iterations, res.slices, res.lateness, res.Q, weighted=True)

    end_time = time.time()

    print(f"Runtime: {end_time - start_time}")


if __name__ == "__main__":
    main()



