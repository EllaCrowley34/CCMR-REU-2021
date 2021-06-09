""" walker.py

    Uses MC method to generate random walk in 3D.
    Plots

    Language: python 3

    Ella Crowley
    Cornell University

    Written for CCMR REU, Summer 2021.
    6/8/21 on Mac.

"""
from mpl_toolkits.mplot3d import Axes3D

import math
import random   #random.random()
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats     # might not need this
import matplotlib.mlab as mlab
import pandas as pd

def rand_yield(dim, n):
    """
        parameters:
            dim (int):  length of matrix elements
            n (int):    number of matrix elements

        Generator object. Yields matrix elements to increase efficiency
        for large values of n by enabling task parallelism.
    """
    yield from np.random.random((n,dim))

def walk(dim, steps, probs):
    """
        parameters:
            dim (int):  dimensions in walk
            steps (int):steps in walk
            reps (int): method repititions

        Performs a 'steps'-step random walk in 'dim' dimensions with
        equal probability of travel in any direction (unless bias is specified).
        Add bias by changing values in 'probs' array. Default if bias is 'True' is 0.7 prob of
        positive travel in xyz directions.

        Returns the average of 'reps' repitions of the walk.
    """
    pos = [[0] * dim]
    for step in rand_yield(dim, steps):
        pos.append([(pos[-1][i] + (step[i] > probs[i][0]) - (step[i] < probs[i][1])) for i in range(dim)])

    return(pos)



def get_dist(pos_list):
    """
        Calculates distances from origin of a list of
        n-dimensional coordinates

    """
    return([ np.sqrt(np.sum(list(map(lambda m: m**2, p)))) for p in pos_list ])



def get_rms(pos_list):
    """
        RMS distance from origin based on distance list
    """
    sq_dist_list = np.array(get_dist(pos_list))**2      # all distances squared
    rms_list = [ sq_dist_list[0] ]

    for i in range(1, len(sq_dist_list)):
        rms_list.append( (rms_list[i-1] + sq_dist_list[i]) )

    for i in range(1, len(sq_dist_list)):
        rms_list[i] /= (i + 1)

    return(np.sqrt(rms_list))



def plot_walk(arr, runs, bias="Unbiased"):
    """
        parameters:
            arr: 3D numpy array

        Creates 3D scatter plot of locations visited in array
    """
    data = pd.DataFrame(arr, columns=['x','y','z'])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(data['x'], data['y'], data['z'],'green', linewidth = 0.5)
    ax.set_title(f"{bias} 3D Random Walk: {len(arr) - 1} Steps, {runs} Runs")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()



def plot_rms(rms_list, runs, bias="Unbiased"):
    """
        parameters:
            rms_list: list of rms values at each step in walk

        Plots in 1D the walker's distance from origin over time.
    """
    plt.plot(rms_list)
    plt.title(f"{bias} Distance from Origin in 3D Random Walk: {len(rms_list) - 1} Steps, {runs} Runs")
    plt.xlabel('Steps')
    plt.ylabel('Distance from Origin')
    plt.show()



def sim(dim, steps, runs, bias=False):

    probs = np.full((dim,2) , 0.5)        # Equal probability in every direction

    if bias:
        probs = [[0.7, 0.3], [0.7, 0.3], [0.7, 0.3]]

    rms_list = np.zeros(steps + 1)

    for run in range(runs):
        rms_list += get_rms(walk(dim, steps, probs))

    rms_list /= runs

    return(rms_list)


####################################
# Main
####################################

DIMENSIONS = 3
RUNS = 1000
STEPS = 100
bias = True

#pos, rms = walk(DIMENSIONS, STEPS, RUNS)
#plot_walk(pos)
#plot_rms(rms)

rms = sim(DIMENSIONS, STEPS, RUNS, bias=bias)
print(rms)
plot_rms(rms, RUNS, bias="Biased")



