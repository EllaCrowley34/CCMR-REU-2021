from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing as mp
import scipy.sparse
from scipy.optimize import curve_fit
from multiprocessing import Pool
from functools import partial
from contextlib import contextmanager
import os
import sys

path = 'Data/Percolation/'
np.set_printoptions(threshold=sys.maxsize)

#Helper functions
def shift(offset,arr):
    return np.concatenate((arr[-offset:],arr[:-offset]))

def save_sparse(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse(filepath):
    # here we need to add .npz extension manually
    loader = np.load(filepath)
    state_sparse = scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    return state_sparse.toarray()

def file_reader(p, t):
    state = np.loadtxt(path + 't=' + str(t) + '_p=' + str(p) + '.txt')
    print('Reading done')
    print(np.sum(state))

def exponential_decay(x,a,l):
    return a*np.exp(-np.abs(x)/l)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def initial_state(steps, L):
    state = np.zeros((steps, L))
    state[0, int(L/2)] = 1
    return state

def initial_prop_matrix(p, period, L):
    # period == tau (periodicity of floquet system)
    propagation_matrix = np.random.RandomState().binomial(1, p, (period, L))
    return propagation_matrix

def initial_death_matrix(t, period, L):
    death_matrix = np.random.RandomState().binomial(1, t, (period, L))
    return death_matrix

def update(state, timestep, p, t):
    # timestep is the row we want to update ie t+1
    L = len(state[0])

    propagation_list = np.random.RandomState().binomial(1, p, L)
    death_list = np.random.RandomState().binomial(1, t, L)

    length_term_left, length_term_right = shift(1,state[timestep-1]), shift(-1,state[timestep-1])
    length_term = length_term_left + length_term_right - length_term_left*length_term_right

    propagation_term = propagation_list*(1 - state[timestep-1])*length_term
    death_term = -1*death_list*state[timestep-1]

    state[timestep] = state[timestep-1] + propagation_term + death_term
    return state

def update_indp(state, timestep, propagation_list, t):
    # timestep is the row we want to update ie t+1
    L = len(state[0])

    death_list = np.random.RandomState().binomial(1, t, L)

    length_term_left, length_term_right = shift(1,state[timestep-1]), shift(-1,state[timestep-1])
    length_term = length_term_left + length_term_right - length_term_left*length_term_right

    propagation_term = propagation_list*(1 - state[timestep-1])*length_term
    death_term = -1*death_list*state[timestep-1]

    state[timestep] = state[timestep-1] + propagation_term + death_term
    return state

def update_indt(state, timestep, p, death_list):
    # timestep is the row we want to update ie t+1
    L = len(state[0])

    propagation_list = np.random.RandomState().binomial(1, p, L)

    length_term_left, length_term_right = shift(1,state[timestep-1]), shift(-1,state[timestep-1])
    length_term = length_term_left + length_term_right - length_term_left*length_term_right

    propagation_term = propagation_list*(1 - state[timestep-1])*length_term
    death_term = -1*death_list*state[timestep-1]

    state[timestep] = state[timestep-1] + propagation_term + death_term
    return state

def update_floquet(state, timestep, p_matrix, t_matrix):
    L = len(state[0])
    period = len(p_matrix)

    length_term_left, length_term_right = shift(1,state[timestep-1]), shift(-1,state[timestep-1])
    length_term = length_term_left + length_term_right - length_term_left*length_term_right

    propagation_term = p_matrix[timestep % period]*(1 - state[timestep-1])*length_term
    death_term = -1*t_matrix[timestep % period]*state[timestep-1]

    state[timestep] = state[timestep-1] + propagation_term + death_term
    return state


def evolution(steps, L, p, t, run_number):
    print('World: ', run_number)
    sys.stdout.flush()
    state = initial_state(steps, L)

    for i in range(1,steps):
        state = update(state, i, p, t)

    return state

def evolution_indp(steps, L, p, t, run_number, g=1):
    p_list = np.random.RandomState().binomial(1, p, L)

    print('World: ', run_number)
    sys.stdout.flush()
    state = initial_state(steps, L)

    for i in range(1,steps):
        if np.random.rand() < g: # else p_list stays same
            p_list = np.random.RandomState().binomial(1, p, L)
        state = update_indp(state, i, p_list, t)

    return state

def evolution_indt(steps, L, p, t, run_number):
    t_list = np.random.RandomState().binomial(1, t, L)

    print('World: ', run_number)
    sys.stdout.flush()
    state = initial_state(steps, L)

    for i in range(1,steps):
        state = update_indt(state, i, p, t_list)

    return state

def evolution_floquet(steps, L, p, t, period, run_number):
    print('World: ', run_number)
    sys.stdout.flush()

    p_matrix = initial_prop_matrix(p, period, L)
    t_matrix = initial_death_matrix(t, period, L)
    state = initial_state(steps, L)

    for i in range(1,steps):
        state = update_floquet(state, i, p_matrix, t_matrix)

    return state

def simulation_parallel(steps, L, p, t, runs, indp=False, indt=False, g=1.0, floquet=False, period=10):
    state_database = np.zeros((steps,L))

    batch_size = mp.cpu_count()
    for i in range(int(runs/batch_size)):
        arguments_list = [[] for a in range(batch_size)]
        for j in range(batch_size):
            arguments_list[j] = i*batch_size + j

        with poolcontext(processes=batch_size) as pool:
            if indp:
                results = pool.map(partial(evolution_indp, steps, L, p, t, g=g), arguments_list)
            elif indt:
                results = pool.map(partial(evolution_indt, steps, L, p, t), arguments_list)
            elif floquet:
                results = pool.map(partial(evolution_floquet, steps, L, p, t, period), arguments_list)
            else:
                results = pool.map(partial(evolution, steps, L, p, t), arguments_list)

        for j in range(batch_size):
            state_database += results[j]

    #Average
    runs = int(runs/batch_size)*batch_size
    state_database = state_database/runs

    #Data Storage
    if indp:
        filename = 'indp_g=' + str(g) + '_t=' + str(t) + '_p=' + str(p)
    elif floquet:
        filename = f'floquet_tau={period}_t={t}_p={p}'
    else:
        filename = 't=' + str(t) + '_p=' + str(p)
    state_sparse = scipy.sparse.csr_matrix(state_database)
    save_sparse(path + filename, state_sparse)
    return None


#ANALYSIS
def analyse_saturation(t):
    p_list, equilibrium_list = [], []
    file_name_check = 't=' + str(t)

    for file in os.listdir(path + 'Data/'):
        if file_name_check in file:
            file_index_start, file_index_end = file.index('p') + 2, file.index('n') - 1
            p = float(file[file_index_start:file_index_end])
            p_list.append(p)

            print('Loading... t=' + str(t) + ' p=' + str(p))
            file_name = 't=' + str(t) + '_p=' + str(p)

            # replace load_sparse with np.genfromtext()
            state_database = load_sparse(path + 'Data/' + file_name)

            origin = int(len(state_database[0]) / 2)
            equilibrium_value = np.mean(state_database[-10:, origin - 10 : origin + 10])
            equilibrium_list.append(equilibrium_value)

    # Save to file
    file_name = 'Equilibrium_t=' + str(t)
    data = [[] for i in range(2)]
    data[0], data[1] = p_list, equilibrium_list
    np.savetxt(path + file_name + '.txt', data)
    return None


def analyse_correlation(t):
    p_list, correlation_list = [], []
    file_name_check = 't=' + str(t)

    for file in os.listdir(path + 'Data/'):
        if file_name_check in file:
            file_index_start, file_index_end = file.index('p') + 2, file.index('n') - 1
            p = float(file[file_index_start:file_index_end])
            p_list.append(p)

            print('Loading... t=' + str(t) + ' p=' + str(p))
            file_name = 't=' + str(t) + '_p=' + str(p)
            state_database = load_sparse(path + 'Data/' + file_name)

            steps, L = len(state_database), len(state_database[0])
            shift_sites, patch_size = 50, 1000

            equilibrium_value = np.mean(state_database[-10:, int(len(state_database[0]) / 2) - 10:int(len(state_database[0]) / 2) + 10])
            patch = state_database[steps - patch_size:, int(L / 2 - patch_size / 2):int(L / 2 + patch_size / 2)]

            if equilibrium_value != 0: #compute correlations when there is a light cone
                correlation, position = np.zeros(shift_sites), np.zeros(shift_sites)

                for i in range(shift_sites):
                    patch_shifted = shift(i, state_database)[steps - patch_size:, int(L/2 - patch_size/2):int(L/2 + patch_size/2)]

                    correlation[i] = np.mean(np.multiply(patch, patch_shifted))
                    position[i] = i

                if correlation[1] > correlation[-1]: #not oscillating "fringes"
                    print('No Oscillations')
                    correlation_list.append(0)

                else:
                    correlation_offset = correlation - (correlation[-1] + correlation[-2])/2
                    popt, pcov = curve_fit(exponential_decay, position, np.abs(correlation_offset))
                    correlation_list.append(popt[1]) #the correlation length
            else: #no light cone
                print('No light cone')
                correlation_list.append(0)

    # Save to file
    file_name = 'Correlation_t=' + str(t)
    data = [[] for i in range(2)]
    data[0], data[1] = p_list, correlation_list
    np.savetxt(path + file_name + '.txt', data)
    return None

def analyse_velocity(t):
    #the butterfly velocity is defined when the value of the otoc is 1/2 of the bulk value
    #error bars represent the tolerance of 10% within 50%

    #first pass to obtain values of p
    p_list = []
    file_name_check = 't=' + str(t)

    for file in os.listdir(path + 'Data/'):
        if file_name_check in file:
            file_index_start, file_index_end = file.index('p') + 2, file.index('n') - 1
            p = float(file[file_index_start:file_index_end])
            p_list.append(p)
    p_list = np.array(p_list)

    velocity_butterfly_list, velocity_error_list = np.zeros(len(p_list)), np.zeros(len(p_list))

    for i in range(len(p_list)):
        print('Computing Velocity for p= ', p_list[i])

        file_name = 't=' + str(t) + '_p=' + str(p_list[i])
        state_database = load_sparse(path + 'Data/' + file_name)
        steps, L = len(state_database), len(state_database[0])

        origin = int(L/2)
        equilibrium_value = np.mean(state_database[-10: , origin - 10 : origin + 10])

        if equilibrium_value == 0.0: #no light cone will have no butterfly velocity
            print('No light cone')
            velocity_butterfly_list[i], velocity_error_list[i] = 0, 0
        else:
            #Find first left and right non zero values at last timestep and corresponding velocities
            state_nonzeros = np.nonzero(state_database[-1])[0]
            left_front, right_front = state_nonzeros[0], state_nonzeros[-1]
            start_front = np.nonzero(state_database[0])[0][0]
            vel_left_front, vel_right_front = (left_front - start_front)/steps, (right_front - start_front)/steps

            #Find indices whereby saturation has happened and corresponding velocities
            saturated_indices = np.where(state_database[-1] > equilibrium_value)[0]
            left_saturated, right_saturated = saturated_indices[0], saturated_indices[-1]
            vel_left_saturated, vel_right_saturated = (left_saturated - start_front)/steps, (right_saturated - start_front)/steps

            #track of time and space indices along const velocities
            indices = np.zeros((steps, 2), dtype=int)
            indices[:,0] = np.arange(steps)

            discretizations = 100

            #left butterfly_velocity
            first_instance = True
            vel_butterfly_left_lower, vel_butterfly_left_upper = 0, 0
            for j in range(discretizations):
                v_left = vel_left_front + j*(vel_left_saturated - vel_left_front)/discretizations

                indices[:,1] = np.round(start_front - v_left*indices[:,0])
                state_data = state_database[tuple(indices.T)]

                #take mean of late time values
                state_mean = np.mean(state_data[-int(steps*0.9):])

                if np.abs(state_mean/equilibrium_value - 0.5) < 0.1 and first_instance:
                    vel_butterfly_left_lower = v_left
                    first_instance = False
                if (state_mean/equilibrium_value - 0.5) > 0.1 and not first_instance:
                    #this means we have already found the lower bound and this is the upper bound
                    vel_butterfly_left_upper = v_left
                    break

            #right butterfly_velocity
            first_instance = True
            vel_butterfly_right_lower, vel_butterfly_right_upper = 0, 0
            for j in range(discretizations):
                v_right = vel_right_front + j*(vel_right_saturated - vel_right_front)/discretizations

                indices[:,1] = np.round(start_front - v_right*indices[:,0])
                state_data = state_database[tuple(indices.T)]

                #take mean of late time values
                state_mean = np.mean(state_data[-int(steps*0.9):])

                if np.abs(state_mean/equilibrium_value - 0.5) < 0.1 and first_instance:
                    vel_butterfly_right_lower = v_right
                    first_instance = False
                if (state_mean/equilibrium_value - 0.5) > 0.1 and not first_instance:
                    #this means we have already found the lower bound and this is the upper bound
                    vel_butterfly_right_upper = v_right
                    break

            velocity_average = np.mean(np.abs([vel_butterfly_left_lower,vel_butterfly_left_upper,vel_butterfly_right_lower,vel_butterfly_right_upper]))
            velocity_error = np.mean(np.abs([vel_butterfly_left_lower-vel_butterfly_left_upper,vel_butterfly_right_lower-vel_butterfly_right_upper]))

            velocity_butterfly_list[i], velocity_error_list[i] = velocity_average, velocity_error

    #Save to file
    file_name = 'Velocity_t=' + str(t)
    data = [[] for i in range(3)]
    data[0], data[1], data[2] = p_list, velocity_butterfly_list, velocity_error_list
    np.savetxt(path + file_name + '.txt', data)

def analyse_broadening(t):
    # first pass to obtain values of p
    p_list = []
    file_name_check = 't=' + str(t)

    for file in os.listdir(path + 'Data/'):
        if file_name_check in file:
            file_index_start, file_index_end = file.index('p') + 2, file.index('n') - 1
            p = float(file[file_index_start:file_index_end])
            p_list.append(p)
    p_list = np.array(p_list)

    diffusion_list, diffusion_error_list = np.zeros(len(p_list)), np.zeros(len(p_list))

    for i in range(len(p_list)):
        print('Computing Width for p= ', p_list[i])

        file_name = 't=' + str(t) + '_p=' + str(p_list[i])
        state_database = load_sparse(path + 'Data/' + file_name)
        steps, L = len(state_database), len(state_database)

        #lower bound is 0.2 and upper bound is 0.8 of saturation value
        left_lower, left_upper, right_lower, right_upper = np.zeros(steps), np.zeros(steps), np.zeros(steps), np.zeros(steps)

        origin = int(len(state_database[0])/2)
        equilibrium_value = np.mean(state_database[-10:, origin - 10 : origin + 10])

        if equilibrium_value == 0.0: #ie we have a light cone
            print('No light cone')
            diffusion_list[i], diffusion_error_list[i] = 0, 0

        else:
            for j in range(steps):
                state_lower = np.where(state_database[j] > 0.2*equilibrium_value)[0]
                left_lower[j], right_lower[j] = state_lower[0], state_lower[-1]

                state_upper = np.where(state_database[j] > 0.8*equilibrium_value)[0]
                left_upper[j], right_upper[j] = state_upper[0], state_upper[-1]

            width_data = np.zeros(steps)
            for j in range(steps):
                width_data[j] = np.mean(np.abs([left_upper[j]-left_lower[j], right_upper[j]-right_lower[j]]))

            diffusion_coeff = np.zeros(steps)
            for j in range(1,steps):
                diffusion_coeff[j] = (width_data[j]/(j**(1/2)))**2

            #estimate diffusion_coefficient from 90%-100% value and error from comparing with 80%-90%
            diffusion_list[i] = np.mean(diffusion_coeff[int(0.5*steps):])
            diffusion_error_list[i] = np.abs(np.mean(diffusion_coeff[int(0.4*steps):int(0.9*steps)]) - diffusion_list[i])

    #Save to file
    file_name = 'Diffusion_t=' + str(t)
    data = [[] for i in range(3)]
    data[0], data[1], data[2] = p_list, diffusion_list, diffusion_error_list
    np.savetxt(path + file_name + '.txt', data)

def analyse_front(p, t, indp=False, g=1.0, floquet=False, tau=10):

    print(f'Finding front for p={p}, t={t}')

    if indp:
        file_name = f'indp_g={g}_t={t}_p={p}.npz'
    elif floquet:
        file_name = f'floquet_tau={tau}_t={t}_p={p}.npz'
    else:
        file_name = f't={t}_p={p}.npz'

    if file_name in os.listdir(path):
        state_database = load_sparse(path + '/' + file_name)
    else:
        print(f'{file_name} not found in {path}')
        return None

    steps, L = len(state_database), len(state_database[0])
    origin = int(L/2)
    bulk_equilibrium = np.mean(state_database[-10:, (origin-10):(origin+10)])
    front = bulk_equilibrium / 2


    above_front = np.where(state_database < front, 0, state_database)

    try:
        front_indices_list = np.array([(np.nonzero(line)[0][0], np.nonzero(line)[0][-1]) for line in above_front])

        if indp:
            file_name = f'indp_front_g={g}_t={t}_p={p}'
        elif floquet:
            file_name = f'floquet_front={tau}_t={t}_p={p}'
        else:
            file_name = f'front_t={t}_p={p}'

        np.save(path + '/Fronts/' + file_name, front_indices_list)
        # Automatically adds '.npy' extension

    except IndexError:
        print(f"No front found for p={p}, t={t}")
        return None



# IN PROGRESS
def analyse_g_variation(t=0.9):
    g_list = np.linspace(0,1,11)
    p_list = np.linspace(0,1,11)
    fg_file = 0

    for p in p_list:
        for g in g_list:
            for file in os.listdir(path):
                if file == 'indp_g=' + str(g) + '_t=' + str(t) + '_p=' + str(p) + '.npz':
                    fg_file = file
                    break

            state_grid = load_sparse(path + '/' + file)
            # want to average over the position of the front in the last 10 rows


# Plots
def map_evolution(p, t, runs, indp=False, g=1.0, floquet=False, tau=10):
    if indp:
        file_name = f'indp_g={g}_t={t}_p={p}.npz'
    elif floquet:
        file_name = f'floquet_tau={tau}_t={t}_p={p}.npz'
    else:
        file_name = f't={t}_p={p}.npz'

    grid = load_sparse(path + '/' + file_name)
    cmap = plt.get_cmap('Greys')
    c = plt.pcolormesh(grid, cmap=cmap, vmin=0)
    plt.colorbar(c)

    if indp:
        plt.title(f'CA, {runs} Runs (g={g}, t={t}, p={p})')
        #plt.savefig(f'Graphs/CA_p={p}_t={t}_g={g}.png')
    elif floquet:
        plt.title(f'CA, {runs} Runs, (T={tau}, t={t}, p={p}')
    else:
        plt.title(f'CA, {runs} Runs, (t={t}, p={p})')
        #plt.savefig(f'Graphs/CA_p={p}_t={t}.png')

    #plt.show()
    #plt.clf()

def map_front_evolution(grid, runs, p, t, indp, g, floquet, tau):
    if indp:
        front_indices_list = np.load(f'{path}/Fronts/indp_front_g={g}_t={t}_p={p}.npy')
    elif floquet:
        front_indices_list = np.load(f'{path}/Fronts/floquet_front={tau}_t={t}_p={p}.npy')
    else:
        front_indices_list = np.load(f'{path}/Fronts/front_t={t}_p={p}.npy')

    # must be for evolution with 1000 steps
    for i in range(1000):
        a, b = front_indices_list[i]
        grid[i][a] = grid[i][b] = -1

    c = plt.pcolormesh(grid, cmap='Greys', vmin=0, vmax=1)
    c.cmap.set_under('r')
    plt.colorbar(c, extend='min')

    if indp:
        plt.title(f'CA with front, {runs} runs (g={g}, t={t}, p={p})')
        plt.savefig(f'Graphs/frontCA_indp_{p}p_{t}t_{g}g.png')
    elif floquet:
        plt.title(f'Floquet system with front, {runs} runs (T={tau}, t={t}, p={p})')
        plt.savefig(f'Graphs/frontCA_floquet_{p}p_{t}t_{tau}T.png')
    else:
        plt.title(f'CA with front, {runs} runs (t={t}, p={p})')
        plt.savefig(f'Graphs/frontCA_{p}p_{t}t.png')

    plt.show()
    plt.clf()

def map_floquet_evolution(grid, runs, p, t, period):
    c = plt.pcolormesh(grid, cmap='Greys')
    plt.colorbar(c)
    plt.title(f"CA Floquet System, {runs} runs (p={p}, t={t}, T={period})")
    plt.savefig(f'Graphs/floquet_{p}p_{t}t_{period}tau.png')
    #plt.show()
    plt.clf()

def map_both(runs, p, t, g_list, indp_only=False, front_only=False):

    for file in os.listdir(path):

        if not indp_only:
            if f"front_database_p={p}_t={t}_g=" in file:
                g = file[file.index('g=') + 2 : file.index('.npz')]
                if g in map(lambda x: str(x), g_list):
                    front_grid = load_sparse(path + '/' + file)
                    print(f"Mapping front ... {file}")
                    map_front_evolution(front_grid, runs, p, t, g)
        if not front_only:
            if "indp" in file and f"t={t}_p={p}" in file:
                g = file[file.index('g=') + 2 : file.index('_t=')]
                print(f"g = {g} for file: {file}")
                if g in map(lambda x: str(x), g_list):
                    grid = load_sparse(path + '/' + file)
                    print(f"Mapping grid ... {file}")
                    map_evolution(grid, runs, p, t, g)

def map_floquet(runs, p, t, period):
    for file in os.listdir(path):
        if f'floquet_tau={period}_t={t}_p={p}' in file:
            grid = load_sparse(path + '/' + file)
            print(f"Mapping grid ... {file}")
            map_floquet_evolution(grid, runs, p, t, period)

def map_front(p, t, runs, indp=False, g=1.0, floquet=False, tau=10):

    if indp:
        file_name = f'indp_g={g}_t={t}_p={p}.npz'
    elif floquet:
        file_name = f'floquet_tau={tau}_t={t}_p={p}.npz'
    else:
        file_name = f't={t}_p={p}.npz'

    if file_name in os.listdir(path):
        grid = load_sparse(path + '/' + file_name)

    else:
        print(f'{file_name} not found in {path}')
        return None

    map_front_evolution(grid, runs, p, t, indp, g, floquet, tau)

def graph_front(p, t, runs, indp=False, g=1.0, floquet=False, tau=10, xlow=0, xhigh=2000):

    if indp:
        front_indices_list = np.load(f'{path}/Fronts/indp_front_g={g}_t={t}_p={p}.npy')
    elif floquet:
        front_indices_list = np.load(f'{path}/Fronts/floquet_front={tau}_t={t}_p={p}.npy')
    else:
        front_indices_list = np.load(f'{path}/Fronts/front_t={t}_p={p}.npy')

    # flatten list
    front_indices_list_L = [a for a,b in front_indices_list]
    front_indices_list_R = [b for a,b in front_indices_list]
    x_vals = np.arange(1000)

    plt.plot(front_indices_list_L, x_vals, color='red', linewidth=0.4)
    plt.plot(front_indices_list_R, x_vals, color='red', linewidth=0.4)
    plt.xlim(xlow,xhigh)
    plt.ylim(0,1000)
    if indp:
        plt.title(f'CA with front, {runs} runs (g={g}, t={t}, p={p})')
        plt.savefig(f'Graphs/frontCA_indp_{p}p_{t}t_{g}g.png')
    elif floquet:
        plt.title(f'Floquet system with front, {runs} runs (T={tau}, t={t}, p={p})')
        plt.savefig(f'Graphs/frontCA_floquet_{p}p_{t}t_{tau}T.png')
    else:
        plt.title(f'CA with front, {runs} runs (t={t}, p={p})')
        plt.savefig(f'Graphs/frontCA_{p}p_{t}t.png')
    plt.show()



# Main

P = 0.9
T = 20
RUNS = 100

p_list = [0.4,0.5,0.6]
t_list = [0.2]

def main_computation():


    for t in t_list:
       for p in p_list:
            print(f"Run: t = {round(t,2)}, p = {round(p,2)}, T = {T}")
            simulation_parallel(1000, 2000, round(p,2), round(t,2), RUNS, floquet=True, period=T)

    #for period in range(11,14):
    #simulation_parallel(1000, 2000, round(P,2), round(T,2), RUNS, floquet=True, period=period)

def main_analysis():
    #for period in range(11,14):
    #    analyse_front(P, T, floquet=True, tau=period)
    for t in t_list:
        for p in p_list:
            analyse_front(p,t, floquet=True, tau=T)

if __name__ == "__main__":

    main_computation()
    main_analysis()

    #for period in range(11,14):
     #   map_front(P, T, RUNS, floquet=True, tau=period)
    for t in t_list:
        for p in p_list:
            map_evolution(p,t,RUNS, floquet=True, tau=T)
            graph_front(p,t,RUNS, floquet=True, tau=T)
