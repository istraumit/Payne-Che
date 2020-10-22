import os, time
import numpy as np
from Partition import *
from run_GSSP import *
from Grid import *
from RandomGrid import *
from Subgrid import *
from random_grid_common import *
import sobol_seq
from scipy.interpolate import interp1d

opt = parse_inp()

N_models = int(opt['N_models_to_sample'][0])
N_models_skip = int(opt['N_models_to_skip'][0])
wave = [float(x) for x in opt['wavelength']]
GSSP_run_cmd = opt['GSSP_run_cmd'][0]

Kurucz = True
if 'Kurucz' in opt:
    Kurucz = opt['Kurucz'][0].lower() in ['true', 'yes', '1']

rnd_grid_dir = opt['output_dir'][0]
if not os.path.exists(rnd_grid_dir):
    os.makedirs(rnd_grid_dir)

grid = {}
for o in opt:
    if o in param_names:
        grid[o] = [float(x) for x in opt[o]]

###############################################################
#       Example of how to generate a quasi-random grid
###############################################################

grid_params = [p for p in param_names if grid[p][0]!=grid[p][1]]

# --- Specify number of free parameters and total grid points -
N_param = len(grid_params)

# --- Calculate Sobol numbers for random sampling -------------
# This creates a N_grid x N_param matrix of quasi-random numbers
# between 0 and 1 using Sobol numbers
N_sobol = sobol_seq.i4_sobol_generate(N_param, N_models)
print()
print('------ N_grid x N_param matrix of Sobol numbers -------')
print(N_sobol)
print()

# --- Prepare linear interpolation functions for mapping ------
# One such function is needed for every free paraneter, i.e. 
# a total of N_param functions

# Range of the Sobol numbers
intrp_range = [0, 1]       

# 1D linear interpolation functions
intrp = [interp1d(intrp_range, grid[p][:2]) for p in grid_params]

# --- Create final quasi-random sampled grid -----------------
# We make a new matrix theta, where each column corresponds 
# to one of the free parameters, whose ranges have been mapped
# onto the Sobol numbers

columns = [v(N_sobol[:,i]) for i,v in enumerate(intrp)]
theta = np.vstack(columns).T
print()
print('------ N_grid x N_param quasi-random grid -------')
print(theta)
print()

np.savetxt('theta.data', theta)

for i in range(N_models):
    if i < N_models_skip: continue
    pp_arr = theta[i,:]
    pp = {}
    for j,v in enumerate(grid_params):
        pp[v] = pp_arr[j]
    
    print('-'*25)
    print('Sampled point:', pp)
    print('Current subgrid (' + str(i) + '):')
    subgrid = {}
    for p in param_names:
        if p in grid_params:
            step = grid[p][2]
            start = pp[p] - pp[p]%step
            subgrid[p] = [start, start + step, step]
        else:
            subgrid[p] = grid[p]
        print(p, subgrid[p])
    
    ok = run_GSSP_grid('subgrid.inp', subgrid, wave, GSSP_run_cmd, Kurucz=Kurucz)
    if not ok:
        print('GSSP exited with error')
        exit()

    GRID = Grid('rgs_files')
    GRID.load()

    RND = RandomGrid(GRID)

    prefix = str(i).zfill(6)
    fn = prefix + '.npz'
    sp = RND.interpolate(pp_arr)
    np.savez(os.path.join(rnd_grid_dir, fn), flux=sp, labels=pp)




print('Done.')

