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

def sample_point(n):
    pp = {}
    for p in range(len(param_names)):
        pp[param_names[p]] = theta[n][p]
    return pp
    
opt = parse_inp()

N_models = int(opt['N_models_to_sample'][0])
wave = [float(x) for x in opt['wavelength']]
GSSP_run_cmd = opt['GSSP_run_cmd'][0]

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

# --- Specify number of free parameters and total grid points -
N_param = len(grid)

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
T_eff_intrp = interp1d(intrp_range, grid['T_eff'][:2])
logg_intrp = interp1d(intrp_range, grid['log(g)'][:2])
vsini_intrp = interp1d(intrp_range, grid['v*sin(i)'][:2])
v_micro_intrp = interp1d(intrp_range, grid['v_micro'][:2])
MH_intrp = interp1d(intrp_range, grid['[M/H]'][:2])

# --- Create final quasi-random sampled grid -----------------
# We make a new matrix theta, where each column corresponds 
# to one of the free parameters, whose ranges have been mapped
# onto the Sobol numbers

theta = np.array(
    [T_eff_intrp(N_sobol[:, 0]).T, logg_intrp(N_sobol[:, 1]).T,
     vsini_intrp(N_sobol[:, 2]).T, v_micro_intrp(N_sobol[:, 3]).T, MH_intrp(N_sobol[:, 4]).T]).T

print()
print('------ N_grid x N_param quasi-random grid -------')
print(theta)
print()


for i in range(N_models):
    pp = sample_point(i)
    pp_list = []
    for p in param_names: pp_list.append(pp[p])
    
    print('-'*25)
    print('Sampled point:', pp)
    print('Current subgrid:')
    subgrid = {}
    for p in param_names:
        step = grid[p][2]
        start = pp[p] - pp[p]%step
        subgrid[p] = [start, start + step, step]
        print(p, subgrid[p])
    
    continue
    run_GSSP_grid('subgrid.inp', subgrid, wave, GSSP_run_cmd)

    GRID = Grid('rgs_files')
    GRID.load()

    RND = RandomGrid(GRID)

    prefix = '%.0f'%(time.time()*1.e6)
    fn = prefix + '.npz'
    sp = RND.interpolate(np.array(pp_list))
    np.savez(os.path.join(rnd_grid_dir, fn), flux=sp, labels=pp)




print('Done.')

