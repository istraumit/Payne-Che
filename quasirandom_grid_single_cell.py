import sys, shutil
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
from multiprocessing import Process
from multiprocessing.pool import Pool
from shutil import copyfile

opt_fn = sys.argv[1]
opt = parse_inp(opt_fn)

N_models = int(opt['N_models_to_sample'][0])
N_models_skip = int(opt['N_models_to_skip'][0])
wave = [float(x) for x in opt['wavelength']]
GSSP_run_cmd = opt['GSSP_run_cmd'][0]
GSSP_data_path = opt['GSSP_data_path'][0]
N_instances = int(opt['N_instances'][0])
N_interpol_threads = int(opt['N_interpol_threads'][0])
scratch_dir = opt['scratch_dir'][0]

Kurucz = True
if 'Kurucz' in opt:
    Kurucz = opt['Kurucz'][0].lower() in ['true', 'yes', '1']

rnd_grid_dir = opt['output_dir'][0]
subgrid_dir = 'subgrids'
for dn in [rnd_grid_dir, subgrid_dir]:
    if not os.path.exists(dn):
        os.makedirs(dn)

copyfile(opt_fn, os.path.join(rnd_grid_dir, '_grid.conf'))

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

work = []

for i in range(N_models):
    if i < N_models_skip: continue
    pp_arr = theta[i,:]
    pp = {}
    for j,v in enumerate(grid_params):
        pp[v] = pp_arr[j]
    
    subgrid = {}
    for p in param_names:
        if len(GSSP_steps[p]) == 1:
            step = GSSP_steps[p][0]
        else:
            step = GSSP_steps[p][1] if pp[p]<GSSP_steps[p][0] else GSSP_steps[p][2]
            
        if p in grid_params:
            start = pp[p] - pp[p]%step
            subgrid[p] = [start, start + step, step]
        else:
            subgrid[p] = grid[p] + [step]

    work_item = (str(i).zfill(6), subgrid, pp, pp_arr)
    work.append(work_item)

    

def run_one_item(item):

    (run_id, subgrid, pp, pp_arr) = item
    inp_fn = os.path.join(subgrid_dir, 'subgrid_' + run_id + '.inp')

    ok = run_GSSP_grid(run_id, inp_fn, subgrid, wave, GSSP_run_cmd, GSSP_data_path, scratch_dir, opt['R'][0], Kurucz=Kurucz)
    if not ok:
        print('GSSP exited with error, item id '+run_id)
        return 1

    rgs_dir = os.path.join('rgs_files', run_id)
    GRID = Grid(rgs_dir)
    GRID.load()

    RND = RandomGrid(GRID)

    fn = run_id + '.npz'
    sp = RND.interpolate(pp_arr, N_interpol_threads)
    np.savez(os.path.join(rnd_grid_dir, fn), flux=sp, labels=pp)
    shutil.rmtree(rgs_dir, ignore_errors=True)

    print('Grid model '+run_id+' complete')

    return 0


#----------------------------------------------
# Avoiding nested Pools with daemonic processes
# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
# Thank you Chris Arndt
class NonDaemonProcess(Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class CustomPool(Pool):
    Process = NonDaemonProcess
#----------------------------------------------


with CustomPool(processes=N_instances) as pool:
    ret = pool.map(run_one_item, work, chunksize=1)


print('Done.')










