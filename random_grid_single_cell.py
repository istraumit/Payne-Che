import os, time
import numpy as np
from Partition import *
from run_GSSP import *
from Grid import *
from RandomGrid import *
from Subgrid import *
from random_grid_common import *

def sample_point(grid):
    pp = {}
    for p in param_names:
        x = np.random.rand()
        q = grid[p]
        v = q[0] + x * (q[1] - q[0])
        pp[p] = v
    return pp
    
opt = parse_inp()

N_models = int(opt['N_models_to_sample'][0])
wave = [float(x) for x in opt['wavelength']]
GSSP_run_cmd = opt['GSSP_run_cmd'][0]

grid = {}
for o in opt:
    if o in param_names:
        grid[o] = [float(x) for x in opt[o]]

for i in range(N_models):
    pp = sample_point(grid)
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
    
    run_GSSP_grid('subgrid.inp', subgrid, wave, GSSP_run_cmd)

    GRID = Grid('rgs_files', '.')
    GRID.load()

    RND = RandomGrid(GRID)

    prefix = '%.0f'%(time.time()*1.e6)
    fn = prefix + '.npz'
    sp = RND.interpolate(np.array(pp_list))
    np.savez(os.path.join(rnd_grid_dir, fn), flux=sp, labels=pp)




print('Done.')













    