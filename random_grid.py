import os, time
import numpy as np
from Partition import *
from run_GSSP import *
from Grid import *
from RandomGrid import *

DEBUG = True

param_names = ['T_eff','log(g)','v*sin(i)','v_micro','[M/H]']
rnd_grid_dir = 'rnd_grid_out'

def parse_inp():
    opt = {}
    with open('random_grid.conf') as f:
        for line in f:
            text = line[:line.find('#')]
            parts = text.split(':')
            if len(parts) < 2: continue
            name = parts[0].strip()
            arr = parts[1].split(',')
            opt[name] = [a.strip() for a in arr]
    return opt
    
opt = parse_inp()

if DEBUG:
    for o in opt: print(o, opt[o])

wave = [float(x) for x in opt['wavelength']]
N_wave_points = int( (wave[1]-wave[0])/wave[2] )
N_bytes_per_model = 8 * N_wave_points

memory_limit = float(opt['memory_limit_GB'][0]) * 1.e9

models_limit = memory_limit / N_bytes_per_model

grid = {}
for o in opt:
    if o in param_names:
        grid[o] = [float(x) for x in opt[o]]

if DEBUG:
    print('-'*25)
    print('Stellar parameters:', grid)

print('-'*25)
PART = Partition(grid, models_limit)
PART.optimize_partition()
PART.check()

subgrid = {}
for p in param_names:
    n = int(PART.dims_info[p].length)
    subgrid[p] = (grid[p][0], grid[p][0]+n*grid[p][2], grid[p][2])

print('-'*25)
print('Current subgrid:')
for p in subgrid: print(p, subgrid[p])

print('-'*25)
sample_density = float(opt['N_models_to_sample'][0]) / PART.total_volume
print('Sample density:', sample_density)

subgrid_samples = int(PART.subgrid_volume * sample_density)
assert subgrid_samples > 0

print('Subgrid samples:', subgrid_samples)

GSSP_run_cmd = opt['GSSP_run_cmd'][0]
print('>>>> RUNNING GSSP')
run_GSSP_grid('subgrid.inp', subgrid, wave, GSSP_run_cmd)
print('>>>> GSSP FINISHED')

print('Loading subgrid')
GRID = Grid('rgs_files', '.')
GRID.load()
print('Subgrid loaded, sampling random grid')

if not os.path.exists(rnd_grid_dir):
    os.makedirs(rnd_grid_dir)

prefix = '%.0f'%(time.time()*1.e6)

RND = RandomGrid(GRID)
for i in range(subgrid_samples):
    fn = prefix + '_' + str(i).zfill(8) + '.npz'
    pp, sp = RND.sample_model()
    np.savez(os.path.join(rnd_grid_dir, fn), flux=sp, labels=pp)

print('Done.')



















