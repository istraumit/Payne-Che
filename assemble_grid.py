import sys, gc
import os
import numpy as np
from random_grid_common import *

grid_fn = '_GRID.npz'

if len(sys.argv) < 2:
    print('Usage:', sys.argv[0], '<path to the folder with models>')

path = sys.argv[1]

files = [fn for fn in os.listdir(path) if fn.endswith('.npz')]

fluxes, params = [],[]

for fn in files:
    if fn == grid_fn: continue
    npz = np.load(os.path.join(path, fn))
    fluxes.append(npz['flux'])
    param_dict = npz['labels'].item()
    pp = []
    for p in param_names:
        pp.append(param_dict[p])
    params.append(pp)
    print(fn)

opt = parse_inp()
wave = [float(x) for x in opt['wavelength']]
wave_grid = np.linspace(wave[0], wave[1], len(fluxes[0]))
np.savez(os.path.join(path, grid_fn), flux=fluxes, labels=params, wvl=wave_grid)

print('Assembled grid saved to', os.path.join(path, grid_fn))


