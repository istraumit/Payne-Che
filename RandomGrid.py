import sys
import numpy as np
from scipy.interpolate import interpn
from multiprocessing import Pool
from Grid import *

class RandomGrid:

    def __init__(self, grid):
        assert(len(grid.models)>0)
        self.grid = grid

        params_list = [m.stellar_params().as_tuple() for m in grid.models]
        Nax = len(params_list[0])
        axes = [set() for i in range(Nax)]
        for i in range(Nax):
            for p in params_list:
                axes[i].add( p[i] )
        axes = [list(s) for s in axes]
        for ax in axes: ax.sort()
        self.axes = tuple([np.array(ax) for ax in axes])
        
        self.params_ranges = [(min(ax), max(ax)) for ax in self.axes]

        L = len(grid.models[0].flux)
        shape = tuple([len(ax) for ax in axes])
        CUBES = []
        for k in range(L): CUBES.append(np.ndarray(shape=shape))

        for i in range(len(grid.models)):
            indices = []
            for j in range(len(axes)):
                ind = axes[j].index(params_list[i][j])
                indices.append(ind)
            for k in range(L):
                CUBES[k][tuple(indices)] = grid.models[i].flux[k]

        self.CUBES = CUBES

    def sample_point_in_param_space(self):
        pp = []
        for pr in self.params_ranges:
            x = np.random.rand()
            p = pr[0] + x*(pr[1]-pr[0])
            pp.append(p)
        return np.array(pp)

    def interpolate_serial(self, p):
        spectrum = []
        for i in range(len(self.CUBES)):
            v = interpn(self.axes, self.CUBES[i], p)
            spectrum.append(v)
        return np.array(spectrum)
    
    def interpolate_parallel(self, p):
        work = [(self.axes, C, p) for C in self.CUBES]
        
        with Pool() as pool:
            print('Running interpolation with', pool._processes, 'processes')
            spectrum = pool.map(interp_call, work)
            
        return np.array(spectrum)
    
    def interpolate(self, p):
        return self.interpolate_parallel(p)
    
    def sample_model(self):
        p = self.sample_point_in_param_space()
        sp = self.interpolate(p)
        return p, sp
        
def interp_call(W):
    return interpn(W[0], W[1], W[2])
    








