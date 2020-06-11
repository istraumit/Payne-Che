import sys
import numpy as np
from scipy.interpolate import interpn
from Grid import *

class RandomGrid:

    def __init__(self, grid):
        assert(len(grid.models)>0)
        self.grid = grid

        params_list = [m.stellar_params().as_tuple() for m in grid.models]
        params_ranges = []
        for i in range(len(params_list[0])):
            pp = [p[i] for p in params_list]
            params_ranges.append( (min(pp), max(pp)) )
        self.params_ranges = params_ranges

        Nax = len(self.params_ranges)
        axes = [set() for i in range(Nax)]
        for i in range(Nax):
            for p in params_list:
                axes[i].add( p[i] )
        axes = [list(s) for s in axes]
        for ax in axes: ax.sort()
        self.axes = tuple([np.array(ax) for ax in axes])

        CUBE = np.ndarray(shape=tuple([len(ax) for ax in axes]), dtype=Model)
        for i in range(len(grid.models)):
            indices = []
            for j in range(len(axes)):
                ind = axes[j].index(params_list[i][j])
                indices.append(ind)
            CUBE[tuple(indices)] = grid.models[i]

        L = len(grid.models[0].wave)
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

        print('Number of models:', len(grid.models))
        print('Parameter ranges:', params_ranges)
        print('Spectrum length:', L)

    def sample_point_in_param_space(self):
        pp = []
        for pr in self.params_ranges:
            x = np.random.rand()
            p = pr[0] + x*(pr[1]-pr[0])
            pp.append(p)
        return np.array(pp)

    def sample_model(self):
        p = self.sample_point_in_param_space()
        print(p)
        spectrum = []
        for i in range(len(self.CUBES)):
            v = interpn(self.axes, self.CUBES[i], p)
            spectrum.append(v)
        return np.array(spectrum)
        










