import os
import sys
from os.path import join
import numpy as np
import hashlib

class StellarParams:
    
    def __init__(self, Teff, logg, vsini, MH, vmicro):
        self.Teff = Teff
        self.logg = logg
        self.vsini = vsini
        self.MH = MH
        self.vmicro = vmicro
        
    def as_tuple(self):
        return (self.Teff, self.logg, self.vsini, self.vmicro, self.MH)
        
    def __str__(self):
        return 'Teff=%.0f, log(g)=%.2f, v*sin(i)=%.0f, v_micro=%.1f, [M/H]=%.1f,'%self.as_tuple()
        
    def __repr__(self):
        return self.__str__()


class Model:
    def __init__(self, name, wave, flux):
        assert(type(wave)==np.ndarray)
        assert(type(flux)==np.ndarray)
        self.wave = wave
        self.flux = flux
        self.name = name
        
    def __str__(self):
        return self.name + '[%d %d]'%(len(self.wave), len(self.flux))
        
    def __repr__(self):
        return self.__str__()
        
    def stellar_params(self):
        a = self.name[2:-4].split('_')
        MH = 0.1*float(a[0])
        if self.name[1]=='m': MH = -MH
        Teff = float(a[1])
        logg = 0.01*float(a[2])
        vmicro = 0.1*float(a[3])
        vsini = float(a[6])
        sp = StellarParams(Teff, logg, vsini, MH, vmicro)
        return sp



class Grid:
    def __init__(self, grid_folder, cache_folder):
        self.folder = grid_folder
        self.models = []
        self.models_dict = {}
        h = hashlib.md5(grid_folder.encode('utf-8')).hexdigest()
        self.cache_folder = cache_folder
        self.cache = join(cache_folder, '__grid_'+h+'.npy')

    def _load_models_from_storage(self):
        A = np.load(self.cache)
        N = []
        with open(self.cache+'.names') as f:
            for line in f:
                N.append(line)
            
        for i in range(A.shape[0]):
            M = Model(N[i], A[i,0,:], A[i,1,:])
            self.models.append(M)
            self.models_dict[str(M.stellar_params())] = M

    def set_cache(self, h):
        self.cache = join(self.cache_folder, '__grid_'+h+'.npy')

    def load(self):
        if True:
            L, names = [],[]
            grid_files = [f for f in os.listdir(self.folder) if f.endswith(".rgs")]
            total = len(grid_files)
            processed = 0
            report_next = 1
            for grid_fn in grid_files:
                fn = join(self.folder, grid_fn)
                rgs = np.loadtxt(fn)

                model_wave = rgs[:,0]
                model_flux = rgs[:,1]
                L.append( [model_wave, model_flux] )
                names.append(grid_fn)
                processed += 1
                progress = 100 * processed/total
                if progress>report_next:
                    #print('%.0f%% '%progress, end='', flush=True)
                    report_next += 1
                
            np.save(self.cache, L)
            with open(self.cache+'.names', 'w') as f:
                for name in names:
                    f.write(name)
                    f.write('\n')

        self._load_models_from_storage()
        
    def __repr__(self):
        r = 'Grid with '+str(len(self.models))+' models'
        if os.path.isfile(self.cache):
            r += '; '+self.cache+' exists.'
        return r
        








