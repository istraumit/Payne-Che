import os, sys
import numpy as np
from bisect import bisect
from DER_SNR import DER_SNR
from Network import Network
from common import param_names, param_units
from Fit import Fit
from fit_common import save_figure
from UncertFit import UncertFit
from random_grid_common import parse_inp
from multiprocessing import Pool, Lock

lock = Lock()

def fit_BOSS(path, NN, wave_start, wave_end, Cheb_order, constraints={}):

    data = np.loadtxt(path)
    wave = data[:,0]
    flux = data[:,1]

    start_idx = bisect(wave, wave_start)
    end_idx = bisect(wave, wave_end)
    wave = wave[start_idx:end_idx]
    flux = flux[start_idx:end_idx]
    SNR = DER_SNR(flux)
    f_mean = np.mean(flux)
    flux /= f_mean
    err = flux / SNR

    grid_params = [p for p in param_names if NN.grid[p][0]!=NN.grid[p][1]]
    bounds_unscaled = np.zeros((2, len(grid_params)))
    for i,v in enumerate(grid_params):
        if v in constraints:
            bounds_unscaled[0,i] = constraints[v][0]
            bounds_unscaled[1,i] = constraints[v][1]
        else:
            bounds_unscaled[0,i] = NN.grid[v][0]
            bounds_unscaled[1,i] = NN.grid[v][1]

    fit = Fit(NN, Cheb_order)
    fit.bounds_unscaled = bounds_unscaled
    #fit.psf_R = 22500
    #fit.psf = np.loadtxt('LAMOST_resolution.txt', skiprows=5)
    fit.N_presearch_iter = 2
    fit.N_pre_search = 2000
    unc_fit = UncertFit(fit, 22500)
    fit_res = unc_fit.run(wave, flux, err)
    CHI2 = fit_res.chi2_func(fit_res.popt)

    name = os.path.basename(path)
    row = [name[:-4].split('_')[-1]]
    k = 0
    for i,v in enumerate(param_names):
        if NN.grid[v][0]!=NN.grid[v][1]:
            row.append('%.2f'%fit_res.popt[k])
            row.append('%.4f'%fit_res.uncert[k])
            k += 1
    row.append('%.2f'%fit_res.popt[k])
    row.append('%.2f'%fit_res.RV_uncert)
    txt = ' '.join(row)

    lock.acquire()
    with open('LOG_BOSS', 'a') as f:
        f.write(txt)
        f.write('\n')
    print(txt)
    lock.release()

    fit_res.wave = wave
    fit_res.model *= f_mean
    return fit_res



if __name__=='__main__':
    path = sys.argv[1]

    #NN_path = '/STER/ilyas/NN/BOSS/BOSS_NN_G6500_n300_b1000_v0.1.npz'
    NN_path = '/home/elwood/Documents/SDSS/NN/APOGEE/G4500_NN_n400_b1000_v0.1.npz'
    NN = Network()
    NN.read_in(NN_path)

    constr = {}
    #constr['[M/H]'] = (0.2-0.01, 0.2+0.01)

    files = [fn for fn in os.listdir(path) if fn.endswith('.txt')]
    
    def process(fn):
        path_fn = os.path.join(path, fn)
        res = fit_BOSS(path_fn, NN, 15000, 17000, 10, constraints=constr)
        M = np.vstack([res.wave, res.model])
        np.save(path_fn + '.mod', M)

    #for fn in files: process(fn)

    #exit()

    with Pool() as pool:
        pool.map(process, files)







