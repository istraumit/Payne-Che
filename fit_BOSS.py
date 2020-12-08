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

PRINT_SHORT = True

def fit_BOSS(path, NN, wave_start, wave_end, Cheb_order):

    data = np.loadtxt(path)
    wave = data[:,0]
    flux = data[:,1]

    start_idx = bisect(wave, wave_start)
    end_idx = bisect(wave, wave_end)
    wave = wave[start_idx:end_idx]
    flux = flux[start_idx:end_idx]
    SNR = DER_SNR(flux)
    flux /= np.mean(flux)
    err = flux / SNR

    fit = Fit(NN, Cheb_order)
    unc_fit = UncertFit(fit, 85000)
    fit_res = unc_fit.run(wave, flux, err)
    CHI2 = fit_res.chi2_func(fit_res.popt)

    name = os.path.basename(path)
    if PRINT_SHORT:
        name_split = name[:-5].split('_')
        row = []
        row.extend(name_split)
        k = 0
        for i,v in enumerate(param_names):
            if NN.grid[v][0]!=NN.grid[v][1]:
                row.append('%.2f'%fit_res.popt[k])
                row.append('%.4f'%fit_res.uncert[k])
                k += 1
        print(' '.join(row))
    else:
        print('SNR:', SNR)
        print('Chi^2:', '%.2e'%CHI2)

        print('-'*25)
        k = 0
        for i,v in enumerate(param_names):
            if NN.grid[v][0]!=NN.grid[v][1]:
                print(v, ':', '%.2f'%fit_res.popt[k], '+/-', '%.4f'%fit_res.uncert[k], param_units[i])
                k += 1

        print('RV:', '%.2f'%fit_res.popt[-1], 'km/s')
        print('-'*25)


if __name__=='__main__':
    path = sys.argv[1]
    NN_path = '/STER/ilyas/NN/BOSS/BOSS_NN_G6500_n300_b1000_v0.1.npz'
    NN = Network()
    NN.read_in(NN_path)

    opt = parse_inp()

    grid = {}
    for o in opt:
        if o in param_names:
            grid[o] = [float(x) for x in opt[o]]

    NN.grid = grid

    files = os.listdir(path)
    files.sort()
    i=0
    for fn in files:
        if not PRINT_SHORT: print(fn)
        path_fn = os.path.join(path, fn)
        fit_BOSS(path_fn, NN, 4200, 5800, 10)
        i += 1
        #if i>2: break









