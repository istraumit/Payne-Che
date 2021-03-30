import os, sys
import numpy as np
from bisect import bisect
from DER_SNR import DER_SNR
from Network import Network
from common import param_names, param_units, parse_inp
from Fit import Fit
from fit_common import save_figure
from UncertFit import UncertFit
from random_grid_common import parse_inp
from multiprocessing import Pool, Lock
from FitLogger import FitLogger
import matplotlib.pyplot as plt

lock = Lock()

def fit_BOSS(path, NN, opt, logger, constraints={}):

    wave_start = float(opt['wave_range'][0])
    wave_end = float(opt['wave_range'][1])
    Cheb_order = int(opt['N_chebyshev'][0])

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
    R = int(opt['spectral_R'][0])
    fit.psf_R = R
    fit.N_presearch_iter = int(opt['N_presearch_iter'][0])
    fit.N_pre_search = int(opt['N_presearch'][0])
    unc_fit = UncertFit(fit, R)
    fit_res = unc_fit.run(wave, flux, err)
    CHI2 = fit_res.chi2_func(fit_res.popt)

    name = os.path.basename(path)
    row = [name]
    k = 0
    for i,v in enumerate(param_names):
        if NN.grid[v][0]!=NN.grid[v][1]:
            row.append('%.2f'%fit_res.popt[k])
            row.append('%.4f'%fit_res.uncert[k])
            k += 1
    row.append('%.2f'%fit_res.popt[k])
    row.append('%.2f'%fit_res.RV_uncert)
    txt = ' '.join(row)

    fit_res.wave = wave
    fit_res.model *= f_mean

    logger.save_plot(wave, flux*f_mean, fit_res.model, name)
    logger.add_record(txt)
    print(txt)

    return fit_res



if __name__=='__main__':
    if len(sys.argv)<3:
        print('Use:', sys.argv[0], '<config file> <path to the data>')
        exit()

    opt = parse_inp(sys.argv[1])
    path = sys.argv[2]

    #fit.psf = np.loadtxt('LAMOST_resolution.txt', skiprows=5)
    NN_path = opt['NN_path'][0]
    NN = Network()
    NN.read_in(NN_path)

    constr = {}
    #constr['[M/H]'] = (0.2-0.01, 0.2+0.01)
    
    logger = FitLogger(opt['log_dir'][0], delete_old=True)

    def process(fn):
        path_fn = os.path.join(path, fn)
        res = fit_BOSS(path_fn, NN, opt, logger)
        #M = np.vstack([res.wave, res.model])
        #np.save(path_fn + '.mod', M)


    if os.path.isfile(path):
        process(path)
    elif os.path.isdir(path):
        files = [fn for fn in os.listdir(path)]
        files.sort()
        parallel = opt['parallel'][0].lower() in ['true', 'yes', '1']
        if parallel:
            print('Parallel processing option is ON')
            with Pool() as pool:
                pool.map(process, files)
        else:
            for fn in files: process(fn)
    else:
        print('Path "'+path+'" is neither file nor directory.')








