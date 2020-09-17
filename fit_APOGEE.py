import sys, os
from SDSS import rdspec, Spec1D
from Network import Network
from Fit import Fit
from UncertFit import UncertFit
import numpy as np
import matplotlib.pyplot as plt
from common import param_names, param_units
from fit_common import save_figure

def fit_APOGEE(path, NN, Cheb_order):

    spec = rdspec(path)
    
    if len(spec.flux.shape)==2:
        wave_ = spec.wave.flatten()
        flux_ = spec.flux.flatten()
        err_ = spec.err.flatten()
    else:
        wave_ = spec.wave
        flux_ = spec.flux
        err_ = spec.err

    wave_ = wave_[::-1]
    flux_ = flux_[::-1]
    err_  = err_[::-1]

    wave, flux, err = [],[],[]
    for i,v in enumerate(flux_):
        if v != 0.0:
            wave.append(wave_[i])
            flux.append(v)
            err.append(err_[i])

    flux_mean = np.mean(flux)
    flux /= flux_mean
    err /= flux_mean

    fit = Fit(NN, Cheb_order)
    unc_fit = UncertFit(fit)
    res = unc_fit.run(wave, flux, err)
    
    popt, pcov, model_spec, chi2_func = res.popt, res.pcov, res.model, res.chi2_func
    
    CHI2 = chi2_func(popt)
    print('-'*25)
    print('Chi^2:', '%.2e'%CHI2)

    if not os.path.exists('FIT'):
        os.makedirs('FIT')

    with open('FIT/LOG', 'a') as flog:
        L = [path, '%.2e'%CHI2]
        L.extend( [str(x) for x in popt] )
        s = ' '.join(L)
        flog.write(s+'\n')

    print('-'*25)

    k = 0
    for i,v in enumerate(param_names):
        if NN.grid[v][0]!=NN.grid[v][1]:
            print(v, ':', '%.2f'%popt[k], '+/-', '%.4f'%res.uncert[k], param_units[i])
            k += 1

    print('RV:', '%.2f'%popt[-1], 'km/s')
    print('-'*25)
    
    name = os.path.basename(path)[:-5]
    plt.title(name)
    plt.plot(wave, flux, label='Data')
    plt.plot(wave, model_spec, label='Model')
    plt.legend()
    plt.xlabel('Wavelength [A]')
    plt.ylabel('Flux')
    save_figure('FIT/'+name+'.png')


if __name__=='__main__':
    if len(sys.argv)<3:
        print('Use:', sys.argv[0], '<path_to_spectrum> <path_to_NN>')
        exit()

    fn = sys.argv[1]

    NN_path = sys.argv[2]
    NN = Network()
    NN.read_in(NN_path)

    fit_APOGEE(fn, NN, 10)














