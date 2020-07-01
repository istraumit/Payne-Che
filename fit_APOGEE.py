import sys, os
from SDSS import rdspec, Spec1D
from Network import Network
from Fit import Fit
import numpy as np
import matplotlib.pyplot as plt
from random_grid_common import *
from MCMC_Fit import MCMC_Fit

def fit_APOGEE(path, NN, Cheb_order, do_mcmc):

    spec = rdspec(path)
    wave_ = spec.wave
    if len(spec.flux.shape)==2:
        flux_ = spec.flux[0,:]
        err_ = spec.err[0,:]
    else:
        flux_ = spec.flux
        err_ = spec.err

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
    if do_mcmc:
        mcmc = MCMC_Fit(fit)
        popt, MAP, model_spec, chi2_func = mcmc.run(wave, flux, err)
    else:
        popt, pcov, model_spec, chi2_func = fit.run(wave, flux, err)
    
    CHI2 = chi2_func(popt)
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
            print(v, ':', '%.2f'%popt[k], param_units[i])
            k += 1

    print('RV:', '%.2f'%popt[-1], 'km/s')
    print('-'*25)

    plt.title(os.path.basename(path))
    plt.plot(wave, flux, label='Data')
    plt.plot(wave, model_spec, label='Model')
    plt.legend()
    plt.xlabel('Wavelength [A]')
    plt.ylabel('Flux')
    plt.show()


if __name__=='__main__':
    if len(sys.argv)<2:
        print('Use:', sys.argv[0], '<path_to_spectrum>')
        exit()

    fn = sys.argv[1]
    do_mcmc = False
    if len(sys.argv)>2 and sys.argv[2].lower()=='mcmc':
        do_mcmc = True

    NN_path = '/STER/ilyas/NN/NN_QRND_APOGEE.npz'
    NN = Network()
    NN.read_in(NN_path)

    fit_APOGEE(fn, NN, 10, do_mcmc)














