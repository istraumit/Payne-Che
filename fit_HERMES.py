import sys,os
from math import *
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import interpolate
from bisect import bisect
from DER_SNR import DER_SNR
from random_grid_common import *
from Network import Network
from Fit import Fit
from numpy.polynomial.chebyshev import chebval

def load_spectrum(fn):

   # Read the data and the header is resp. 'spec' and 'header'
   flux = fits.getdata(fn)
   header = fits.getheader(fn)
   #
   # Make the equidistant wavelengthgrid using the Fits standard info
   # in the header
   #
   ref_pix = int(header['CRPIX1'])-1
   ref_val = float(header['CRVAL1'])
   ref_del = float(header['CDELT1'])
   numberpoints = flux.shape[0]
   unitx = header['CTYPE1']
   wavelengthbegin = ref_val - ref_pix*ref_del
   wavelengthend = wavelengthbegin + (numberpoints-1)*ref_del
   wavelengths = np.linspace(wavelengthbegin,wavelengthend,numberpoints)
   wavelengths = np.exp(wavelengths)

   return wavelengths, flux

def multiplot(wave, flux, N, title, lbl, xlbl, ylbl):
    di = len(wave)//N
    for i in range(N):
        i1 = i*di
        i2 = (i+1)*di
        plt.subplot(100*N + i + 11)
        if i==0: plt.title(title)
        plt.plot(wave[i1:i2], flux[i1:i2], label=lbl)
        if ylbl!=None: plt.ylabel(ylbl)
        if i==N-1:
            if xlbl!=None: plt.xlabel(xlbl)
            plt.legend()

def save_figure(save_to):
    fig = plt.gcf()
    fig.set_size_inches(50, 10)
    plt.tight_layout()
    fig.savefig(save_to, dpi=200)
    fig.clf()

def get_indices(wave, w1, w2):
   i1 = bisect(wave, w1)
   i2 = bisect(wave, w2)
   return i1, i2

def get_path(night, seq_id):
    return '/STER/mercator/hermes/'+night+'/reduced/'+seq_id.zfill(8)+'_HRF_OBJ_ext_CosmicsRemoved_log_merged_cf.fits'

def fit_HERMES(night, seq_id, NN, wave_start, wave_end, Cheb_order=5):
    fn = get_path(night, seq_id)
    wave, flux = load_spectrum(fn)
    hdr = fits.getheader(fn)
    obj_name = hdr['OBJECT']
    print('-'*25)
    print('Object:', obj_name)
    print('Night, SeqId:', night, seq_id)

    start_idx = bisect(wave, wave_start)
    end_idx = bisect(wave, wave_end)
    wave = wave[start_idx:end_idx]
    flux = flux[start_idx:end_idx]
    SNR = DER_SNR(flux)
    print('SNR:', SNR)
    flux /= np.mean(flux)
    err = flux / SNR

    fit = Fit(NN, Cheb_order)
    popt, pcov, model_spec, chi2_func = fit.run(wave, flux, err)
    CHI2 = chi2_func(popt)
    print('Chi^2:', '%.2e'%CHI2)

    if not os.path.exists('FIT'):
        os.makedirs('FIT')

    with open('FIT/LOG', 'a') as flog:
        L = [night, seq_id, '%.2e'%CHI2]
        L.extend( [str(x) for x in popt] )
        s = ' '.join(L)
        flog.write(s+'\n')

    print('-'*25)

    for i,v in enumerate(param_names):
        print(v, ':', '%.2f'%popt[i], param_units[i])

    print('RV:', '%.2f'%popt[-1], 'km/s')
    print('-'*25)

    N_subplots = 5
    xlbl = 'Wavelength [A]'
    ylbl = 'Flux'
    name = night + '_'+ seq_id + '.pdf'

    multiplot(wave, flux, N_subplots, obj_name, 'Data', xlbl, ylbl)
    multiplot(wave, model_spec, N_subplots, obj_name, 'Model', None, None)
    save_figure('FIT/FIT_' + name)
    plt.clf()
    
    resid = flux - model_spec
    multiplot(wave, resid, N_subplots, obj_name, 'Residuals', xlbl, ylbl)
    save_figure('FIT/RESID_' + name)
    plt.clf()
    
    return
    che_coef = popt[-Cheb_order-1:-1]
    print(che_coef)
    che_x = np.linspace(-1, 1, len(flux))
    che_poly = chebval(che_x, che_coef)
    norm_flux = che_poly
    multiplot(wave, norm_flux, 2, obj_name, 'Normalized flux', xlbl, ylbl)
    save_figure('FIT/NORM_' + name)
    plt.clf()
    

if __name__=='__main__':

    if len(sys.argv)<4:
        print('Use:', sys.argv[0], '<night> <sequence_id> <start_wavelength> <end_wavelength>')
        exit()

    night = sys.argv[1]
    seq_id = sys.argv[2]
    wave_start = float(sys.argv[3])
    wave_end = float(sys.argv[4])

    NN_path = '/STER/ilyas/inbox/NN_n1000_b512_v0.1.npz'
    NN = Network()
    NN.read_in(NN_path)

    fit_HERMES(night, seq_id, NN, wave_start, wave_end, Cheb_order=5)
    








