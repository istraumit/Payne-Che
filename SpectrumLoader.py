import sys,os
import numpy as np
from astropy.io import fits
from SDSS import rdspec, Spec1D
from DER_SNR import DER_SNR


class SpectrumData:
    def __init__(self, wave, flux, err):
        self.wave = wave
        self.flux = flux
        self.err = err


def load_HERMES(fn):

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
    SNR = DER_SNR(flux)
    err = flux/SNR

    sd = SpectrumData(wavelengths, flux, err)
    sd.obj_id = os.path.basename(fn)
    return sd

def load_APOGEE(path):
    spec = rdspec(path)
    wave_ = spec.wave
    
    if len(spec.flux.shape)==2:
        flux_ = spec.flux[0,:]
        err_ = spec.err[0,:]
    else:
        flux_ = spec.flux
        err_ = spec.err

    objid = fits.getheader(path, 0)['OBJID']
    dirname = os.path.dirname(path)
    LSF_dir = os.path.join(dirname, 'LSF')
    if os.path.isdir(LSF_dir):
        files = os.listdir(LSF_dir)
        LSF = [fn for fn in files if ('LSF' in fn) and (objid in fn)]
        if len(LSF)==1:
            LSF_path = os.path.join(LSF_dir, LSF[0])
            LSF_data = fits.getdata(LSF_path, 0)

    wave, flux, err, LSF = [],[],[],[]
    for i,fv in enumerate(flux_):
        if fv == 0.0 or np.isnan(fv): continue
        if 'LSF_data' in locals() and any(np.isnan(LSF_data[i,:])): continue
        wave.append(wave_[i])
        flux.append(fv)
        err.append(err_[i])
        if 'LSF_data' in locals():
            LSF.append(LSF_data[i,:])

    sd = SpectrumData(np.array(wave), np.array(flux), np.array(err))
    sd.obj_id = objid
    sd.file_path = path
    if len(LSF) > 0:
        sd.LSF_data = np.array(LSF)

    return sd


def load_ASCII(path):
    data = np.loadtxt(path)
    wave = data[:,0]
    flux = data[:,1]
    if data.shape[1]>2:
        err = data[:,2]
    else:
        SNR = DER_SNR(flux)
        err = flux/SNR
    sd = SpectrumData(wave, flux, err)
    sd.obj_id = os.path.basename(path)
    return sd


def load_NPZ(path):
    npz = np.load(path)
    flux = np.squeeze(npz['flux'])
    w = npz['wave']
    N = len(flux)
    wave = np.linspace(w[0], w[1], N)
    err = 1e-3 + np.zeros(len(flux))
    sd = SpectrumData(wave, flux, err)
    sd.obj_id = os.path.basename(path)
    return sd


class SpectrumWrapper():
    """
    The purpose of this class is to provide loading 'on demand' to avoid 
    loading all the data into memory in case if a large number of spectra
    is being processed
    """
    def __init__(self, load_func, path):
        self._load = load_func
        self._path = path

    def load(self):
        return self._load(self._path)

    def __repr__(self):
        return self._path


class SpectrumLoader():
    def __init__(self, format):
        _selector = {
            'HERMES':load_HERMES,
            'APOGEE':load_APOGEE,
            'ASCII':load_ASCII, 
            'NPZ':load_NPZ,}

        if not format in _selector:
            raise Exception('Unknown spectrum format '+format)

        self._load_func = _selector[format]

    def get_spectra(self, path, ext_filter=''):
        if os.path.isfile(path):
            return [SpectrumWrapper(self._load_func, path)]
        elif os.path.isdir(path):
            files = [fn for fn in os.listdir(path) if fn.endswith(ext_filter)]
            files.sort()
            return [SpectrumWrapper(self._load_func, os.path.join(path, fn)) for fn in files]

if __name__=='__main__':
    import matplotlib.pyplot as plt

    SL = SpectrumLoader('APOGEE')

    spectra = SL.get_spectra('/home/elwood/Documents/SDSS/Pathfinder/APOGEE/DR17/spectra')

    for sp in spectra:
        sd = sp.load()

        plt.title(sp._path)
        plt.plot(sd.wave, sd.err)
        plt.show()














