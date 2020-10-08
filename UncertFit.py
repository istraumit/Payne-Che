import math
import numpy as np
from common import param_names, param_units

class UncertFit:
    """
    Wrapper around Fit that implements uncertainty estimation
    on top of the actual fit routine
    """
    def __init__(self, fit, spectral_resolution):
        self.fit = fit
        self.grid = fit.network.grid
        self.resol = spectral_resolution

    def run(self, wave, flux, flux_err):
        return self._run_2(wave, flux, flux_err)
    
    def _run_2(self, wave, flux, flux_err):
        wave_start = min(wave)
        wave_end = max(wave)
        ndegree = 4 * self.resol * (wave_end - wave_start)/(wave_end + wave_start)
        
        CHI2_C = 1.0 + math.sqrt(2.0 / ndegree)
        
        res = self.fit.run(wave, flux, flux_err)
        popt, pcov, model_spec, chi2_func = res.popt, res.pcov, res.model, res.chi2_func
        
        uncert = []
        i=0
        for pn in param_names:
            if self.grid[pn][0]!=self.grid[pn][1]:
                step = self.grid[pn][2]
                xx = [popt[i]-step, popt[i], popt[i]+step]
                yy = []
                for x in xx:
                    pp = np.copy(popt)
                    pp[i] = x
                    yy.append(chi2_func(pp))
                poly_coef = np.polyfit(xx, yy, 2)
                poly_coef[-1] -= CHI2_C * yy[1]
                roots = np.roots(poly_coef)
                sigma = 0.5*abs(roots[0] - roots[1])
                uncert.append(sigma)
                i+=1
        res.uncert = uncert
        return res
        
    
    def _run_1(self, wave, flux, flux_err):
        res = self.fit.run(wave, flux, flux_err)
        popt, pcov, model_spec, chi2_func = res.popt, res.pcov, res.model, res.chi2_func
        
        uncert = []
        i=0
        for pn in param_names:
            if self.grid[pn][0]!=self.grid[pn][1]:
                step = self.grid[pn][2]
                xx = [popt[i]-step, popt[i], popt[i]+step]
                yy = []
                for x in xx:
                    pp = np.copy(popt)
                    pp[i] = x
                    yy.append(chi2_func(pp))
                poly_coef = np.polyfit(xx, yy, 2)
                if poly_coef[0]>0:
                    sigma = math.sqrt(2.0/poly_coef[0])
                else:
                    sigma = 0.0
                uncert.append(sigma)
                i+=1
        res.uncert = uncert
        return res
  
  
  