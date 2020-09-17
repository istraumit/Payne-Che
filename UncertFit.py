import math
import numpy as np
from common import param_names, param_units

class UncertFit:
    """
    Wrapper around Fit that implements uncertainty estimation
    on top of the actual fit routine
    """
    def __init__(self, fit):
        self.fit = fit
        self.grid = fit.network.grid

    def run(self, wave, flux, flux_err):
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
                sigma = 1.0/math.sqrt(poly_coef[0])
                uncert.append(sigma)
                i+=1
        res.uncert = uncert
        return res
  
  
  