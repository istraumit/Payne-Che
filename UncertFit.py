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
        return self._run_3(wave, flux, flux_err)
    
    def _run_3(self, wave, flux, flux_err, p0 = None):

        #Returns list of indices that have val as value
        def get_indices(val,param):
            indices = []
            for i in range(0,len(param)):
                if param[i] == val:
                    indices.append(i)
            return indices

        #Removes all elements at the given indices
        def remove_elements(indices,lis):
            for i in list(reversed(indices)):
                del lis[i]

        def extract_smallest(param,chi2):
            params = param.tolist()
            chi2s = chi2.tolist()
            #plt.plot(params,chi2s,'k.') 

            param_smallest = []
            chi2_smallest = []
            while len(params) != 0:
                value = params[0]
                indices = get_indices(value,params)
                min_chi2 = np.inf
                for j in indices:
                    if chi2s[j] < min_chi2:
                        min_chi2 = chi2s[j]
                remove_elements(indices,params)
                remove_elements(indices,chi2s)
                param_smallest.append(value)
                chi2_smallest.append(min_chi2)
            return param_smallest, chi2_smallest

        def make_fit(x,y,degree,weights = None):
            P = np.polyfit(x,y,degree,w=weights)
            grid_step = abs(x[0]-x[1])
            abscis = np.linspace(min(x)-1*grid_step,max(x)+1*grid_step,1000)
            fit = [0]*len(abscis)
            for i in range(len(P)):
                fit = [f+P[i]*x**(len(P)-1-i) for x,f in zip(abscis,fit)]
            return abscis, fit, P

        def get_param(param):
            data = Chi2_table
            chi2 = data[:,-1] 
            i = param_names.index(param) 

            #plt.figure()
            par, ch = extract_smallest(data[:,i], chi2-CHI2_C*min(chi2))

            abscis,fit, P = make_fit(par,ch,2)
            c_err = [0]*len(abscis)
            roots = np.roots(P)
            sigma = 0.5*abs(roots[0] - roots[1])

            #plt.plot(par,ch,'.k')
            #plt.xlabel(param, fontsize=18)
            #plt.ylabel(r'$\chi^{2}$',fontsize = 18)
            #plt.plot(abscis,c_err,'k')
            #plt.plot(abscis,fit,'k')
            #plt.show()

            return sigma

        wave_start = min(wave)
        wave_end = max(wave)
        ndegree = 4 * self.resol * (wave_end - wave_start)/(wave_end + wave_start)
        CHI2_C = 1.0 + math.sqrt(2.0 / ndegree)

        res = self.fit.run(wave, flux, flux_err, p0=p0)
        popt, pcov, model_spec, chi2_func = res.popt, res.pcov, res.model, res.chi2_func

        ranges = [[popt[param_names.index(pn)]-self.grid[pn][2], popt[param_names.index(pn)], popt[param_names.index(pn)]+self.grid[pn][2]] for pn in param_names]

        Chi2 = []
        new_params = [[i,j,k,l,m] for i in ranges[0] for j in ranges[1] for k in ranges[2] for l in ranges[3] for m in ranges[4]]
        for c in new_params:
            pp = np.copy(popt)
            pp[:5] = c
            c.append(chi2_func(pp))
            Chi2.append(c)

        Chi2_table = np.array(Chi2)
        uncert = []
        for pn in param_names:
            uncert.append(get_param(pn))
            
        res.uncert = uncert
        return res

    
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
  
  
  
