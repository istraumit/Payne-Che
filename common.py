param_names = ['T_eff','log(g)','v*sin(i)','v_micro','[M/H]']
param_units = ['[K]','[cm/s^2]','[km/s]','[km/s]','[dex]']

GSSP_steps = {}
GSSP_steps['T_eff'] = [10000, 100, 250]
GSSP_steps['log(g)'] = [0.1]
GSSP_steps['v_micro'] = [0.5]
GSSP_steps['[M/H]'] = [0.1]
GSSP_steps['v*sin(i)'] = [10]

assert all([key in param_names for key in GSSP_steps])
assert all([key in GSSP_steps for key in param_names])
