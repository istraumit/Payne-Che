DEBUG = True

param_names = ['T_eff','log(g)','v*sin(i)','v_micro','[M/H]']
param_units = ['[K]','[cm/s^2]','[km/s]','[km/s]','[dex]']
rnd_grid_dir = 'rnd_grid_out'

def parse_inp():
    opt = {}
    with open('random_grid.conf') as f:
        for line in f:
            text = line[:line.find('#')]
            parts = text.split(':')
            if len(parts) < 2: continue
            name = parts[0].strip()
            arr = parts[1].split(',')
            opt[name] = [a.strip() for a in arr]
    return opt
