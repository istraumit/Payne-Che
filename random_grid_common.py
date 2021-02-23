from common import *

DEBUG = True

rnd_grid_dir = 'rnd_grid_out'

def parse_inp(fn='random_grid.conf'):
    opt = {}
    with open(fn) as f:
        for line in f:
            text = line[:line.find('#')]
            parts = text.split(':')
            if len(parts) < 2: continue
            name = parts[0].strip()
            arr = parts[1].split(',')
            opt[name] = [a.strip() for a in arr]
    return opt






