import sys, shutil
import subprocess as sp


def run_GSSP_once(string_list):

    star_name = string_list[0]
    Teff = string_list[1]
    logg = string_list[2]
    vmicro = string_list[3]
    vsini = string_list[4]
    metal = string_list[5]

    fn = star_name+'.inp.auto'
    with open(fn, 'w') as f:
        f.write(' '.join([Teff, '100.0', Teff]) + '\n' )
        f.write(' '.join([logg, '0.1', logg]) + '\n' )
        f.write(' '.join([vmicro, '1.0', vmicro]) + '\n' )
        f.write(' '.join([vsini, '10.0', vsini]) + '\n' )
        f.write('skip 0.03 0.02 0.07\n')
        f.write(' '.join(['skip', metal, '0.1', metal]) + '\n' )
        f.write('He 0.04 0.005 0.06\n')
        f.write('0.0 1.0 0.0 85000\n')
        f.write('/home/andrew/GSSP/abundances/\n')
        f.write('/home/andrew/GSSP/Kurucz/\n')
        f.write('2 1\n')
        f.write('ST\n')
        f.write('1 0.0156 fit\n')
        f.write('input_data/'+star_name+'.norm\n')
        f.write('0.5 0.99 5.9295 adjust\n')
        f.write('4205.0 5800.0\n')
        
    shutil.rmtree('output_files', ignore_errors=True)
    o = sp.check_output(['./GSSP', fn])
    lines = o.split('\n')
    chi2 = -1
    chi2_sigma = -1
    for line in lines:
        arr = line.split()
        if len(arr)<3: continue
        if arr[0] == 'Chi2_red_min':
            chi2 = float( arr[2] )
        if arr[0] == 'chi2_red_1sigma':
            chi2_sigma = float(arr[2])
    if chi2==-2: print(o)
    return (chi2, chi2_sigma)


def run_GSSP_grid(output_path, parameters, wave_range, GSSP_cmd, Kurucz=True):
    """
    This routine runs GSSP in grid mode.
    ------------------------------------
    output_path: where to save the generated inp file
    parameters: dictionary with keys ['T_eff','log(g)','v*sin(i)','v_micro','[M/H]']
                every entry is a list/tuple [start, end, step]
                example: parameters['T_eff'] = [5000, 9000, 100]
    wave_range: tuple with wavelength range and step (start, end, step)
    GSSP_cmd: command to run GSSP, such as './GSSP' for single thread
    Kurucz: use Kurucz models if True, otherwise LLmodels
    """

    def s(L, ns):
        fmt = '%.' + str(ns) + 'f'
        return [fmt%x for x in L]

    Teff = s(parameters['T_eff'], 0)
    logg = s(parameters['log(g)'], 1)
    vsini = s(parameters['v*sin(i)'], 0)
    vmicro = s(parameters['v_micro'], 1)
    metal = s(parameters['[M/H]'], 1)
    wave = s(wave_range, 4)

    with open(output_path, 'w') as f:
        f.write(' '.join([Teff[0], Teff[2], Teff[1]]) + '\n' )
        f.write(' '.join([logg[0], logg[2], logg[1]]) + '\n' )
        f.write(' '.join([vmicro[0], vmicro[2], vmicro[1]]) + '\n' )
        f.write(' '.join([vsini[0], vsini[2], vsini[1]]) + '\n' )
        f.write('skip 0.03 0.02 0.07\n')
        f.write(' '.join(['skip', metal[0], metal[2], metal[1]]) + '\n' )
        f.write('He 0.04 0.005 0.06\n')
        f.write('0.0 1.0 0.0 85000\n')
        f.write('/home/andrew/GSSP/abundances/\n')
        if Kurucz:
            f.write('/home/andrew/GSSP/Kurucz/\n')
        else:
            f.write('/home/andrew/GSSP/LLmodels/\n')
        f.write('2 1\n')
        f.write('ST\n')
        f.write('1 '+wave[2]+' grid\n')
        f.write('input_data.norm\n')
        f.write('0.5 0.99 5.9295 adjust\n')
        f.write(wave[0]+' '+wave[1]+'\n')
        
    shutil.rmtree('rgs_files', ignore_errors=True)
    
    o = sp.check_output(GSSP_cmd + ' ' + output_path, shell=True, stderr=sp.STDOUT)
    log_fn = output_path+'.log'
    with open(log_fn, 'wb') as f:
        f.write(o)




if __name__ == '__main__':
    out_fn = 'grid_gen_test.inp'
    p = {}
    p['T_eff'] = [6000, 6000, 100]
    p['log(g)'] = [4.0, 4.0, 0.1]
    p['v*sin(i)'] = [50, 50, 10]
    p['v_micro'] = [0, 2, 1]
    p['[M/H]'] = [-0.1, 0.1, 0.1]
    
    wave_range = [4500, 5000, 1]

    run_GSSP_grid(out_fn, p, wave_range, '/home/ilyas/SDSS/GSSP/bin/GSSP')
    



