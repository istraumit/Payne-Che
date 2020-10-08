# Payne-Che
The [Payne](https://en.wikipedia.org/wiki/Cecilia_Payne-Gaposchkin) code with [Chebyshev](https://en.wikipedia.org/wiki/Pafnuty_Chebyshev) polynomials

The original Payne code: https://github.com/tingyuansen/The_Payne

The reference paper (Ting et al, 2019): https://doi.org/10.3847/1538-4357/ab2331

This is a modified version of the original Payne code, which estimates stellar parameters based on a normalized spectrum. The modified code 
performs estimation of stellar parameters with a non-normalized spectrum. Instead of normalizing the spectrum, a model spectrum is constructed
that contains the instrumental response function approximated by a Chebyshev polynomial series. The parameters
of the series are searched simultaneously with stellar parameters via optimization routine.

The repository contains modules that perform three different functions:
1. Creating grids of model spectra to be used as training sets;
2. Training neural networks based on grids of model spectra;
3. Fitting models to spectra using trained neural networks.

## Creating grids of model spectra

Run 'quasirandom_grid_single_cell.py' with no arguments. All the parameters are read out from 'random_grid.conf' file.
The parameters define grid extent and step, wavelength grid, the number of models in the grid. The script creates a
quasi-random grid based on a [Sobol sequence](https://en.wikipedia.org/wiki/Sobol_sequence). For each point in the
quasi-random grid, the script creates a single-cell subgrid that contains this point, by running GSSP code. The model
spectrum for the sampled point is then obtained by linearly interpolating models in the subgrid.

## Training neural networks

Assemble a grid into a single 'npz' file using 'assemble_grid.py' module and run 'train_NN.py' with it. The training algorithm 
uses torch framework and requires a CUDA device. The neural network is saved into 'NN_\*.npz' file.

## Fitting model spectra
Use 'fit_HERMES.py' or 'fit_APOGEE.py' to fit model spectra. 

'fit_APOGEE.py' takes as arguments a path to an APOGEE spectrum (apStar or apVisit) and a path to the neural network.

'fit_HERMES.py' takes night and sequence id as arguments and reads a HERMES spectrum from /STER filesystem. 
No need to normalize it. The path to the neural network is hardcoded at the moment.

