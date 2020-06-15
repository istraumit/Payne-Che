# Payne-Che
The [Payne](https://en.wikipedia.org/wiki/Cecilia_Payne-Gaposchkin) code with [Chebyshev](https://en.wikipedia.org/wiki/Pafnuty_Chebyshev) polynomials

The original Payne code: https://github.com/tingyuansen/The_Payne

Creating random grids: method 1
-------------------------------
1. Use 'free -g' command to check for available memory (column 'available', not 'free' memory).
2. Make sure that the free space on your /scratch file system is larger than the memory amount.
3. Move the code to /scratch file system.
4. Edit 'random_grid.conf' file, specify the 'memory_limit_GB' and other parameters.
5. Run 'random_grid.py'.

Method 1 partitions the full grid into a series of subgrids and randomly samples models within each subgrid. Designed to overcome memory limitation when dealing with large grids.

Creating random grids: method 2
-------------------------------
1. Edit 'random_grid.conf' file,
2. Run 'random_grid_single_cell.py'.

Method 2 samples a point in the parameter space randomly and runs GSSP to create a single-cell grid around this point. The model fluxes are then linearly interpolated for the sampled point. The process is repeated 'N_models_to_sample' (parameter in the config file) times.

Models in the random grid are saved into 'rnd_grid_out' folder as NPZ files. The name of a file is formed from a current timestamp. Each NPZ file contains a flux array and a dictionary with the stellar parameter values. GSSP input and output (including stderr) are saved into 'subgrid.inp' and 'subgrid.inp.log'.
