# Payne-Che
The [Payne](https://en.wikipedia.org/wiki/Cecilia_Payne-Gaposchkin) code with [Chebyshev](https://en.wikipedia.org/wiki/Pafnuty_Chebyshev) polynomials

The original Payne code: https://github.com/tingyuansen/The_Payne

Creating random grids
---------------------
1. Use 'free -g' command to check for available memory (column 'available', not 'free' memory).
2. Make sure that the free space on your /scratch file system is larger than the memory amount.
3. Move the code to /scratch file system.
4. Edit 'random_grid.conf' file, specify the 'memory_limit_GB' and other parameters.
5. Run 'random_grid.py'.
