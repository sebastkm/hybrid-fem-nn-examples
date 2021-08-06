# Hybrid FEM-NN Examples

This repository contains code for the examples in the paper "[Hybrid FEM-NN models: Combining artificial neural networks with the finite element method](https://doi.org/10.1016/j.jcp.2021.110651)".

### Setup

In order for the scripts to work, the `shared_code` folder must be in the python path.
You can do this by sourcing `setup.rc` from the root of the repository.
Note that `setup.rc` will also set the number of OMP threads to 1.
Feel free to change this, however, for some examples using a single thread is faster.

Also note that the PyTorch L-BFGS implementation was slightly modified to enable a callback-function.

