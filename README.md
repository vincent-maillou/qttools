# qttools
Quantum Transport Algorithms Toolbox

## How to install
```
1. Create a conda environment
    $ conda env create -f environment.yml

2. Install Qttools
    $ cd path/to/qttools
    $ pip install --no-dependencies -e .
```

Note: CuPy and mpi4py are shipped with the environment and configured w.r.t local clusters. If you are using a different cluster, you might need to install them manually.
