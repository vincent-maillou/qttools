Clone the repository
```bash
git clone git@github.com:vincent-maillou/qttools.git
cd qttools
```

Create a conda environment from the provided `environment.yml` file
```bash
conda env create -f environment.yml
conda activate quatrex
```

Install mpi4py
```bash
conda install -c conda-forge mpi4py mpich
```
!!! warning

    - You might want to install a different mpi backend depending on the available 
    libraries in your system.

Install CuPy
```bash
conda install -c conda-forge cupy cuda-version=XX.X
```
!!! tip

    - To find out which version of cuda is running on your machine use
    `nvcc --version` or `nvidia-smi`.



Install qttools
```bash
pip install --no-dependencies --editable .
```

