# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

if __name__ == "__main__":
    rows = 10
    cols = 5
    shape = (rows, cols)

    data = np.arange(rows * cols).reshape(*shape) if comm.rank == 0 else None
    data = comm.bcast(data, root=0)

    print(data) if comm.rank == 0 else None

    local_data = data[
        comm.rank * (rows // comm.size) : (comm.rank + 1) * (rows // comm.size)
    ].copy()

    comm.barrier()

    print(f"Data on rank {comm.rank}:\n", local_data)
    local_data = local_data.T.reshape(comm.size, local_data.shape[0], -1)
    print(f"Data on rank {comm.rank}:\n", local_data)

    comm.Alltoall(MPI.IN_PLACE, local_data)

    print(f"Data on rank {comm.rank}:\n", local_data)
    comm.barrier()

    print(f"Data on rank {comm.rank}:\n", local_data.flatten())
    # subarray_dtype.Free()
