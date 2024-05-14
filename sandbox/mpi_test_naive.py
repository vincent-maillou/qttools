# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
import numpy.lib.stride_tricks as npst
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm


class MPIBuffer:
    def __init__(self, data):
        self.data = np.ascontiguousarray(data)
        self.shape = data.shape

    def dtranspose_forward(self):
        """Transpose the data in a distributed fashion."""
        self.data = np.ascontiguousarray(
            npst.as_strided(
                self.data,
                shape=(comm.size, self.shape[0], self.shape[1] // comm.size),
                strides=(
                    (self.shape[1] // comm.size) * self.data.itemsize,
                    self.shape[1] * self.data.itemsize,
                    self.data.itemsize,
                ),
            )
        )

        comm.Alltoall(MPI.IN_PLACE, self.data)

        self.shape = self.shape[0] * comm.size, self.shape[1] // comm.size
        self.data = self.data.reshape(self.shape)

    def dtranspose_backward(self):
        """Transpose the data in a distributed fashion."""
        self.data = self.data.reshape(
            comm.size, self.shape[0] // comm.size, self.shape[1]
        )
        comm.Alltoall(MPI.IN_PLACE, self.data)

        self.data = self.data.transpose(1, 0, 2)

        self.shape = self.shape[0] // comm.size, self.shape[1] * comm.size
        self.data = self.data.reshape(self.shape)


def main():
    num_cols = 7
    num_rows = 10
    if not num_rows % comm.size == 0:
        raise ValueError("Number of rows must be divisible by number of ranks")
    if not num_cols % comm.size == 0:
        raise ValueError("Number of cols must be divisible by number of ranks")

    data = (
        np.arange(num_rows * num_cols).reshape(num_rows, num_cols)
        if comm.rank == 0
        else None
    )
    data = comm.bcast(data, root=0)

    print(data) if comm.rank == 0 else None

    comm.barrier()

    mpi_buffer = MPIBuffer(
        data[
            comm.rank
            * (data.shape[0] // comm.size) : (comm.rank + 1)
            * (data.shape[0] // comm.size)
        ],
    )
    print(f"Data on rank {comm.rank}\n", mpi_buffer.data)

    mpi_buffer.dtranspose_forward()

    print(f"Data on rank {comm.rank}\n", mpi_buffer.data)

    mpi_buffer.dtranspose_backward()

    print(f"Data on rank {comm.rank}\n", mpi_buffer.data)


if __name__ == "__main__":
    main()
