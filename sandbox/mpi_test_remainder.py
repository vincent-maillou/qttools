# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
import numpy.lib.stride_tricks as npst
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

from qttools.utils.mpi_utils import get_num_elements_per_section


class MPIBuffer:
    def __init__(self, data, global_shape):
        self.global_shape = global_shape
        local_sizes_rows = get_num_elements_per_section(global_shape[0])
        local_sizes_cols = get_num_elements_per_section(global_shape[1])
        local_offsets_cols = np.cumsum([0] + local_sizes_cols)

        self.shape = max(local_sizes_rows), max(local_sizes_cols) * comm.size
        self.data = np.zeros(self.shape, dtype=data.dtype)

        for i in range(comm.size):
            self.data[
                : data.shape[0],
                i * max(local_sizes_cols) : i * max(local_sizes_cols)
                + local_sizes_cols[i],
            ] = data[:, local_offsets_cols[i] : local_offsets_cols[i + 1]]

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
    num_rows = 20
    num_cols = 10

    data = (
        np.arange(num_rows * num_cols).reshape(num_rows, num_cols)
        if comm.rank == 0
        else None
    )
    data = comm.bcast(data, root=0)

    print(data) if comm.rank == 0 else None

    comm.barrier()

    mpi_buffer = MPIBuffer(
        np.array_split(data, comm.size, axis=0)[comm.rank],
        (num_rows, num_cols),
    )

    print(f"Data on rank {comm.rank}\n", mpi_buffer.data)

    mpi_buffer.dtranspose_forward()

    print(f"Data on rank {comm.rank}\n", mpi_buffer.data)

    mpi_buffer.dtranspose_backward()

    print(f"Data on rank {comm.rank}\n", mpi_buffer.data)


if __name__ == "__main__":
    main()
