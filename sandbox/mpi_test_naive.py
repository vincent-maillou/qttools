import numpy as np
import numpy.lib.stride_tricks as npst
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm


class MPIBuffer:
    def __init__(self, energies, data):
        self.all_energies = energies
        self.all_data = data
        self.nnz = data.shape[1]

        num_energies = len(energies)

        self.energies = np.ascontiguousarray(
            energies[
                comm.rank
                * (num_energies // comm.size) : (comm.rank + 1)
                * (num_energies // comm.size)
            ]
        )
        self.data = data[
            comm.rank
            * (num_energies // comm.size) : (comm.rank + 1)
            * (num_energies // comm.size)
        ]

    def dtranspose(self):
        self.data = np.ascontiguousarray(
            npst.as_strided(
                self.data,
                shape=(comm.size, self.energies.size, self.nnz // comm.size),
                strides=(
                    (self.nnz // comm.size) * self.data.itemsize,
                    self.nnz * self.data.itemsize,
                    self.data.itemsize,
                ),
            )
        )
        comm.Alltoall(MPI.IN_PLACE, self.data)

        self.energies = self.all_energies
        self.nnz = self.nnz // comm.size

        self.data = npst.as_strided(
            self.data,
            shape=(self.energies.size, self.nnz),
            strides=(self.nnz * self.data.itemsize, self.data.itemsize),
        )

        self.data = self.data.reshape(-1, self.nnz)


def main():
    num_energies = 8
    nnz = 16
    if not nnz % comm.size == 0:
        raise ValueError("Number of non-zeros must be divisible by number of ranks")
    if not num_energies % comm.size == 0:
        raise ValueError("Number of energies must be divisible by number of ranks")

    energies = np.arange(0, num_energies, 1) if comm.rank == 0 else None
    data = (
        np.arange(num_energies * nnz).reshape(num_energies, nnz)
        if comm.rank == 0
        else None
    )

    energies = comm.bcast(energies, root=0)
    data = comm.bcast(data, root=0)
    print(data) if comm.rank == 0 else None

    comm.barrier()

    mpi_buffer = MPIBuffer(energies, data)
    print(comm.rank, mpi_buffer.energies)
    print(comm.rank, mpi_buffer.data)

    mpi_buffer.dtranspose()
    print(comm.rank, mpi_buffer.energies)
    print(comm.rank, mpi_buffer.data)


if __name__ == "__main__":
    main()
