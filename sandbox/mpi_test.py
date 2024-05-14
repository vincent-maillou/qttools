# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from mpi4py.util.dtlib import from_numpy_dtype


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
        self.data = np.ascontiguousarray(
            data[
                comm.rank
                * (num_energies // comm.size) : (comm.rank + 1)
                * (num_energies // comm.size)
            ],
            dtype=np.int32,
        )

        subarray_dtype = from_numpy_dtype(self.data.dtype).Create_subarray(
            sizes=(num_energies // comm.size, self.nnz),
            subsizes=(num_energies // comm.size, self.nnz // comm.size),
            starts=(0, 0),
        )

        self.subarray_dtype = subarray_dtype.Commit()

    def dtranspose(self):
        comm.Alltoall(MPI.IN_PLACE, (self.data, 2, self.subarray_dtype))

        self.energies = self.all_energies
        self.nnz = self.nnz // comm.size

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
    # print(data) if comm.rank == 0 else None

    comm.barrier()

    mpi_buffer = MPIBuffer(energies, data)
    mpi_buffer.dtranspose()
    print(comm.rank, mpi_buffer.energies)
    print(comm.rank, mpi_buffer.data)

    # print(comm.rank, mpi_buffer.energies)
    # print(comm.rank, mpi_buffer.data)


def _blockdist(N, size, rank):
    q, r = divmod(N, size)
    r = 0
    n = q
    s = rank * q + min(rank, r)
    return (n, s)


def _subarraytypes(comm, shape, axis, subshape, dtype):
    N = shape[axis]
    datatype = MPI._typedict[dtype.char]
    sizes = list(subshape)
    subsizes = sizes[:]
    substarts = [0] * len(sizes)
    datatypes = []
    for i in range(comm.size):
        n, s = _blockdist(N, comm.size, i)
        subsizes[axis] = n
        substarts[axis] = s
        print(sizes, subsizes, substarts)
        newtype = datatype.Create_subarray(sizes, subsizes, substarts).Commit()
        datatypes.append(newtype)
    return tuple(datatypes)


def test():
    num_energies = 12
    nnz = 8

    data = (
        np.arange(num_energies * nnz).reshape(num_energies, nnz)
        if comm.rank == 0
        else None
    )
    data = comm.bcast(data, root=0)
    print(data) if comm.rank == 0 else None
    comm.barrier()

    data = np.ascontiguousarray(
        data[
            comm.rank
            * (num_energies // comm.size) : (comm.rank + 1)
            * (num_energies // comm.size)
        ],
        dtype=np.int32,
    )
    energy_types = _subarraytypes(
        comm, (num_energies, nnz), 0, (num_energies, nnz // comm.size), data.dtype
    )
    nnz_types = _subarraytypes(
        comm, (num_energies, nnz), 1, (num_energies // comm.size, nnz), data.dtype
    )
    counts_displs = ([1] * comm.size, [0] * comm.size)

    print(comm.rank, data)

    comm.Alltoallw(
        [data, counts_displs, nnz_types], [data, counts_displs, energy_types]
    )

    print(comm.rank, data.reshape(-1, nnz // comm.size))

    [dtype.Free() for dtype in energy_types]
    [dtype.Free() for dtype in nnz_types]


if __name__ == "__main__":
    test()
    # main()
