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

        packet_dtype = from_numpy_dtype(self.data.dtype).Create_vector(
            count=self.energies.size,
            blocklength=self.nnz // comm.size,
            stride=self.nnz,
        )
        packet_dtype.Commit()

        block_type = MPI.Datatype.Create_struct(
            blocklengths=[1] * comm.size,
            displacements=[i * self.nnz // comm.size for i in range(comm.size)],
            datatypes=[packet_dtype] * comm.size,
        )
        block_type.Commit()

        self.packet_dtype = packet_dtype

    def dtranspose(self):
        # self.data = np.ascontiguousarray(
        #     npst.as_strided(
        #         self.data,
        #         shape=(comm.size, self.energies.size, self.nnz // comm.size),
        #         strides=(
        #             (self.nnz // comm.size) * self.data.itemsize,
        #             self.nnz * self.data.itemsize,
        #             self.data.itemsize,
        #         ),
        #     )
        # )
        # print(comm.rank, (comm.size - 1 - comm.rank) * (self.nnz // comm.size))
        # print(comm.rank, ((comm.size - comm.rank)) * (self.nnz // comm.size))
        comm.Alltoallv(MPI.IN_PLACE, (self.data, self.packet_dtype))

        self.energies = self.all_energies
        self.nnz = self.nnz // comm.size

        # self.data = npst.as_strided(
        #     self.data,
        #     shape=(self.energies.size, self.nnz),
        #     strides=(self.nnz * self.data.itemsize, self.data.itemsize),
        # )

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
    # print(comm.rank, mpi_buffer.energies)
    # print(comm.rank, mpi_buffer.data)

    mpi_buffer.dtranspose()
    # print(comm.rank, mpi_buffer.energies)
    # print(comm.rank, mpi_buffer.data)


if __name__ == "__main__":
    main()
