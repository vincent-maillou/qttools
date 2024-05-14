import time

import numpy as np
import scipy.sparse as sps
from mpi4py.MPI import COMM_WORLD as comm

from qttools.datastructures.dbcsr import DBCSR
from qttools.utils.mpi_utils import get_num_elements_per_section

# np.random.seed(1)


GLOBAL_STACK_SHAPE = (11,)

if comm.rank == 0:
    data = np.arange(1, 8)
    cols = np.array([0, 1, 2, 0, 1, 2, 3])
    rows = np.array([0, 0, 0, 1, 1, 1, 1])
    coo = sps.coo_array((data, (rows, cols)), shape=(4, 4))
else:
    coo = None

coo = comm.bcast(coo, root=0)

dbcsr = DBCSR.from_sparray(
    coo,
    [1] * 4,
    stack_shape=(get_num_elements_per_section(GLOBAL_STACK_SHAPE[0])[0][comm.rank],),
    global_stack_shape=GLOBAL_STACK_SHAPE,
)

# print(f"DBCSR data on rank {comm.rank}:\n", dbcsr.data)

print(comm.rank, dbcsr._padded_data)
print(comm.rank, dbcsr.data)
print(comm.rank, dbcsr._distribution_state)
t0 = time.perf_counter()

dbcsr.dtranspose()

print(comm.rank, dbcsr._padded_data)
print(comm.rank, dbcsr.data)
print(comm.rank, dbcsr._distribution_state)

t1 = time.perf_counter()
# print(comm.rank, dbcsr._distribution_state)
# print(comm.rank, dbcsr.masked_data)

dbcsr.dtranspose()

t2 = time.perf_counter()
# print(comm.rank, dbcsr.data)
# print(comm.rank, dbcsr.masked_data)
# print(comm.rank, dbcsr.stack_padding_mask)
# print(comm.rank, dbcsr._distribution_state)


# print(f"DBCSR data on rank {comm.rank}:\n", dbcsr.data)

print(f"Time for forward dtranspose: {t1-t0} s")
print(f"Time for backward dtranspose: {t2-t1} s")

dbcsr.return_dense = True


dbcsr.dtranspose()

# print(comm.rank, dbcsr.data)
# print(comm.rank, dbcsr.masked_data)
# print(comm.rank, dbcsr.stack_padding_mask)
