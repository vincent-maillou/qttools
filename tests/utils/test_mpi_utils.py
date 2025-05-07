# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from pathlib import Path

import pytest
import scipy.sparse as sps
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import sparse as sparse
from qttools import xp
from qttools.comm import comm
from qttools.utils.mpi_utils import distributed_load, get_local_slice, get_section_sizes


def setup_module():
    """setup any state specific to the execution of the given module."""
    _default_config = {
        "all_to_all": "device_mpi",
        "all_gather": "device_mpi",
        "all_reduce": "device_mpi",
        "bcast": "device_mpi",
    }
    # Configure the comm singleton.
    comm.configure(
        block_comm_size=1,
        block_comm_config=_default_config,
        stack_comm_config=_default_config,
        override=True,
    )


@pytest.mark.parametrize(
    "num_elements, num_sections, strategy, expected",
    [
        (10, 2, "balanced", ([5, 5], 10)),
        (10, 3, "greedy", ([4, 4, 2], 12)),
        (7, 3, "balanced", ([3, 2, 2], 9)),
        (7, 3, "greedy", ([3, 3, 1], 9)),
        (7, 7, "balanced", ([1, 1, 1, 1, 1, 1, 1], 7)),
        (7, 7, "greedy", ([1, 1, 1, 1, 1, 1, 1], 7)),
    ],
)
def test_get_section_sizes(
    num_elements: int,
    num_sections: int,
    strategy: str,
    expected: tuple[list[int], int],
):
    assert (
        get_section_sizes(
            num_elements=num_elements,
            num_sections=num_sections,
            strategy=strategy,
        )
        == expected
    )


@pytest.mark.mpi(min_size=2)
def test_distributed_load_npy(mpi_tmp_path: Path):
    """Test the distributed_load function."""
    arr = None
    if global_comm.rank == 0:
        arr = xp.random.rand(10)
        xp.save(mpi_tmp_path / "arr.npy", arr)
    arr = global_comm.bcast(arr, root=0)

    loaded_arr = distributed_load(mpi_tmp_path / "arr.npy")
    assert xp.allclose(arr, loaded_arr)


@pytest.mark.mpi(min_size=2)
def test_distributed_load_npz(mpi_tmp_path: Path):
    """Test the distributed_load function."""
    coo = None
    if global_comm.rank == 0:
        coo = sps.random(10, 10, density=0.5)
        sps.save_npz(mpi_tmp_path / "coo.npz", coo)
    coo = sparse.coo_matrix(global_comm.bcast(coo, root=0))

    loaded_arr = distributed_load(mpi_tmp_path / "coo.npz")
    assert xp.allclose(coo.toarray(), loaded_arr.toarray())


@pytest.mark.mpi(min_size=2)
def test_get_local_slice():
    """Test the distributed_load function."""
    global_array = xp.arange(10)
    local_arrays = xp.array_split(global_array, global_comm.size)
    assert xp.allclose(local_arrays[global_comm.rank], get_local_slice(global_array))
