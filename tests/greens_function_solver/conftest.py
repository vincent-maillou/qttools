# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, sparse, xp
from qttools.datastructures import DSBCOO, DSBCSR, DSBSparse
from qttools.greens_function_solver import RGF, GFSolver, Inv

GFSOLVERS_TYPE = [Inv, RGF]

DSBSPARSE_TYPES = [DSBCOO, DSBCSR]

SMALL_BLOCK_SIZES = [
    pytest.param(xp.array([2] * 10), id="constant-block-size-2"),
    pytest.param(xp.array([5] * 10), id="constant-block-size-5"),
    pytest.param(xp.array([2] * 3 + [4] * 2 + [2] * 3), id="mixed-block-size-2"),
    pytest.param(xp.array([5] * 3 + [10] * 2 + [5] * 3), id="mixed-block-size-5"),
]

MIDDLE_BLOCK_SIZES = [
    pytest.param(xp.array([20] * 10), id="constant-block-size-20"),
    pytest.param(xp.array([50] * 10), id="constant-block-size-50"),
    pytest.param(xp.array([20] * 3 + [40] * 2 + [20] * 3), id="mixed-block-size-20"),
    pytest.param(xp.array([50] * 3 + [100] * 2 + [50] * 3), id="mixed-block-size-50"),
]

LARGE_BLOCK_SIZES = [
    pytest.param(xp.array([200] * 10), id="large-constant-block-size-200"),
    pytest.param(xp.array([500] * 10), id="large-constant-block-size-500"),
    pytest.param(xp.array([200] * 3 + [400] * 2 + [200] * 3), id="large-mixed-block-size-200"),
    pytest.param(xp.array([500] * 3 + [1000] * 2 + [500] * 3), id="large-mixed-block-size-500"),
]

BATCHING_TYPE = [
    pytest.param(1, id="no-batching"),
    pytest.param(2, id="2-batching"),
    pytest.param(100, id="all-batching"),
]

OUT = [
    pytest.param(True, id="out_true"),
    pytest.param(False, id="out_false"),
]

RETURN_RETARDED = [
    pytest.param(True, id="return_retarded"),
    pytest.param(False, id="not_return_retarded"),
]

GLOBAL_STACK_SHAPES = [
    pytest.param((10,), id="1D-stack"),
    pytest.param((7, 2), id="2D-stack"),
    pytest.param((9, 2, 4), id="3D-stack"),
]

class BlockSizes:
    def __init__(self):
        if xp.__name__ == "cupy":
            device = xp.cuda.Device(0)
            free_bytes = device.mem_info[0]
            print(f"Free memory: {free_bytes}")
            if free_bytes > 1e10:
                self.sizes = MIDDLE_BLOCK_SIZES
            else:
                self.sizes = SMALL_BLOCK_SIZES
            return
        self.sizes = SMALL_BLOCK_SIZES


@pytest.fixture(params=BlockSizes().sizes, autouse=True)
def block_sizes(request: pytest.FixtureRequest) -> NDArray:
    return request.param


@pytest.fixture(params=GFSOLVERS_TYPE, autouse=True)
def gfsolver_type(request: pytest.FixtureRequest) -> GFSolver:
    return request.param


@pytest.fixture(params=DSBSPARSE_TYPES, autouse=True)
def dsbsparse_type(request: pytest.FixtureRequest) -> DSBSparse:
    return request.param


def _random_block(m: int, n: int) -> NDArray:
    """Generates a quasi-sparse random block of size m x n."""
    coo = sparse.random(int(m), int(n), density=0.5, format="coo").astype(xp.complex128)
    coo.data += 1j * xp.random.uniform(size=coo.nnz)
    return coo.toarray()


@pytest.fixture(scope="function", autouse=False)
def bt_dense(block_sizes: NDArray) -> NDArray:
    """Generates a random block-tridiagonal matrix."""
    block_offsets = xp.hstack(([0], xp.cumsum(block_sizes)))
    num_blocks = len(block_sizes)
    size = int(xp.sum(block_sizes))

    arr = xp.zeros((size, size), dtype=xp.complex128)

    # Fill the block-tridiagonal blocks
    for i in range(num_blocks):
        block = _random_block(block_sizes[i], block_sizes[i]) + xp.identity(
            int(block_sizes[i]), dtype=xp.complex128
        )

        arr[
            block_offsets[i] : block_offsets[i + 1],
            block_offsets[i] : block_offsets[i + 1],
        ] = block

        if i > 0:
            block = _random_block(block_sizes[i], block_sizes[i - 1])

            arr[
                block_offsets[i] : block_offsets[i + 1],
                block_offsets[i - 1] : block_offsets[i],
            ] = block

            block = _random_block(block_sizes[i - 1], block_sizes[i])

            arr[
                block_offsets[i - 1] : block_offsets[i],
                block_offsets[i] : block_offsets[i + 1],
            ] = block

    # Make the matrix diagonally dominant
    for i in range(arr.shape[0]):
        arr[i, i] = (1 + 1j) + complex(xp.sum(arr[i, :]))

    return arr


@pytest.fixture(params=BATCHING_TYPE, autouse=True)
def max_batch_size(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=OUT, autouse=True)
def out(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=RETURN_RETARDED, autouse=True)
def return_retarded(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=GLOBAL_STACK_SHAPES, autouse=True)
def global_stack_shape(request: pytest.FixtureRequest) -> tuple:
    return request.param
