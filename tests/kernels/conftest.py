# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numba as nb
import pytest

from qttools import xp

nb.set_num_threads(1)

NUM_INDS = [
    pytest.param(10, id="10"),
    pytest.param(42, id="42"),
]

NNZ = [
    pytest.param(200, id="200"),
    pytest.param(700, id="700"),
]

NUM_BLOCKS = [
    pytest.param(5, id="5"),
    pytest.param(10, id="10"),
]

COMM_SIZE = [
    pytest.param(5, id="5"),
    pytest.param(10, id="10"),
]

SHAPE = [
    pytest.param((50, 50), id="50x50"),
    pytest.param((97, 97), id="97x97"),
]

BLOCK_COORDS = [
    pytest.param((0, 2), id="10x10"),
    pytest.param((3, 4), id="20x20"),
]

M_N_K = [
    pytest.param((10, 10, 10), id="100x100x100"),
    pytest.param((20, 10, 10), id="200x100x100"),
    pytest.param((10, 20, 10), id="100x200x100"),
    pytest.param((10, 10, 20), id="100x100x200"),
    pytest.param((10, 10, 20), id="200x200x100"),
]

BATCHSHAPE = [
    pytest.param((1,), id="1"),
    pytest.param((1, 1), id="1x1"),
    pytest.param((3,), id="3"),
    pytest.param((3, 2), id="3x2"),
    pytest.param((11, 1), id="11x1"),
    pytest.param((1, 11, 1), id="1x11x1"),
]

DTYPE_COMPUTE_TYPE = [
    pytest.param((xp.float32, "32F"), id="32F 32F"),
    pytest.param((xp.float64, "64F"), id="64F 64F"),
    pytest.param((xp.complex64, "32F"), id="32C 32F"),
    pytest.param((xp.complex128, "64F"), id="64C 64F"),
    pytest.param((xp.float32, "16F"), id="32F 16F"),
    pytest.param((xp.float32, "16BF"), id="32F 16BF"),
    pytest.param((xp.float32, "32TF"), id="32F 32TF"),
    pytest.param((xp.complex64, "16F"), id="32C 16F"),
    pytest.param((xp.complex64, "16BF"), id="32C 16BF"),
    pytest.param((xp.complex64, "32TF"), id="32C 32TF"),
]

ORDER = [
    pytest.param("C", id="C"),
    pytest.param("F", id="F"),
]


@pytest.fixture(params=NUM_INDS)
def num_inds(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=NNZ)
def nnz(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=COMM_SIZE)
def comm_size(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=SHAPE)
def shape(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=NUM_BLOCKS)
def num_blocks(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=BLOCK_COORDS)
def block_coords(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=M_N_K)
def m_n_k(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=BATCHSHAPE)
def batchshape(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=DTYPE_COMPUTE_TYPE)
def dtype_compute_type(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=ORDER)
def order(request: pytest.FixtureRequest):
    return request.param
