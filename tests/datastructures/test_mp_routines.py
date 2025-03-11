# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, sparse, xp
from qttools.datastructures import DSBCOO
from qttools.datastructures import (
    DSBSparse,
    mp_bd_matmul,
    mp_bd_sandwich,
)

def _create_btd_coo(sizes: NDArray) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""
    size = int(xp.sum(sizes))
    offsets = xp.hstack(([0], xp.cumsum(sizes)))

    arr_r = xp.zeros((size, size), dtype=xp.float64)
    arr_i = xp.zeros((size, size), dtype=xp.float64)
    for i in range(len(sizes)):
        # Diagonal block.
        block_shape = (int(sizes[i]), int(sizes[i]))
        arr_r[offsets[i] : offsets[i + 1], offsets[i] : offsets[i + 1]] = (
            xp.random.rand(*block_shape)
        )
        arr_i[offsets[i] : offsets[i + 1], offsets[i] : offsets[i + 1]] = (
            xp.random.rand(*block_shape)
        )
        # Superdiagonal block.
        if i < len(sizes) - 1:
            block_shape = (int(sizes[i]), int(sizes[i + 1]))
            arr_r[offsets[i] : offsets[i + 1], offsets[i + 1] : offsets[i + 2]] = (
                xp.random.rand(*block_shape)
            )
            arr_i[offsets[i] : offsets[i + 1], offsets[i + 1] : offsets[i + 2]] = (
                xp.random.rand(*block_shape)
            )
            arr_r[offsets[i + 1] : offsets[i + 2], offsets[i] : offsets[i + 1]] = (
                xp.random.rand(*block_shape).T
            )
            arr_i[offsets[i + 1] : offsets[i + 2], offsets[i] : offsets[i + 1]] = (
                xp.random.rand(*block_shape).T
            )
    rng = xp.random.default_rng()
    cutoff = rng.uniform(low=0.1, high=0.4)
    arr_r[xp.abs(arr_r) < cutoff] = 0
    arr_i[xp.abs(arr_r) < cutoff] = 0
    return sparse.coo_matrix(arr_r), sparse.coo_matrix(arr_i)

def test_bd_matmul(
    block_sizes: NDArray,
    global_stack_shape: tuple,
):
    coo_r, coo_i = _create_btd_coo(block_sizes)    
    global_stack_shape = global_stack_shape + (2,) 
    dsbsparse = DSBCOO.from_sparray(coo_r, block_sizes, global_stack_shape)
    for i in range(dsbsparse.stack_shape[0]):
        dsbsparse.data[i, 1, :] = coo_i.todense()[dsbsparse.rows, dsbsparse.cols]

    out = DSBCOO.from_sparray(
        sparse.coo_matrix(coo_r @ coo_r), block_sizes, global_stack_shape
    )

    mp_bd_matmul(dsbsparse, dsbsparse, out=out)

    a = dsbsparse.to_dense()[...,0,:,:] + 1j*dsbsparse.to_dense()[...,1,:,:]
    b = a @ a 

    assert xp.allclose(out.to_dense()[...,0,:,:], b.real)    
    assert xp.allclose(out.to_dense()[...,1,:,:], b.imag)


def test_bd_sandwich(
    block_sizes: NDArray,
    global_stack_shape: tuple,
):
    coo_r, coo_i = _create_btd_coo(block_sizes)    
    global_stack_shape = global_stack_shape + (2,) 
    dsbsparse = DSBCOO.from_sparray(coo_r, block_sizes, global_stack_shape)
    for i in range(dsbsparse.stack_shape[0]):
        dsbsparse.data[i, 1, :] = coo_i.todense()[dsbsparse.rows, dsbsparse.cols]

    out = DSBCOO.from_sparray(
        sparse.coo_matrix(coo_r @ coo_r @ coo_r), block_sizes, global_stack_shape
    )

    mp_bd_sandwich(dsbsparse, dsbsparse, out=out)

    a = dsbsparse.to_dense()[...,0,:,:] + 1j*dsbsparse.to_dense()[...,1,:,:]
    b = a @ a @ a

    assert xp.allclose(out.to_dense()[...,0,:,:], b.real)
    assert xp.allclose(out.to_dense()[...,1,:,:], b.imag)  

def _get_last_2d(x):
    m, n = x.shape[-2:]
    return x.flat[: m * n].reshape(m, n)
