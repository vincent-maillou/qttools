# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from numpy.typing import ArrayLike
from scipy import sparse

from qttools.datastructures import DSBSparse
from qttools.greens_function_solver import RGF, GFSolver
from qttools.utils.gpu_utils import get_host, xp


def test_selected_inv(
    bt_dense: ArrayLike,
    gfsolver_type: GFSolver,
    dsbsparse_type: DSBSparse,
    out: bool,
    max_batch_size: int,
    block_sizes: ArrayLike,
    global_stack_shape: int | tuple,
):
    bt_mask = bt_dense.astype(bool)
    ref_inv = xp.linalg.inv(bt_dense) * bt_mask

    coo = sparse.coo_matrix(get_host(bt_dense))

    block_sizes = get_host(block_sizes)
    densify_blocks = [(i, i) for i in range(len(block_sizes))]
    dsbsparse = dsbsparse_type.from_sparray(
        coo, block_sizes, global_stack_shape, densify_blocks
    )

    solver = gfsolver_type()

    if out:
        gf_inv = dsbsparse_type.zeros_like(dsbsparse)
        solver.selected_inv(dsbsparse, out=gf_inv, max_batch_size=max_batch_size)
    else:
        gf_inv = solver.selected_inv(dsbsparse, max_batch_size=max_batch_size)

    bt_mask_broadcasted = xp.broadcast_to(
        bt_mask, (*global_stack_shape, *bt_mask.shape)
    )

    assert xp.allclose(
        xp.broadcast_to(ref_inv, (*global_stack_shape, *ref_inv.shape)),
        gf_inv.to_dense() * bt_mask_broadcasted,
    )
