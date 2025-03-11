# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, sparse, xp
from qttools.datastructures import DSBCOO, DSBSparse
from qttools.kernels import dsbcoo_kernels
from qttools.profiling import Profiler

profiler = Profiler()


@profiler.profile(level="api")
def correct_out_range_index(i: int, k: int, num_blocks: int):
    # find the index of block in the matrix being repeated into open-end
    # based on the difference of row and col, ie diagonal
    diag = k - i
    k_1 = min(max(k, 0), num_blocks - 1)
    i_1 = k_1 - diag  # keep the same diag
    i_2 = min(max(i_1, 0), num_blocks - 1)
    k_2 = i_2 + diag  # keep the same diag
    return (i_2, k_2)



@profiler.profile(level="debug")
def _set_block_C2R(
    dsbcoo: DSBCOO,
    row: int,
    col: int,
    block: NDArray,
    form: str = "v",
    offset=None,
    scaling=None,
) -> None:
    data_stack = dsbcoo.data
    block_slice = dsbcoo._get_block_slice(row, col)
    # print(data_stack.shape)
    assert data_stack.shape[-2] == 2

    if block_slice.start is None and block_slice.stop is None:
        # No data in this block, nothing to do.
        return

    if scaling is None:        
        scaling = xp.array([1.0, 1.0]) 
    
    if offset is None:
        offset = xp.array([0.0, 0.0])

    if form == 'v':
        sparsify_block_v(
                block,
                dsbcoo.rows[block_slice] - dsbcoo.block_offsets[row],
                dsbcoo.cols[block_slice] - dsbcoo.block_offsets[col],
                data_stack[..., block_slice],
                scaling,
                offset
            )
    

@profiler.profile(level="debug")
def _get_block_R2C(
    dsbcoo: DSBCOO,
    row: int,
    col: int,
    form: str = "v",
    out_dtype=None,
    offset=None,
    scaling=None,
) -> NDArray | tuple:
    """Gets a block from the data structure.

    This is supposed to be a low-level method that does not perform
    any checks on the input. These are handled by the block indexer.
    The index is assumed to already be renormalized.

    Parameters
    ----------
    row : int
        Row index of the block.
    col : int
        Column index of the block.
    form : 'v' or 'm'
        form of the block.
    offset : tuple of float
        offset of the real and imaginary part.
    scaling : tuple of float
        scaling of the real and imaginary part.


    Returns
    -------
    block : NDArray | tuple[NDArray, NDArray, NDArray]
        The block at the requested index. This is an array of shape
        `(*local_stack_shape, block_sizes[row], block_sizes[col])` if
        `return_dense` is True, otherwise it is a tuple of arrays
        `(rows, cols, data)`.

    """
    data_stack = dsbcoo.data
    block_slice = dsbcoo._get_block_slice(row, col)
    # print(data_stack.shape)
    assert data_stack.shape[-2] == 2

    if out_dtype is None: 
        out_dtype = data_stack.dtype

    if scaling is None:        
        scaling = xp.array([1.0, 1.0]) 
    
    if offset is None:
        offset = xp.array([0.0, 0.0])

    if form == "v":        
        block = xp.zeros(
            data_stack.shape[:-2]
            + (int(dsbcoo.block_sizes[row]), int(dsbcoo.block_sizes[col]) * 2),
            dtype=out_dtype,
        )
    elif form == "m":
        block = xp.zeros(
            data_stack.shape[:-2]
            + (int(dsbcoo.block_sizes[row]) * 2, int(dsbcoo.block_sizes[col]) * 2),
            dtype=out_dtype,
        )
    if block_slice.start is None and block_slice.stop is None:
        # No data in this block, return an empty block.
        return block

    if form == "v":
        densify_block_v(
            block,
            dsbcoo.rows[block_slice] - dsbcoo.block_offsets[row],
            dsbcoo.cols[block_slice] - dsbcoo.block_offsets[col],
            data_stack[..., block_slice],
            scaling,
            offset
        )
    elif form == "m":
        densify_block_m(
            block,
            dsbcoo.rows[block_slice] - dsbcoo.block_offsets[row],
            dsbcoo.cols[block_slice] - dsbcoo.block_offsets[col],
            data_stack[..., block_slice],
            scaling,
            offset
        )

    return block


@profiler.profile(level="api")
def densify_block_v(block: NDArray, rows: NDArray, cols: NDArray, data: NDArray, scaling: NDArray, offset: NDArray):
    """Fills the dense block with the given data.

    Note
    ----
    This is not a raw kernel, as there seems to be no performance gain
    for this operation on the GPU.

    Parameters
    ----------
    rows : NDArray
        The rows at which to fill the block.
    cols : NDArray
        The columns at which to fill the block.
    data : NDArray
        The data to fill the block with.
    block : NDArray
        Preallocated dense block. Should be filled with zeros.

    """
    # TODO: The bare API implementation on the GPU is faster than the
    # very simple, non-general kernel i came up with. Thus, for now i
    # will just use the CuPy API directly. Since for very large blocks
    # (10'000x10'000) this starts to break even, this needs to be
    # revisited!
    (scaling_r, scaling_i) = scaling
    (offset_r,offset_i) = offset    
    block[..., rows, cols * 2] = data[..., 0, :] * scaling_r + offset_r
    block[..., rows, cols * 2 + 1] = data[..., 1, :] * scaling_i + offset_i


@profiler.profile(level="api")
def densify_block_m(block: NDArray, rows: NDArray, cols: NDArray, data: NDArray, scaling: NDArray, offset: NDArray):
    """Fills the dense block with the given data.

    Note
    ----
    This is not a raw kernel, as there seems to be no performance gain
    for this operation on the GPU.

    Parameters
    ----------
    rows : NDArray
        The rows at which to fill the block.
    cols : NDArray
        The columns at which to fill the block.
    data : NDArray
        The data to fill the block with.
    block : NDArray
        Preallocated dense block. Should be filled with zeros.

    """
    # TODO: The bare API implementation on the GPU is faster than the
    # very simple, non-general kernel i came up with. Thus, for now i
    # will just use the CuPy API directly. Since for very large blocks
    # (10'000x10'000) this starts to break even, this needs to be
    # revisited!
    block[..., rows * 2, cols * 2] = data[..., 0, :] * scaling[0] + offset[0]
    block[..., rows * 2, cols * 2 + 1] = data[..., 1, :] * scaling[1] + offset[1]
    block[..., rows * 2 + 1, cols * 2] = -data[..., 1, :] * scaling[1] + offset[1]
    block[..., rows * 2 + 1, cols * 2 + 1] = data[..., 0, :] * scaling[0] + offset[0]


@profiler.profile(level="api")
def sparsify_block_v(block: NDArray, rows: NDArray, cols: NDArray, data: NDArray, scaling: NDArray, offset: NDArray):
    """Fills the data with the given dense block.

    Note
    ----
    This is not a raw kernel, as there seems to be no performance gain
    for this operation on the GPU.

    Parameters
    ----------
    block : NDArray
        The dense block to sparsify.
    rows : NDArray
        The rows at which to fill the block.
    cols : NDArray
        The columns at which to fill the block.
    data : NDArray
        The data to be filled with the block.

    """
    # TODO: Test whether a custom kernel could be faster here.    
    # print(rows)
    # print(cols)
    data[..., 0, :] = (block[..., rows, cols * 2] - offset[0]) / scaling[0]
    data[..., 1, :] = (block[..., rows, cols * 2 + 1] - offset[1]) / scaling[1]


@profiler.profile(level="api")
def correct_out_range_index(i: int, k: int, num_blocks: int):
    # find the index of block in the matrix being repeated into open-end
    # based on the difference of row and col, ie diagonal
    diag = k - i
    k_1 = min(max(k, 0), num_blocks - 1)
    i_1 = k_1 - diag  # keep the same diag
    i_2 = min(max(i_1, 0), num_blocks - 1)
    k_2 = i_2 + diag  # keep the same diag
    return (i_2, k_2)


@profiler.profile(level="api")
def mp_bd_matmul(
    a: DSBSparse,
    b: DSBSparse,
    out: DSBSparse | None,
    in_num_diag: int = 3,
    out_num_diag: int = 5,
    spillover_correction: bool = False,
    accumulator_dtype=None,
):
    """Matrix multiplication of two `a @ b` BD DSBSparse matrices.

    Parameters
    ----------
    a : DSBSparse
        The first block diagonal matrix.
    b : DSBSparse
        The second block diagonal matrix.
    out : DSBSparse
        The output matrix. This matrix must have the same block size as
        `a` and `b`. It will compute up to `out_num_diag` diagonals.
    in_num_diag: int
        The number of diagonals in input matrices
    out_num_diag: int
        The number of diagonals in output matrices
    spillover_correction : bool, optional
        Whether to apply spillover corrections to the output matrix.
        This is necessary when the matrices represent open-ended
        systems. The default is False.
    accumulator_dtype : data type, optional
        The data type of the temporary accumulator matrices. The default is complex128.

    TODO: replace @ by appropriate gemm

    """
    if a.distribution_state == "nnz" or b.distribution_state == "nnz":
        raise ValueError(
            "Matrix multiplication is not supported for matrices in nnz distribution state."
        )
    num_blocks = len(a.block_sizes)

    if accumulator_dtype is None:
        accumulator_dtype = a.dtype

    # Make sure the output matrix is initialized to zero.
    if out is not None:
        out.data = 0
        out_block = False
    else:
        out_block = True
        out = {}

    for i in range(num_blocks):
        for j in range(
            max(i - out_num_diag // 2, 0), min(i + out_num_diag // 2 + 1, num_blocks)
        ):
            
            if out_block:
                partsum = xp.zeros(
                    (
                        a.stack_shape[:-1]
                        + tuple([int(a.block_sizes[i]), int(a.block_sizes[j]) * 2])
                    ), dtype=accumulator_dtype
                )
            else:
                partsum = (_get_block_R2C(out,i,j)).astype(accumulator_dtype)

            for k in range(i - in_num_diag // 2, i + in_num_diag // 2 + 1):                
                if abs(j - k) > in_num_diag // 2:
                    continue
                out_range = (k < 0) or (k >= num_blocks)
                if out_range and (not spillover_correction):
                    continue
                else:
                    if out_range:
                        i_a, k_a = correct_out_range_index(i, k, num_blocks)
                        k_b, j_b = correct_out_range_index(k, j, num_blocks)                        
                    else:
                        i_a, k_a = i, k 
                        k_b, j_b = k, j 
                    a_ik = _get_block_R2C(a, i_a, k_a)
                    b_kj = _get_block_R2C(b, k_b, j_b,form='m')                    
                    partsum += a_ik @ b_kj                        

            if out_block:
                out[i, j] = partsum
            else:
                # print(i,j)
                _set_block_C2R(out,i, j, partsum)

    if out_block:
        return out


@profiler.profile(level="api")
def mp_bd_sandwich(
    a: DSBSparse,
    b: DSBSparse,
    out: DSBSparse | None,
    in_num_diag: int = 3,
    out_num_diag: int = 7,
    spillover_correction: bool = False,
    accumulator_dtype=None,
):
    """Compute the sandwich product `a @ b @ a` BTD DSBSparse matrices.

    Parameters
    ----------
    a : DSBSparse
        The first block tridiagonal matrix.
    b : DSBSparse
        The second block tridiagonal matrix.
    out : DSBSparse
        The output matrix. This matrix must have the same block size as
        `a`, and `b`. It will compute up to `out_num_diag` diagonals.
    in_num_diag: int
        The number of diagonals in input matrices
    out_num_diag: int
        The number of diagonals in output matrices
    spillover_correction : bool, optional
        Whether to apply spillover corrections to the output matrix.
        This is necessary when the matrices represent open-ended
        systems. The default is False.
    accumulator_dtype : data type, optional
        The data type of the temporary accumulator matrices. The default is complex128.

    TODO: replace @ by appropriate gemm

    """
    if a.distribution_state == "nnz" or b.distribution_state == "nnz":
        raise ValueError(
            "Matrix multiplication is not supported for matrices in nnz distribution state."
        )
    num_blocks = len(a.block_sizes)

    if accumulator_dtype is None:
        accumulator_dtype = a.dtype

    # Make sure the output matrix is initialized to zero.
    if out is not None:
        out.data = 0
        out_block = False
    else:
        out_block = True
        out = {}

    for i in range(num_blocks):

        ab_ik = [None] * num_blocks * 2

        for m in range(i - in_num_diag // 2, i + in_num_diag // 2 + 1):

            out_range = (m < 0) or (m >= num_blocks)
            if out_range and (not spillover_correction):
                continue
            else:
                if out_range:
                    a_i, a_m = correct_out_range_index(i, m, num_blocks)
                else:
                    a_i, a_m = i, m

            a_im = _get_block_R2C(a, a_i, a_m) 

            for k in range(m - in_num_diag // 2, m + in_num_diag // 2 + 1):
                out_range = (k < 0) or (k >= num_blocks) or (m < 0) or (m >= num_blocks)
                if out_range and (not spillover_correction):
                    continue
                else:
                    if out_range:
                        b_m, b_k = correct_out_range_index(m, k, num_blocks)
                    else:
                        b_m, b_k = m, k

                b_mk = _get_block_R2C(b, b_m, b_k, form='m')

                if ab_ik[k] is None:                    
                    ab_ik[k] = (a_im @ b_mk).astype(
                        accumulator_dtype
                    )  # cast data type

                else:
                    ab_ik[k] += (a_im @ b_mk).astype(
                        accumulator_dtype
                    )  # cast data type

        for j in range(
            max(i - out_num_diag // 2, 0), min(i + out_num_diag // 2 + 1, num_blocks)
        ):

            if out_block:
                partsum = xp.zeros(
                    (
                        a.stack_shape[:-1]
                        + tuple([int(a.block_sizes[i]), int(a.block_sizes[j]) * 2])
                    ), dtype=accumulator_dtype
                )
            else:
                partsum = (_get_block_R2C(out,i,j)).astype(accumulator_dtype)

            for k in range(j - in_num_diag // 2, j + in_num_diag // 2 + 1):
                out_range = (k < 0) or (k >= num_blocks)
                if out_range and (not spillover_correction):
                    continue
                else:
                    if out_range:
                        a_k, a_j = correct_out_range_index(k, j, num_blocks)
                    else:
                        a_k, a_j = k, j
                
                if ab_ik[k] is None:
                    continue

                a_kj = _get_block_R2C(a, a_k, a_j, form='m')

                partsum += (ab_ik[k] @ a_kj).astype(
                    accumulator_dtype
                )  # cast data type

            if out_block:
                out[i, j] = partsum
            else:
                _set_block_C2R(out,i, j, partsum)

    if out_block:
        return out


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


    
if __name__ == "__main__":
    print("--- homemade complex tests ---")
    block_sizes = xp.array([2] * 10)
    global_stack_shape = (7, 2)
    