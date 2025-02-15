from qttools import NDArray, xp

if xp.__name__ == "cupy":
    import cupy as cp
    import numpy as np
    from cupy._core import _dtype
    from cupy.cuda import cublas, device


def gemm(
    a: NDArray,
    b: NDArray,
    c: None | NDArray = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    compute_type: None | str = None,
) -> NDArray | None:
    """Computes the General matrix-matrix product of three arrays: alpha * a @ b + beta * c.

    Parameters
    ----------
    a : NDArray
        The first input array.
    b : NDArray
        The second input array.
    c : None or NDArray, optional
        The third input array. If None, the result is returned in a new array.
    alpha : float, optional
        The scalar alpha, default is 1.0.
    beta : float, optional
        The scalar beta, default is 0.0.
    compute_type : None or str, optional
        The compute type to use. If None, the default compute type is used
        which is the same as the dtype of the input arrays.
        For float32 and complex64, the compute types are:
        - "32F": 32-bit floating point
        - "16F": 16-bit floating point
        - "16BF": brain float
        - "32TF": tensor float
        For float64 and complex128, the compute types are:
        - "64F": 64-bit floating point

    Returns
    -------
    c : None or NDArray if c is None
        The result of the matrix-matrix product.
    """

    if c is None and beta != 0.0:
        # else gemmEx will make a wrong result
        raise ValueError("c must be provided if beta is not zero")

    if xp.__name__ == "numpy":
        return _gemm_host(a, b, c, alpha, beta, compute_type)
    elif xp.__name__ == "cupy":
        xp.linalg._util._assert_cupy_array(a)
        xp.linalg._util._assert_stacked_2d(a)

        xp.linalg._util._assert_cupy_array(b)
        xp.linalg._util._assert_stacked_2d(b)

        if a.ndim <= 2:
            return _gemm_device(a, b, c, alpha, beta, compute_type)
        else:
            return _batched_gemm_device(a, b, c, alpha, beta, compute_type)
    else:
        raise ValueError("Invalid backend")


def _gemm_host(
    a: NDArray,
    b: NDArray,
    c: None | NDArray = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    compute_type: None | str = None,
):
    dtype = xp.common_type(a, b)

    if not (
        (
            (compute_type == "32F" or compute_type is None)
            and (dtype == xp.float32 or dtype == xp.complex64)
        )
        or (
            (compute_type == "64F" or compute_type is None)
            and (dtype == xp.float64 or dtype == xp.complex128)
        )
    ):
        raise ValueError("Invalid dtype and compute_type combination")

    if c is None:
        return alpha * xp.matmul(a, b)

    c[:] = alpha * xp.matmul(a, b) + beta * c


def _get_compute_type(dtype: xp.dtype, compute_type: str):
    if compute_type == "16F" and (dtype == xp.float32 or dtype == xp.complex64):
        return cublas.CUBLAS_COMPUTE_32F_FAST_16F
    elif compute_type == "16BF" and (dtype == xp.float32 or dtype == xp.complex64):
        return cublas.CUBLAS_COMPUTE_32F_FAST_16BF
    elif (compute_type == "32F" or compute_type is None) and (
        dtype == xp.float32 or dtype == xp.complex64
    ):
        return cublas.CUBLAS_COMPUTE_32F
    elif compute_type == "32TF" and (dtype == xp.float32 or dtype == xp.complex64):
        return cublas.CUBLAS_COMPUTE_32F_FAST_TF32
    elif (compute_type == "64F" or compute_type is None) and (
        dtype == xp.float64 or dtype == xp.complex128
    ):
        return cublas.CUBLAS_COMPUTE_64F
    else:
        raise ValueError("Invalid dtype and compute_type combination")


def _gemm_device(
    a: NDArray,
    b: NDArray,
    c: None | NDArray = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    compute_type: str = "default",
):
    # .ndim must be >= 3
    if c is None:
        dtype = xp.common_type(
            a,
            b,
        )
    else:
        dtype = xp.common_type(
            a,
            b,
            c,
        )

    if b.dtype != dtype:
        raise ValueError("b's dtype must be the same as a dtype")

    m = a.shape[-2]
    n = b.shape[-1]
    k = a.shape[-1]

    if b.shape[-2] != k:
        raise ValueError("b's shape[-2] must be equal to a's shape[-1]")

    cublas_type = _dtype.to_cuda_dtype(xp.dtype(dtype))
    cublas_compute_type = _get_compute_type(dtype, compute_type)

    alpha = np.array(alpha, dtype=dtype)
    beta = np.array(beta, dtype=dtype)

    # potential copy is needed to ensure correct memory layout
    # TODO: no memory copy, but correct strides
    a = cp.ascontiguousarray(a, dtype=dtype)
    b = cp.ascontiguousarray(b, dtype=dtype)

    # allocate c if not provided
    return_c = False
    if c is None:
        return_c = True
        c = xp.empty((m, n), dtype=dtype)

    if m == 0 or n == 0 or k == 0:
        return c

    # potential copy is needed to ensure correct memory layout
    # and to avoid memory corruption
    is_contiguous = c.flags["C_CONTIGUOUS"] or c.flags["F_CONTIGUOUS"]
    is_fortran_order = c.flags["F_CONTIGUOUS"]
    if not return_c and (is_fortran_order or not is_contiguous):
        original_c = c
        c = cp.ascontiguousarray(c, dtype=dtype)

    handle = device.get_cublas_handle()

    cublas.gemmEx(
        handle,
        cublas.CUBLAS_OP_N,
        cublas.CUBLAS_OP_N,
        n,
        m,
        k,
        alpha.ctypes.data,
        b.data.ptr,
        cublas_type,
        n,
        a.data.ptr,
        cublas_type,
        k,
        beta.ctypes.data,
        c.data.ptr,
        cublas_type,
        n,
        cublas_compute_type,
        cublas.CUBLAS_GEMM_DEFAULT,
    )

    if return_c:
        return c
    elif is_fortran_order or not is_contiguous:
        original_c[:] = c


def _batched_gemm_device(
    a: NDArray,
    b: NDArray,
    c: None | NDArray = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    compute_type: str = "default",
):
    # .ndim must be >= 3
    if c is None:
        dtype = xp.common_type(
            a,
            b,
        )
    else:
        dtype = xp.common_type(
            a,
            b,
            c,
        )

    if b.dtype != dtype:
        raise ValueError("b's dtype must be the same as a dtype")

    m = a.shape[-2]
    n = b.shape[-1]
    k = a.shape[-1]
    batchshape = a.shape[:-2]

    if b.shape[-2] != k:
        raise ValueError("b's shape[-2] must be equal to a's shape[-1]")
    if batchshape != b.shape[:-2]:
        raise ValueError("a and b must have the same batchshape")

    cublas_type = _dtype.to_cuda_dtype(xp.dtype(dtype))
    cublas_compute_type = _get_compute_type(dtype, compute_type)

    alpha = np.array(alpha, dtype=dtype)
    beta = np.array(beta, dtype=dtype)

    # potential copy is needed to ensure correct memory layout
    # TODO: no memory copy, but correct strides
    a = cp.ascontiguousarray(a, dtype=dtype)
    b = cp.ascontiguousarray(b, dtype=dtype)

    batchsize = np.prod(batchshape)

    # allocate c if not provided
    return_c = False
    if c is None:
        return_c = True
        c = xp.empty((*batchshape, m, n), dtype=dtype)

    if m == 0 or n == 0 or k == 0 or batchsize == 0:
        return c

    # potential copy is needed to ensure correct memory layout
    # and to avoid memory corruption
    is_contiguous = c.flags["C_CONTIGUOUS"] or c.flags["F_CONTIGUOUS"]
    is_fortran_order = c.flags["F_CONTIGUOUS"]
    if not return_c and (is_fortran_order or not is_contiguous):
        original_c = c
        c = cp.ascontiguousarray(c, dtype=dtype)

    handle = device.get_cublas_handle()

    cublas.gemmStridedBatchedEx(
        handle,
        cublas.CUBLAS_OP_N,
        cublas.CUBLAS_OP_N,
        n,
        m,
        k,
        alpha.ctypes.data,
        b.data.ptr,
        cublas_type,
        n,
        n * k,
        a.data.ptr,
        cublas_type,
        k,
        m * k,
        beta.ctypes.data,
        c.data.ptr,
        cublas_type,
        n,
        n * m,
        batchsize,
        cublas_compute_type,
        cublas.CUBLAS_GEMM_DEFAULT,
    )

    if return_c:
        return c
    elif is_fortran_order or not is_contiguous:
        original_c[:] = c
