from typing import Tuple

import numpy as np
import pytest

from qttools import xp
from qttools.kernels import gemm


@pytest.mark.usefixtures("m_n_k", "dtype_compute_type", "order")
def test_gemm(
    m_n_k: Tuple[int, int, int], dtype_compute_type: Tuple[xp.dtype, str], order: str
):
    m, n, k = m_n_k
    dtype, compute_type = dtype_compute_type
    real_dtype = np.dtype(dtype).type(0).real.dtype

    # certain compute types not allowed with numpy
    if xp.__name__ == "numpy":
        if compute_type in ["16F", "16BF", "32TF"]:
            return

    rng = xp.random.default_rng(seed=1)
    if dtype in [xp.complex64, xp.complex128]:
        a = rng.random((m, k), dtype=real_dtype) + 1j * rng.random(
            (m, k), dtype=real_dtype
        )
        b = rng.random((k, n), dtype=real_dtype) + 1j * rng.random(
            (k, n), dtype=real_dtype
        )
    elif dtype in [xp.float32, xp.float64]:
        a = rng.random((m, k), dtype=real_dtype)
        b = rng.random((k, n), dtype=real_dtype)
    else:
        raise ValueError("Invalid dtype")

    if order == "F":
        a = xp.asfortranarray(a)
        b = xp.asfortranarray(b)
    elif order == "C":
        a = xp.ascontiguousarray(a)
        b = xp.ascontiguousarray(b)
    else:
        raise ValueError("Invalid order")

    a_copy = a.copy()
    b_copy = b.copy()

    alpha = np.random.randn(1)[0]

    c = gemm(a, b, alpha=alpha, compute_type=compute_type)

    c_ref = alpha * a @ b

    # NOTE: tolerance needs to be quite high for float32 and complex64
    if dtype in [xp.float32, xp.complex64]:
        reltol = 1e-6
    else:
        reltol = 1e-14

    assert xp.allclose(a, a_copy)
    assert xp.allclose(b, b_copy)
    assert xp.linalg.norm(c - c_ref) / xp.linalg.norm(c_ref) < reltol


@pytest.mark.usefixtures("m_n_k", "dtype_compute_type", "order")
def test_gemm_inplace(
    m_n_k: Tuple[int, int, int], dtype_compute_type: Tuple[xp.dtype, str], order: str
):
    m, n, k = m_n_k
    dtype, compute_type = dtype_compute_type
    real_dtype = np.dtype(dtype).type(0).real.dtype

    # certain compute types not allowed with numpy
    if xp.__name__ == "numpy":
        if compute_type in ["16F", "16BF", "32TF"]:
            return

    rng = xp.random.default_rng(seed=1)
    if dtype in [xp.complex64, xp.complex128]:
        a = rng.random((m, k), dtype=real_dtype) + 1j * rng.random(
            (m, k), dtype=real_dtype
        )
        b = rng.random((k, n), dtype=real_dtype) + 1j * rng.random(
            (k, n), dtype=real_dtype
        )
        c = rng.random((m, n), dtype=real_dtype) + 1j * rng.random(
            (m, n), dtype=real_dtype
        )
    elif dtype in [xp.float32, xp.float64]:
        a = rng.random((m, k), dtype=real_dtype)
        b = rng.random((k, n), dtype=real_dtype)
        c = rng.random((m, n), dtype=real_dtype)
    else:
        raise ValueError("Invalid dtype")

    if order == "F":
        a = xp.asfortranarray(a)
        b = xp.asfortranarray(b)
        c = xp.asfortranarray(c)
    elif order == "C":
        a = xp.ascontiguousarray(a)
        b = xp.ascontiguousarray(b)
        c = xp.ascontiguousarray(c)
    else:
        raise ValueError("Invalid order")

    a_copy = a.copy()
    b_copy = b.copy()
    c_copy = c.copy()

    alpha = np.random.randn(1)[0]
    beta = np.random.randn(1)[0]

    gemm(a, b, c=c, alpha=alpha, beta=beta, compute_type=compute_type)

    c_ref = alpha * a @ b + beta * c_copy

    # NOTE: tolerance needs to be quite high for float32 and complex64
    if dtype in [xp.float32, xp.complex64]:
        reltol = 1e-6
    else:
        reltol = 1e-14

    assert xp.allclose(a, a_copy)
    assert xp.allclose(b, b_copy)
    assert xp.linalg.norm(c - c_ref) / xp.linalg.norm(c_ref) < reltol


@pytest.mark.usefixtures("m_n_k", "dtype_compute_type", "order")
def test_gemm_noncontiguous(
    m_n_k: Tuple[int, int, int], dtype_compute_type: Tuple[xp.dtype, str], order: str
):
    m, n, k = m_n_k
    dtype, compute_type = dtype_compute_type
    real_dtype = np.dtype(dtype).type(0).real.dtype

    # certain compute types not allowed with numpy
    if xp.__name__ == "numpy":
        if compute_type in ["16F", "16BF", "32TF"]:
            return

    rng = xp.random.default_rng(seed=1)
    if dtype in [xp.complex64, xp.complex128]:
        a = rng.random((2 * m, 2 * k), dtype=real_dtype) + 1j * rng.random(
            (2 * m, 2 * k), dtype=real_dtype
        )
        b = rng.random((2 * k, 2 * n), dtype=real_dtype) + 1j * rng.random(
            (2 * k, 2 * n), dtype=real_dtype
        )
        c = rng.random((2 * m, 2 * n), dtype=real_dtype) + 1j * rng.random(
            (2 * m, 2 * n), dtype=real_dtype
        )
    elif dtype in [xp.float32, xp.float64]:
        a = rng.random((2 * m, 2 * k), dtype=real_dtype)
        b = rng.random((2 * k, 2 * n), dtype=real_dtype)
        c = rng.random((2 * m, 2 * n), dtype=real_dtype)
    else:
        raise ValueError("Invalid dtype")

    if order == "F":
        a = xp.asfortranarray(a)
        b = xp.asfortranarray(b)
        c = xp.asfortranarray(c)
    elif order == "C":
        a = xp.ascontiguousarray(a)
        b = xp.ascontiguousarray(b)
        c = xp.ascontiguousarray(c)
    else:
        raise ValueError("Invalid order")

    a_copy = a.copy()
    b_copy = b.copy()
    c_copy = c.copy()

    alpha = np.random.randn(1)[0]
    beta = np.random.randn(1)[0]

    gemm(
        a[:m, :k],
        b[:k, :n],
        c=c[:m, :n],
        alpha=alpha,
        beta=beta,
        compute_type=compute_type,
    )

    c_ref = alpha * a[:m, :k] @ b[:k, :n] + beta * c_copy[:m, :n]

    # NOTE: tolerance needs to be quite high for float32 and complex64
    if dtype in [xp.float32, xp.complex64]:
        reltol = 1e-6
    else:
        reltol = 1e-14

    assert xp.allclose(a, a_copy)
    assert xp.allclose(b, b_copy)
    assert xp.linalg.norm(c[:m, :n] - c_ref) / xp.linalg.norm(c_ref) < reltol
    assert xp.allclose(c[:m, n:], c_copy[:m, n:])
    assert xp.allclose(c[m:, :n], c_copy[m:, :n])
    assert xp.allclose(c[m:, n:], c_copy[m:, n:])


@pytest.mark.usefixtures("m_n_k", "batchshape", "dtype_compute_type", "order")
def test_gemm_batched(
    m_n_k: Tuple[int, int, int],
    batchshape: int,
    dtype_compute_type: Tuple[xp.dtype, str],
    order: str,
):
    m, n, k = m_n_k
    dtype, compute_type = dtype_compute_type
    real_dtype = np.dtype(dtype).type(0).real.dtype

    # certain compute types not allowed with numpy
    if xp.__name__ == "numpy":
        if compute_type in ["16F", "16BF", "32TF"]:
            return

    rng = xp.random.default_rng(seed=1)
    if dtype in [xp.complex64, xp.complex128]:
        a = rng.random((*batchshape, m, k), dtype=real_dtype) + 1j * rng.random(
            (*batchshape, m, k), dtype=real_dtype
        )
        b = rng.random((*batchshape, k, n), dtype=real_dtype) + 1j * rng.random(
            (*batchshape, k, n), dtype=real_dtype
        )
    elif dtype in [xp.float32, xp.float64]:
        a = rng.random((*batchshape, m, k), dtype=real_dtype)
        b = rng.random((*batchshape, k, n), dtype=real_dtype)
    else:
        raise ValueError("Invalid dtype")

    if order == "F":
        a = xp.asfortranarray(a)
        b = xp.asfortranarray(b)
    elif order == "C":
        a = xp.ascontiguousarray(a)
        b = xp.ascontiguousarray(b)
    else:
        raise ValueError("Invalid order")

    a_copy = a.copy()
    b_copy = b.copy()

    alpha = np.random.randn(1)[0]

    c = gemm(a, b, alpha=alpha, compute_type=compute_type)

    c_ref = alpha * a @ b
    # NOTE: tolerance needs to be quite high for float32 and complex64
    if dtype in [xp.float32, xp.complex64]:
        reltol = 1e-6
    else:
        reltol = 1e-14

    print(a.shape)
    print(a_copy.shape)

    assert xp.allclose(a, a_copy)
    assert xp.allclose(b, b_copy)
    assert xp.linalg.norm(c - c_ref) / xp.linalg.norm(c_ref) < reltol


@pytest.mark.usefixtures("m_n_k", "batchshape", "dtype_compute_type", "order")
def test_gemm_batched_inplace(
    m_n_k: Tuple[int, int, int],
    batchshape: Tuple[int],
    dtype_compute_type: Tuple[xp.dtype, str],
    order: str,
):
    m, n, k = m_n_k
    dtype, compute_type = dtype_compute_type
    real_dtype = np.dtype(dtype).type(0).real.dtype

    # certain compute types not allowed with numpy
    if xp.__name__ == "numpy":
        if compute_type in ["16F", "16BF", "32TF"]:
            return

    rng = xp.random.default_rng(seed=1)
    if dtype in [xp.complex64, xp.complex128]:
        a = rng.random((*batchshape, m, k), dtype=real_dtype) + 1j * rng.random(
            (*batchshape, m, k), dtype=real_dtype
        )
        b = rng.random((*batchshape, k, n), dtype=real_dtype) + 1j * rng.random(
            (*batchshape, k, n), dtype=real_dtype
        )
        c = rng.random((*batchshape, m, n), dtype=real_dtype) + 1j * rng.random(
            (*batchshape, m, n), dtype=real_dtype
        )
    elif dtype in [xp.float32, xp.float64]:
        a = rng.random((*batchshape, m, k), dtype=real_dtype)
        b = rng.random((*batchshape, k, n), dtype=real_dtype)
        c = rng.random((*batchshape, m, n), dtype=real_dtype)
    else:
        raise ValueError("Invalid dtype")

    if order == "F":
        a = xp.asfortranarray(a)
        b = xp.asfortranarray(b)
        c = xp.asfortranarray(c)
    elif order == "C":
        a = xp.ascontiguousarray(a)
        b = xp.ascontiguousarray(b)
        c = xp.ascontiguousarray(c)
    else:
        raise ValueError("Invalid order")

    a_copy = a.copy()
    b_copy = b.copy()
    c_copy = c.copy()

    alpha = np.random.randn(1)[0]
    beta = np.random.randn(1)[0]

    gemm(a, b, c=c, alpha=alpha, beta=beta, compute_type=compute_type)

    c_ref = alpha * a @ b + beta * c_copy

    # NOTE: tolerance needs to be quite high for float32 and complex64
    if dtype in [xp.float32, xp.complex64]:
        reltol = 1e-6
    else:
        reltol = 1e-14

    assert xp.allclose(a, a_copy)
    assert xp.allclose(b, b_copy)
    assert xp.linalg.norm(c - c_ref) / xp.linalg.norm(c_ref) < reltol


@pytest.mark.usefixtures("m_n_k", "batchshape", "dtype_compute_type", "order")
def test_gemm_batched_noncontiguous(
    m_n_k: Tuple[int, int, int],
    batchshape: Tuple[int],
    dtype_compute_type: Tuple[xp.dtype, str],
    order: str,
):
    m, n, k = m_n_k
    dtype, compute_type = dtype_compute_type
    real_dtype = np.dtype(dtype).type(0).real.dtype

    # certain compute types not allowed with numpy
    if xp.__name__ == "numpy":
        if compute_type in ["16F", "16BF", "32TF"]:
            return

    rng = xp.random.default_rng(seed=1)
    if dtype in [xp.complex64, xp.complex128]:
        a = rng.random((*batchshape, 2 * m, 2 * k), dtype=real_dtype) + 1j * rng.random(
            (*batchshape, 2 * m, 2 * k), dtype=real_dtype
        )
        b = rng.random((*batchshape, 2 * k, 2 * n), dtype=real_dtype) + 1j * rng.random(
            (*batchshape, 2 * k, 2 * n), dtype=real_dtype
        )
        c = rng.random((*batchshape, 2 * m, 2 * n), dtype=real_dtype) + 1j * rng.random(
            (*batchshape, 2 * m, 2 * n), dtype=real_dtype
        )
    elif dtype in [xp.float32, xp.float64]:
        a = rng.random((*batchshape, 2 * m, 2 * k), dtype=real_dtype)
        b = rng.random((*batchshape, 2 * k, 2 * n), dtype=real_dtype)
        c = rng.random((*batchshape, 2 * m, 2 * n), dtype=real_dtype)
    else:
        raise ValueError("Invalid dtype")

    if order == "F":
        a = xp.asfortranarray(a)
        b = xp.asfortranarray(b)
        c = xp.asfortranarray(c)
    elif order == "C":
        a = xp.ascontiguousarray(a)
        b = xp.ascontiguousarray(b)
        c = xp.ascontiguousarray(c)
    else:
        raise ValueError("Invalid order")

    a_copy = a.copy()
    b_copy = b.copy()
    c_copy = c.copy()

    alpha = np.random.randn(1)[0]
    beta = np.random.randn(1)[0]

    gemm(
        a[..., :m, :k],
        b[..., :k, :n],
        c=c[..., :m, :n],
        alpha=alpha,
        beta=beta,
        compute_type=compute_type,
    )

    c_ref = alpha * a[..., :m, :k] @ b[..., :k, :n] + beta * c_copy[..., :m, :n]

    # NOTE: tolerance needs to be quite high for float32 and complex64
    if dtype in [xp.float32, xp.complex64]:
        reltol = 1e-6
    else:
        reltol = 1e-14

    assert xp.allclose(a, a_copy)
    assert xp.allclose(b, b_copy)
    assert xp.linalg.norm(c[..., :m, :n] - c_ref) / xp.linalg.norm(c_ref) < reltol
    assert xp.allclose(c[..., :m, n:], c_copy[..., :m, n:])
    assert xp.allclose(c[..., m:, :n], c_copy[..., m:, :n])
    assert xp.allclose(c[..., m:, n:], c_copy[..., m:, n:])
