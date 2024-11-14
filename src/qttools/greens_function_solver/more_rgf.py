from qttools.utils.gpu_utils import xp


def block_to_dense(A, block_size=None, dtype=xp.complex128):
    num_blocks = len(A)
    if block_size is None:
        block_size = A[0][0].shape[0]
    tmp = xp.zeros((num_blocks * block_size, num_blocks * block_size), dtype=dtype)
    for i in range(num_blocks):
        for j in range(num_blocks):
            if A[i][j] is not None:
                tmp[
                    block_size * i : block_size * (i + 1),
                    block_size * j : block_size * (j + 1),
                ] = A[i][j]
    return tmp


def random_block(
    block_size: int, num_blocks: int, num_offdiag: int = 1, complex=False, symmetry=None
):
    tmp = [[None for j in range(num_blocks)] for i in range(num_blocks)]
    for i in range(num_blocks):
        if complex:
            tmp[i][i] = xp.random.random(
                (block_size, block_size)
            ) + 1j * xp.random.random((block_size, block_size))
        else:
            tmp[i][i] = xp.random.random((block_size, block_size))

        if symmetry == "H":
            a = (tmp[i][i] + tmp[i][i].conj().T) / 2
            tmp[i][i] = a
        elif symmetry == "AH":
            a = (tmp[i][i] - tmp[i][i].conj().T) / 2
            tmp[i][i] = a

        for idiag in range(num_offdiag):
            if i < num_blocks - 1 - idiag:
                if complex:
                    tmp[i][i + 1 + idiag] = (
                        xp.random.random((block_size, block_size))
                        + 1j * xp.random.random((block_size, block_size))
                    ) * 0.5 ** (idiag + 1)

                    if symmetry == "H":
                        tmp[i + 1 + idiag][i] = (
                            (tmp[i][i + 1 + idiag]).conj().T
                        ).copy()
                    elif symmetry == "AH":
                        tmp[i + 1 + idiag][i] = -(
                            (tmp[i][i + 1 + idiag]).conj().T
                        ).copy()
                    else:
                        tmp[i + 1 + idiag][i] = (
                            xp.random.random((block_size, block_size))
                            + 1j * xp.random.random((block_size, block_size))
                        ) * 0.5 ** (idiag + 1)
                else:
                    tmp[i][i + 1 + idiag] = (
                        xp.random.random((block_size, block_size))
                    ) * 0.5 ** (idiag + 1)

                    if symmetry == "H":
                        tmp[i + 1 + idiag][i] = ((tmp[i][i + 1 + idiag]).T).copy()
                    elif symmetry == "AH":
                        tmp[i + 1 + idiag][i] = -((tmp[i][i + 1 + idiag]).T).copy()
                    else:
                        tmp[i + 1 + idiag][i] = (
                            xp.random.random((block_size, block_size))
                        ) * 0.5 ** (idiag + 1)

    return tmp


def zeros_block(
    block_size: int, num_blocks: int, num_offdiag: int = 1, dtype=xp.complex128
):
    tmp = [[None for j in range(num_blocks)] for i in range(num_blocks)]
    for i in range(num_blocks):
        tmp[i][i] = xp.zeros((block_size, block_size), dtype=dtype)
        for idiag in range(num_offdiag):
            if i < num_blocks - idiag - 1:
                tmp[i][i + idiag + 1] = xp.zeros((block_size, block_size), dtype=dtype)
                tmp[i + idiag + 1][i] = xp.zeros((block_size, block_size), dtype=dtype)
    return tmp


def block_matmul_single(alpha, A, B, block_sizes, dtype, i: int, j: int):
    tmp = xp.zeros((block_sizes[i], block_sizes[j]), dtype=dtype)
    A_num_blocks = len(A)
    for k in range(A_num_blocks):
        if (A[i][k] is not None) and ((B[k][j] is not None)):
            tmp += alpha * A[i][k] @ B[k][j]
    # if the block is not empty then return dense block, otherwise return None
    if not ((xp.abs(tmp) < 1e-30).all()):
        return tmp
    else:
        return None


def block_matmul(alpha, A, B, C, block_sizes, dtype):
    C_num_blocks = len(C)
    for i in range(C_num_blocks):
        for j in range(C_num_blocks):
            C[i][j] = block_matmul_single(alpha, A, B, block_sizes, dtype, i=i, j=j)


def duplicate_block_matrix(A):
    num_blocks = len(A)
    B = [[None for j in range(num_blocks)] for i in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(num_blocks):
            if A[i][j] is not None:
                B[i][j] = A[i][j].copy()
    return B


def hermitian_conj_block_matrix(A):
    num_blocks = len(A)
    B = [[None for j in range(num_blocks)] for i in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(num_blocks):
            if A[i][j] is not None:
                B[j][i] = A[i][j].copy().conj().T
    return B


def assign_block_matrix(A, B):
    num_blocks = len(A)
    for i in range(num_blocks):
        for j in range(num_blocks):
            if A[i][j] is not None:
                B[i][j] = xp.zeros_like(A[i][j])
                B[i][j] += A[i][j]


def block_norm(A):
    num_blocks = len(A)
    normA = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            if A[i][j] is not None:
                normA += xp.sum(xp.abs(A[i][j]))
    return normA


def block_add(A, B, alpha, beta):
    # alpha * A + beta * B -> A
    aii_blocksize = len(A)
    for ii in range(aii_blocksize):
        for jj in range(aii_blocksize):
            if A[ii][jj] is not None:
                A[ii][jj] *= alpha
                if B[ii][jj] is not None:
                    A[ii][jj] += B[ii][jj] * beta
            else:
                if B[ii][jj] is not None:
                    A[ii][jj] = xp.zeros_like(B[ii][jj])
                    A[ii][jj] += B[ii][jj] * beta


def selected_inv(a_, num_offdiag: int = 1, out=None, GN0=None, G0N=None):
    num_blocks = len(a_)
    block_size = a_[0][0].shape[0]
    dtype = a_[0][0].dtype
    if out is not None:
        x_ = out
    else:
        x_ = zeros_block(block_size, num_blocks, num_offdiag, dtype=dtype)

    x_[0][0] = xp.linalg.inv(a_[0][0])

    # Forwards sweep.
    for i in range(num_blocks - 1):
        for j in range(i + 1, num_blocks):
            for k in range(i + 1, num_blocks):
                if (a_[j][i] is not None) and (a_[i][k] is not None):
                    a_[j][k] -= a_[j][i] @ x_[i][i] @ a_[i][k]
        j = i + 1
        x_[j][j] = xp.linalg.inv(a_[j][j])

    # Backwards sweep.
    for i in range(num_blocks - 2, -1, -1):
        # off-diagonal blocks of invert
        for j in range(i + 1, num_blocks):
            for k in range(i + 1, num_blocks):
                if (
                    (a_[j][i] is not None)
                    and (x_[k][j] is not None)
                    and (x_[k][i] is not None)
                ):
                    x_[k][i] -= x_[k][j] @ a_[j][i] @ x_[i][i]
                if (
                    (a_[i][j] is not None)
                    and (x_[j][k] is not None)
                    and (x_[i][k] is not None)
                ):
                    x_[i][k] -= x_[i][i] @ a_[i][j] @ x_[j][k]

        if GN0 is not None:
            if i == num_blocks - 2:
                # $G_{N,N-1}$
                GN0 = -x_[i + 1][i + 1] @ a_[i + 1][i] @ x_[i][i]
            else:
                # $G_{N,i}$
                GN0 = -GN0 @ a_[i + 1][i] @ x_[i][i]
        if G0N is not None:
            if i == num_blocks - 2:
                # $G_{N-1,N}$
                G0N = -x_[i][i] @ a_[i][i + 1] @ x_[i + 1][i + 1]
            else:
                # $G_{i,N}$
                G0N = -x_[i][i] @ a_[i][i + 1] @ G0N

        # diagonal blocks of invert
        tmp = xp.zeros_like(x_[i][i])
        for j in range(i + 1, num_blocks):
            if (x_[j][i] is not None) and (a_[i][j] is not None):
                tmp -= x_[i][i] @ a_[i][j] @ x_[j][i]
        x_[i][i] += tmp

    if (G0N is not None) and (G0N is not None):
        return x_, G0N, GN0
    elif G0N is not None:
        return x_, G0N
    elif GN0 is not None:
        return x_, GN0
    if out is None:
        return x_


def test_selected_inv(num_blocks, block_size, num_offdiag):
    H = random_block(block_size, num_blocks, num_offdiag)
    h = block_to_dense(H)

    Gfull = xp.linalg.inv(h)

    x = selected_inv(H, num_offdiag)

    def get_block(C, bs, i, j):
        return C[bs * i : bs * (i + 1), bs * j : bs * (j + 1)]

    # test diagonal blocks
    for i in range(num_blocks):
        j = i
        g00 = get_block(Gfull, block_size, i, j)
        print(i, j, xp.allclose(x[i][j], g00))
    # test off-diagonal blocks
    for ioffdiag in range(1, num_offdiag + 1):
        for i in range(num_blocks - ioffdiag):
            j = i + ioffdiag
            g00 = get_block(Gfull, block_size, i, j)
            print(i, j, xp.allclose(x[i][j], g00))
            g00 = get_block(Gfull, block_size, j, i)
            print(j, i, xp.allclose(x[j][i], g00))


def sancho(
    a_ii,
    a_ij,
    a_ji,
    block_sizes,
    max_iterations,
    convergence_tol,
    alpha=None,
    beta=None,
    epsilon=None,
    epsilon_s=None,
    GBB=None,
    plot=False,
):
    num_blocks = len(a_ii)
    m = [[None for j in range(num_blocks)] for i in range(num_blocks)]
    if epsilon is None:
        epsilon = duplicate_block_matrix(a_ii)
    if epsilon_s is None:
        epsilon_s = duplicate_block_matrix(a_ii)
    if alpha is None:
        alpha = duplicate_block_matrix(a_ji)
    if beta is None:
        beta = duplicate_block_matrix(a_ij)

    C = [[None for j in range(num_blocks)] for i in range(num_blocks)]
    D = [[None for j in range(num_blocks)] for i in range(num_blocks)]
    dtype = xp.complex128
    num_offdiag = 1  # num_blocks - 1
    delta = float("inf")

    for iter in range(max_iterations):

        assign_block_matrix(epsilon, m)
        G0N = xp.zeros_like(a_ii[0][0])
        GN0 = xp.zeros_like(a_ii[0][0])
        inverse, G0N, GN0 = selected_inv(m, num_offdiag=num_offdiag, G0N=G0N, GN0=GN0)

        block_matmul(1, alpha, inverse, C, block_sizes, dtype)
        block_matmul(1, C, beta, D, block_sizes, dtype)

        block_add(epsilon, D, 1, -1)
        block_add(epsilon_s, D, 1, -1)

        block_matmul(1, beta, inverse, C, block_sizes, dtype)
        block_matmul(1, C, alpha, D, block_sizes, dtype)
        block_add(epsilon, D, 1, -1)

        # beta : a_{n0} = a_{n0} G_{0n} a_{n0}
        # alpha : a_{0n} = a_{0n} G_{n0} a_{0n}

        tmp = beta[-1][0] @ G0N @ beta[-1][0]
        beta[-1][0] = tmp

        tmp = alpha[0][-1] @ GN0 @ alpha[0][-1]
        alpha[0][-1] = tmp

        delta = (block_norm(alpha) + block_norm(beta)) / 2

        if delta < convergence_tol:
            break

    else:  # Did not break, i.e. max_iterations reached.
        raise RuntimeError("Surface Green's function did not converge.")

    x_ii = selected_inv(epsilon_s, num_offdiag=num_offdiag)
    print("sancho converges in ", iter)
    if GBB is not None:
        inverse = selected_inv(epsilon, num_offdiag=num_offdiag)
        GBB = inverse

    return x_ii


def nested_block_to_dense(A, level, dtype=xp.complex128):
    if A is None:
        return None, 0
    else:
        if level == 0:
            return A, A.shape[0]
        else:
            num_blocks = len(A)
            tmp = [[None for j in range(num_blocks)] for i in range(num_blocks)]
            diag_sizes = xp.zeros((num_blocks), dtype=int)
            for i in range(num_blocks):
                for j in range(num_blocks):
                    if i == j:
                        tmp[i][j], diag_sizes[i] = nested_block_to_dense(
                            A[i][j], level - 1, dtype=dtype
                        )
                    else:
                        tmp[i][j], _ = nested_block_to_dense(
                            A[i][j], level - 1, dtype=dtype
                        )

            mat_size = int(xp.sum(diag_sizes))
            mat = xp.zeros((mat_size, mat_size), dtype=dtype)
            for i in range(num_blocks):
                for j in range(num_blocks):
                    if tmp[i][j] is not None:
                        i0 = sum(diag_sizes[:i])
                        j0 = sum(diag_sizes[:j])
                        mat[i0 : i0 + diag_sizes[i], j0 : j0 + diag_sizes[j]] = tmp[i][
                            j
                        ]
            return mat, sum(diag_sizes)


def nested_selected_inv(a_, block_sizes):
    num_blocks = len(a_)

    x_ = [[None for j in range(num_blocks)] for i in range(num_blocks)]

    aii_blocksize = len(a_[0][0])

    x_[0][0] = selected_inv(a_[0][0], num_offdiag=aii_blocksize - 1)
    dtype = xp.complex128

    # Forwards sweep.
    for i in range(num_blocks - 1):
        j = i + 1
        aii_blocksize = len(a_[j][j])

        ax = [[None for j in range(aii_blocksize)] for i in range(aii_blocksize)]
        axa = [[None for j in range(aii_blocksize)] for i in range(aii_blocksize)]

        block_matmul(1, a_[j][i], x_[i][i], ax, block_sizes, dtype)
        block_matmul(1, ax, a_[i][j], axa, block_sizes, dtype)
        block_add(axa, a_[j][j], -1, 1)

        x_[j][j] = selected_inv(axa, num_offdiag=aii_blocksize - 1)

    for i in range(num_blocks - 2, -1, -1):

        j = i + 1

        x_ii = x_[i][i]
        x_jj = x_[j][j]
        a_ij = a_[i][j]

        aii_blocksize = len(a_[i][i])

        xa = [[None for j in range(aii_blocksize)] for i in range(aii_blocksize)]
        aji_xii = [[None for j in range(aii_blocksize)] for i in range(aii_blocksize)]
        xii_aij = [[None for j in range(aii_blocksize)] for i in range(aii_blocksize)]
        x_ji = [[None for j in range(aii_blocksize)] for i in range(aii_blocksize)]
        x_ij = [[None for j in range(aii_blocksize)] for i in range(aii_blocksize)]

        block_matmul(1, a_[j][i], x_ii, aji_xii, block_sizes, dtype)
        block_matmul(-1, x_jj, aji_xii, x_ji, block_sizes, dtype)
        x_[j][i] = x_ji

        block_matmul(1, x_ii, a_ij, xii_aij, block_sizes, dtype)
        block_matmul(-1, xii_aij, x_jj, x_ij, block_sizes, dtype)
        x_[i][j] = x_ij

        block_matmul(1, xii_aij, x_ji, xa, block_sizes, dtype)

        block_add(x_ii, xa, 1, -1)

    return x_
