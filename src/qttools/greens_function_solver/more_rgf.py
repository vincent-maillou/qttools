from qttools.utils.gpu_utils import xp


def block_to_dense(A,block_size = None,dtype=xp.complex128):
    num_blocks = len(A)
    if (block_size == None):
        block_size = A[0][0].shape[0]
    tmp = xp.zeros(
        (num_blocks * block_size, num_blocks * block_size), dtype=dtype
    )
    for i in range(num_blocks):
        for j in range(num_blocks):
            if A[i][j] is not None:
                tmp[
                    block_size * i : block_size * (i + 1),
                    block_size * j : block_size * (j + 1),
                ] = A[i][j]
    return tmp


def random_block(block_size: int, num_blocks: int, num_offdiag: int = 1, complex=False, symmetry=None):
    tmp = [[None for j in range(num_blocks)] for i in range(num_blocks)]
    for i in range(num_blocks):
        if complex:
            tmp[i][i] = xp.random.random(
                (block_size, block_size)
            ) + 1j * xp.random.random((block_size, block_size))
        else:
            tmp[i][i] = xp.random.random((block_size, block_size))
        
        if (symmetry == 'H'):
            a = (tmp[i][i] + tmp[i][i].conj().T) / 2 
            tmp[i][i] = a
        elif (symmetry == 'AH'): 
            a = (tmp[i][i] - tmp[i][i].conj().T) / 2
            tmp[i][i] = a 

        for idiag in range(num_offdiag):
            if i < num_blocks - 1 - idiag:
                if complex:
                    tmp[i][i + 1 + idiag] = (
                        xp.random.random((block_size, block_size))
                        + 1j * xp.random.random((block_size, block_size))
                    ) * 0.5 ** (idiag + 1)

                    if (symmetry == 'H'):
                        tmp[i + 1 + idiag][i] = ((tmp[i][i + 1 + idiag]).conj().T).copy()
                    elif (symmetry == 'AH'):
                        tmp[i + 1 + idiag][i] = - ((tmp[i][i + 1 + idiag]).conj().T).copy()
                    else:
                        tmp[i + 1 + idiag][i] = (
                            xp.random.random((block_size, block_size))
                            + 1j * xp.random.random((block_size, block_size))
                        ) * 0.5 ** (idiag + 1)
                else:
                    tmp[i][i + 1 + idiag] = (
                        xp.random.random((block_size, block_size))
                    ) * 0.5 ** (idiag + 1)

                    if (symmetry == 'H'):
                        tmp[i + 1 + idiag][i] = ((tmp[i][i + 1 + idiag]).T).copy()
                    elif (symmetry == 'AH'):
                        tmp[i + 1 + idiag][i] = - ((tmp[i][i + 1 + idiag]).T).copy()
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

def block_matmul_single(alpha,A,B,block_sizes,dtype, i:int,j:int):    
    tmp = xp.zeros((block_sizes[i],block_sizes[j]),dtype=dtype)
    A_num_blocks = len(A)
    for k in range(A_num_blocks):
        if ( ((A[i][k] is not None) and ((B[k][j] is not None))) ):
            tmp += alpha * A[i][k] @ B[k][j]
    # if the block is not empty then return dense block, otherwise return None        
    if ( not( (xp.abs(tmp) < 1e-30).all() ) ):
        return tmp
    else: 
        return None

def block_matmul(alpha,A,B,C,block_sizes,dtype):    
    C_num_blocks = len(C)
    for i in range(C_num_blocks):
        for j in range(C_num_blocks):            
            C[i][j] = block_matmul_single(alpha,A,B,block_sizes,dtype,i=i,j=j)

def duplicate_block_matrix(A):
    num_blocks = len(A)
    B = [[None for j in range(num_blocks)] for i in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(num_blocks):
            if (A[i][j] is not None):
                B[i][j] = A[i][j].copy()
    return B

def assign_block_matrix(A,B):
    num_blocks = len(A)
    for i in range(num_blocks):
        for j in range(num_blocks):
            if (A[i][j] is not None):
                B[i][j] = xp.zeros_like(A[i][j])                
                B[i][j] += A[i][j]

def block_norm(A):
    num_blocks = len(A)
    normA = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            if (A[i][j] is not None):
                normA += xp.sum(xp.abs(A[i][j]))
    return normA   

def block_add(A,B,alpha,beta):
    # alpha * A + beta * B -> A
    aii_blocksize = len(A)
    for ii in range(aii_blocksize):
        for jj in range(aii_blocksize):
            if ( (not(A[ii][jj] is None)) ):
                A[ii][jj] *= alpha
                if (not(B[ii][jj] is None)):
                    A[ii][jj] += B[ii][jj] * beta
            else:
                if (not(B[ii][jj] is None)):
                    A[ii][jj] = xp.zeros_like(B[ii][jj])
                    A[ii][jj] += B[ii][jj] * beta

def selected_inv(a_, num_offdiag: int = 1, out = None):
    num_blocks = len(a_)
    block_size = a_[0][0].shape[0]

    if (out is not None):
        x_ = out
    else:
        x_ = zeros_block(block_size, num_blocks, num_offdiag, dtype=a_[0][0].dtype)
    
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
        for j in range(i + 1, num_blocks):
            for k in range(i + 1, num_blocks):
                if ((a_[j][i] is not None) 
                    and (x_[k][j] is not None)
                    and (x_[k][i] is not None)
                ):
                    x_[k][i] -= x_[k][j] @ a_[j][i] @ x_[i][i]
                if ((a_[i][j] is not None)
                    and (x_[j][k] is not None)
                    and (x_[i][k] is not None)
                ):
                    x_[i][k] -= x_[i][i] @ a_[i][j] @ x_[j][k]

        tmp = xp.zeros_like(x_[i][i])
        for j in range(i + 1, num_blocks):
            if (x_[j][i] is not None) and (a_[i][j] is not None):
                tmp -= x_[i][i] @ a_[i][j] @ x_[j][i]
        x_[i][i] += tmp
    
    if (out is None):
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


def sancho(a_ii, a_ij, a_ji,block_sizes, max_iterations, convergence_tol, 
           alpha = None, beta = None, epsilon = None, epsilon_s = None, GBB = None):
        num_blocks = len(a_ii)
        m = [[None for j in range(num_blocks)] for i in range(num_blocks)]
        if (epsilon is None):      
            epsilon = duplicate_block_matrix(a_ii)
        if (epsilon_s is None):
            epsilon_s = duplicate_block_matrix(a_ii)
        if (alpha is None):
            alpha = duplicate_block_matrix(a_ji)
        if (beta is None):
            beta = duplicate_block_matrix(a_ij)   

        C = [[None for j in range(num_blocks)] for i in range(num_blocks)]
        D = [[None for j in range(num_blocks)] for i in range(num_blocks)]
        dtype = xp.complex128
        num_offdiag = num_blocks - 1
        delta = float("inf")
        for iter in range(max_iterations):
            
            assign_block_matrix(epsilon,m)
            inverse = selected_inv(m,num_offdiag=num_offdiag)         

            block_matmul(1,alpha,inverse,C,block_sizes,dtype)
            block_matmul(1,C,beta,D,block_sizes,dtype)
            
            block_add(epsilon,D,1,-1)
            block_add(epsilon_s,D,1,-1)

            block_matmul(1,C,alpha,D,block_sizes,dtype)
            
            block_matmul(1,beta,inverse,C,block_sizes,dtype)
            block_matmul(1,C,alpha,D,block_sizes,dtype)            
            block_add(epsilon,D,1,-1)

            block_matmul(1,C,beta,D,block_sizes,dtype)
            assign_block_matrix(D,beta)

            block_matmul(1,alpha,inverse,C,block_sizes,dtype)
            block_matmul(1,C,alpha,D,block_sizes,dtype)
            assign_block_matrix(D,alpha)

            # epsilon = epsilon - alpha @ inverse @ beta - beta @ inverse @ alpha
            # epsilon_s = epsilon_s - alpha @ inverse @ beta

            # alpha = alpha @ inverse @ alpha
            # beta = beta @ inverse @ beta            

            delta = (block_norm(alpha) + block_norm(beta)) / 2

            if delta < convergence_tol:
                break

        else:  # Did not break, i.e. max_iterations reached.
            raise RuntimeError("Surface Green's function did not converge.")

        x_ii = selected_inv(epsilon_s,num_offdiag=num_offdiag)
        # print('sancho converges in ', iter)
        if (GBB is not None):
            inverse = selected_inv(epsilon,num_offdiag=num_offdiag)
            GBB = inverse

        return x_ii