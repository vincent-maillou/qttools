from qttools.utils.gpu_utils import get_host, xp

def block_to_dense(A):
    num_blocks = len(A)
    block_size = A[0][0].shape[0]
    tmp = xp.zeros((num_blocks*block_size,num_blocks*block_size),dtype=A[0][0].dtype)
    for i in range(num_blocks):
        for j in range(num_blocks):
            if (not(A[i][j] is None)):
                tmp[block_size*i:block_size*(i+1),block_size*j:block_size*(j+1)] = A[i][j]
    return tmp

def random_block(block_size,num_blocks,num_offdiag=1):        
    tmp = [[None for j in range(num_blocks)] for i in range(num_blocks)]    
    for i in range(num_blocks):
        tmp[i][i] = xp.random.random((block_size,block_size))
        for idiag in range(num_offdiag):
            if i < num_blocks-1-idiag:
                tmp[i][i+1+idiag] = xp.random.random((block_size,block_size)) * 0.5**(idiag+1)
                tmp[i+1+idiag][i] = xp.random.random((block_size,block_size)) * 0.5**(idiag+1)
    return tmp

def zeros_block(block_size,num_blocks,num_diag=1):
    tmp = [[None for j in range(num_blocks)] for i in range(num_blocks)]   
    for i in range(num_blocks):
        tmp[i][i] = xp.zeros((block_size,block_size))
        for idiag in range(num_diag):
            if i < num_blocks-idiag-1:
                tmp[i][i+idiag+1] = xp.zeros((block_size,block_size))
                tmp[i+idiag+1][i] = xp.zeros((block_size,block_size))
    return tmp

def selected_inv(a_,num_offdiag: int = 1):
    num_blocks = len(a_)
    block_size = a_[0][0].shape[0]

    x_ = zeros_block(block_size,num_blocks,num_offdiag)
    x_[0][0] = xp.linalg.inv(a_[0][0])
    
    # Forwards sweep.
    for i in range(num_blocks - 1):
        
        for j in range(i + 1, num_blocks):
            for k in range(i + 1, num_blocks):
                if (not(a_[j][i] is None) and not(a_[i][k] is None)):
                    a_[j][k] -= a_[j][i] @ x_[i][i] @ a_[i][k] 
        j = i + 1
        x_[j][j] = xp.linalg.inv(a_[j][j])

    # Backwards sweep.
    for i in range(num_blocks - 2, -1, -1):
        
        for j in range(i + 1, num_blocks):
            if (not(a_[j][i] is None) and not(a_[i][j] is None)): 
                for k in range(i + 1, num_blocks):
                    if (not(x_[k][j] is None) and not(x_[j][k] is None) and
                        not(x_[k][i] is None) and not(x_[i][k] is None)):                                
                        x_[k][i] -= x_[k][j] @ a_[j][i] @ x_[i][i]
                        x_[i][k] -= x_[i][i] @ a_[i][j] @ x_[j][k]

        tmp = xp.zeros_like(x_[i][i])
        for j in range(i + 1, num_blocks):
            if (not(x_[j][i] is None) and not(a_[i][j] is None)): 
                tmp -= x_[i][i] @ a_[i][j] @ x_[j][i]       
        x_[i][i] += tmp
    return x_

def test_selected_inv(num_blocks,block_size,num_offdiag):    
    H = random_block(block_size,num_blocks,num_offdiag)
    h = block_to_dense(H)

    Gfull = xp.linalg.inv(h)

    x = selected_inv(H,num_offdiag)

    def get_block(C, bs, i, j):
        return(C[bs*i:bs*(i+1),bs*j:bs*(j+1)])

    # test diagonal blocks
    for i in range(num_blocks):
        j = i
        g00=get_block(Gfull, block_size, i, j)
        print(i,j, xp.allclose(x[i][j], g00))
    # test off-diagonal blocks
    for ioffdiag in range(1,num_offdiag+1):
        for i in range(num_blocks-ioffdiag):
            j = i+ioffdiag
            g00=get_block(Gfull, block_size, i, j)
            print(i,j, xp.allclose(x[i][j], g00))
            g00=get_block(Gfull, block_size, j, i)
            print(j,i, xp.allclose(x[j][i], g00))   

