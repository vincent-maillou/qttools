# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.


from qttools.utils.gpu_utils import xp
from qttools.datastructures.dsbsparse import DSBSparse
from qttools.greens_function_solver.solver import GFSolver
from qttools.utils.solvers_utils import get_batches


class moreRGF(GFSolver):
    def selected_inv(
        self, a: DSBSparse, out=None, max_batch_size: int = 1
    ) -> None | DSBSparse:
        """
        Perform the selected inversion of a general block sparse matrix.

        Parameters
        ----------
        a : DBSparse
            Matrix to invert.
        out : _type_, optional
            Output matrix, by default None.
        max_batch_size : int, optional
            Maximum batch size to use when inverting the matrix, by default 1.

        Returns
        -------
        None | DBSparse
            If `out` is None, returns None. Otherwise, returns the inverted matrix
            as a DBSparse object.
        """
        # Get list of batches to perform
        batches_sizes, batches_slices = get_batches(a.shape[0], max_batch_size)

        if out is not None:
            x = out
        else:
            x = a.__class__.zeros_like(a)

        for b in range(len(batches_sizes)):
            stack_slice = slice(int(batches_slices[b]), int(batches_slices[b + 1]), 1)

            a_ = a.stack[stack_slice]
            x_ = x.stack[stack_slice]

            x_.blocks[0, 0] = xp.linalg.inv(a_.blocks[0, 0])

            # Forwards sweep.
            for i in range(a.num_blocks - 1):
                for j in range(i + 1, a.num_blocks):
                    for k in range(i + 1, a.num_blocks):
                        if (a_.blocks[j,i] is not None) and (a_.blocks[i,k] is not None):
                            a_.blocks[j,k] -= a_.blocks[j,i] @ x_.blocks[i,i] @ a_.blocks[i,k]
                j = i + 1
                x_.blocks[j,j] = xp.linalg.inv(a_.blocks[j,j])

            # Backwards sweep.
            for i in range(a.num_blocks - 2, -1, -1):
                # off-diagonal blocks of invert
                for j in range(i + 1, a.num_blocks):
                    for k in range(i + 1, a.num_blocks):
                        if (
                            (a_.blocks[j,i] is not None)
                            and (x_.blocks[k,j] is not None)
                            and (x_.blocks[k,i] is not None)
                        ):
                            x_.blocks[k,i] -= x_.blocks[k,j] @ a_.blocks[j,i] @ x_.blocks[i,i]
                        if (
                            (a_.blocks[i,j] is not None)
                            and (x_.blocks[j,k] is not None)
                            and (x_.blocks[i,k] is not None)
                        ):
                            x_.blocks[i,k] -= x_.blocks[i,i] @ a_.blocks[i,j] @ x_.blocks[j,k]
                
                # diagonal blocks of invert
                tmp = xp.zeros_like(x_.blocks[i,i])
                for j in range(i + 1, a.num_blocks):
                    if (x_.blocks[j,i] is not None) and (a_.blocks[i,j] is not None):
                        tmp -= x_.blocks[i,i] @ a_.blocks[i,j] @ x_.blocks[j,i]
                x_.blocks[i,i] += tmp

        if out is None:
            return x
