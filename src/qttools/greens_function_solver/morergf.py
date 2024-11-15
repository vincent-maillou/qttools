# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.


from qttools import xp
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

        x.return_None = True
        a.return_None = True

        for b in range(len(batches_sizes)):
            stack_slice = slice(int(batches_slices[b]), int(batches_slices[b + 1]), 1)

            a_ = a.stack[stack_slice]
            x_ = x.stack[stack_slice]

            x_ii = xp.linalg.inv(a_.blocks[0, 0])

            x_.blocks[0, 0] = x_ii

            # Forwards sweep.
            for i in range(a.num_blocks - 1):

                for j in range(i + 1, a.num_blocks):
                    a_ji = a_.blocks[j, i]
                    if a_ji is not None:
                        for k in range(i + 1, a.num_blocks):
                            a_ik = a_.blocks[i, k]
                            a_jk = a_.blocks[j, k]
                            if (a_ik is not None) and (a_jk is not None):
                                a_.blocks[j, k] = a_jk - a_ji @ x_ii @ a_ik

                j = i + 1
                x_ii = xp.linalg.inv(a_.blocks[j, j])
                x_.blocks[j, j] = x_ii

            # Backwards sweep.
            for i in range(a.num_blocks - 2, -1, -1):

                x_ii = x_.blocks[i, i]
                dx_ii = xp.zeros_like(x_ii)

                # off-diagonal blocks of invert
                # need to keep them as dense for the diagonal block
                # because calling the x_.blocks[k,i] will set the sparsity of a_, thus losing precision.

                for k in range(i + 1, a.num_blocks):

                    x_ki = x_.blocks[k, i]
                    x_ik = x_.blocks[i, k]

                    if (x_ki is not None) and (x_ik is not None):
                        for j in range(i + 1, a.num_blocks):

                            a_ji = a_.blocks[j, i]
                            a_ij = a_.blocks[i, j]

                            if (a_ji is not None) and (a_ij is not None):
                                x_kj = x_.blocks[k, j]
                                if x_kj is not None:
                                    x_ki -= x_kj @ a_ji @ x_ii
                                x_jk = x_.blocks[j, k]
                                if x_jk is not None:
                                    x_ik -= x_ii @ a_ij @ x_jk

                        x_.blocks[k, i] = x_ki
                        x_.blocks[i, k] = x_ik

                        a_ik = a_.blocks[i, k]
                        dx_ii += x_ii @ a_ik @ x_ki

                # diagonal blocks of invert
                x_.blocks[i, i] = x_ii - dx_ii

        if out is None:
            return x

    def selected_solve(
        self,
        a: DSBSparse,
        sigma_lesser: DSBSparse,
        sigma_greater: DSBSparse,
        out: tuple | None = None,
        return_retarded: bool = False,
        max_batch_size: int = 1,
    ) -> None | tuple:
        """Perform a selected-solve of the congruence matrix equation: A * X * A^T = B.

        Note
        ----
        If the diagonal blocks of the input matrix ```a: DSBSparse``` are not dense,
        the selected-solve will not be performed correctly. This happen because during
        the forward sweep, only the elements of the inverse that match the sparsity
        pattern of the input diagonal blocks will be stored. Leading to incomplete
        matrix-multiplications down the line.

        Parameters
        ----------
        a : DBSparse
            Matrix to invert.
        sigma_lesser : DBSparse
            Lesser matrix. This matrix is expected to be skewed-hermitian.
        sigma_greater : DBSparse
            Greater matrix. This matrix is expected to be skewed-hermitian.
        out : tuple | None, optional
            Output matrix, by default None
        return_retarded : bool, optional
            Weither the retarded Green's functioln should be returned, by default False
        max_batch_size : int, optional
            Maximum batch size to use when inverting the matrix, by default 1


        Returns
        -------
        None | tuple
            If `out` is None, returns None. Otherwise, returns the inverted matrix
            as a DBSparse object. If `return_retarded` is True, returns a tuple with
            the retarded Green's function as the last element.
        """
        raise NotImplementedError(
            "Selected solve with general sparse matrices is not implemented."
        )
