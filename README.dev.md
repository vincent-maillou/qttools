# Interface Specification

## Datastructures
- `DBSparse`
	- `from_sparray(sparray, blocksizes: np.ndarray, stackshape=(1,), densify=None, pinned=False)`
	- `to_dense() -> np.ndarray`
	- `zeros_like(dbsparse) -> DBSparse`
	- `get_block(i, j, dense=False) -> np.ndarray`
	- `set_block(i, j, block: np.ndarray)`
	- `block_diagonal(offset, dense=False) -> list[sparray] | list[np.ndarray]`
	- `diagonal() -> np.ndarray`
	- `local_transpose(copy=False)`
	- `distributed_transpose()`
	- `__iadd__(self, other)`
	- `__imul__(self, other)`
	- `__neg__(self)`


## `Solver`
- `selected_inv(a: DBSparse, out=None, **kwargs) -> None | DBSparse`
- `selected_solve(a: DBSparse , sigma_lesser: DBSparse, sigma_greater: DBSparse, out: tuple | None = None, return_retarded: bool=False  **kwargs) -> None | tuple`

Example:
```
solver = RGF(config.solver)
x = solver.selected_inv(a)
```

## `OBC`
- `__call__(self, a_ii, a_ij, a_ji, contact, **kwargs)`

Example:
```
obc = SanchoRubio(config.obc)
g_surface = obc(a_ii, a_ij, a_ji, contact="left")
```

## Convolutions
- `fftconvolve(a: DBSparse, b: DBSparse, out=None) -> None | DBSparse`
- `fftcorrelate(a: DBSparse, b: DBSparse) -> None | DBSparse`

