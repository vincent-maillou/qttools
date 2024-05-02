# Interface Specification

## Datastructures
- `DBSparse`
	- `from_sparray(self, a: sp.sparray, blocksizes: np.ndarray, stackshape=(1,), densify_blocks=None, pinned=False) -> None`
	- `to_dense(self) -> np.ndarray`
	- `zeros_like(self, a: DBSparse) -> DBSparse`
	- `block_diagonal(self, offset: int=0, dense: bool=False) -> list[sparray] | list[np.ndarray]`
	- `diagonal(self) -> np.ndarray`
	- `local_transpose(self, copy=False)`
	- `distributed_transpose(self)`
	- `__setitem__(self, idx: tuple[int, int], block: np.ndarray) -> None`
	- `__getitem__(self, idx: tuple[int, int]) -> np.ndarray | sparray`
	- `__iadd__(self, other) -> self`
	- `__imul__(self, other) -> self`
	- `__neg__(self) -> self`
	- `__matmul__(self, other) -> DBSparse`

## Green's function solver
- `Solver`
	- `selected_inv(a: DBSparse, out=None, **kwargs) -> None | DBSparse`
	- `selected_solve(a: DBSparse , sigma_lesser: DBSparse, sigma_greater: DBSparse, out: tuple | None = None, return_retarded: bool=False  **kwargs) -> None | tuple`

Example:
```python
solver = RGF(config.solver)
x = solver.selected_inv(a)
```

## Open Boundary Conditions
- `OBC`
	- `__call__(self, a_ii, a_ij, a_ji, contact, **kwargs)`

Example:
```
obc = SanchoRubio(config.obc)
g_surface = obc(a_ii, a_ij, a_ji, contact="left")
```

## Convolutions
- `fftconvolve(a: DBSparse, b: DBSparse, out=None) -> None | DBSparse`
- `fftcorrelate(a: DBSparse, b: DBSparse) -> None | DBSparse`

## Poisson solvers
- `PoissonSolver`
	- `__call__(self, density)`

Example:
```python
if config.poisson.method == "orbital":
	poisson_solver = OrbitalPoisson(config.poisson)
if config.poisson.method == "point-charge":
	poisson_solver = PointChargePoisson(config.poisson)
potential = poisson_solver(density)
```
