import numpy as np
import sympy as sp
from poisson import Poisson
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
import matplotlib.pyplot as plt

x, y = sp.symbols("x,y")

# Below we create a solver that reuses some of the implementation from
# the 1D solver in poisson.py.


class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with Dirichlet boundary conditions.
    """

    def __init__(self, Lx: float, Ly: float):
        self.px = Poisson(Lx)  # we can reuse some of the code from the 1D case
        self.py = Poisson(Ly)

    def create_mesh(self, Nx: int, Ny: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a 2D Cartesian mesh

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in x-direction
        Ny : int
            The number of uniform intervals in y-direction
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh
        """
        x = np.linspace(0, self.px.L, Nx+1)
        y = np.linspace(0, self.py.L, Ny+1)
        xij, yij = np.meshgrid(x,y, indexing = "ij")
        return xij, yij

    def laplace(self, Nx: int, Ny: int) -> sparse.lil_matrix:
        """Return a vectorized Laplace operator

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in x-direction
        Ny : int
            The number of uniform intervals in y-direction

        Returns
        -------
        A : scipy sparse LIL matrix
            The vectorized Laplace operator
        """
        xij, yij = self.create_mesh(Nx, Ny)
        dx = self.px.L/Nx
        dy = self.py.L/Ny
        D2x = self.px.D2(Nx,dx)
        D2y = self.py.D2(Ny,dy)
        return (sparse.kron(D2x, sparse.eye(Ny+1)) +
            sparse.kron(sparse.eye(Nx+1), D2y))



    def assemble(self, Nx: int, Ny: int, f: sp.Expr, ue: sp.Expr) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Return assembled coefficient matrix A and right hand side vector b

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in x-direction
        Ny : int
            The number of uniform intervals in y-direction
        f : Sympy expression
            The right hand side as a Sympy expression in x and y
        ue : Sympy expression
            The exact solution as a Sympy expression in x and y

        Returns
        -------
        A : scipy sparse CSR matrix
            Coefficient matrix
        b : 1D array
            Right hand side vector
        """
        xij, yij = self.create_mesh(Nx, Ny)
        b = sp.lambdify((x, y), f)(xij, yij)
        print(b)
        b = b.ravel()
        B = np.ones((Nx+1, Ny+1), dtype=bool)
        B[1:-1, 1:-1] = 0
        bnds = np.where(B.ravel() == 1)[0]
        ue_values = sp.lambdify((x, y), ue)(xij, yij).ravel()
        b[bnds] = ue_values[bnds]
        A = self.laplace(Nx,Ny)
        A = A.tolil()
        for i in bnds:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()
        return A, b




    def l2_error(self, u: np.ndarray, ue: sp.Expr) -> float:
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy expression
            The exact solution

        Returns
        -------
        float - The l2-error
        """
        xij, yij = self.create_mesh(u.shape[0]-1,u.shape[1]-1)
        dx = self.px.L/(u.shape[0]-1)
        dy = self.py.L/(u.shape[1]-1)
        sol = sp.lambdify((x,y),ue)(xij,yij)
        diffsq = (u-sol)*(u-sol)
        plt.contourf(xij, yij, u)
        plt.show()
        return np.sqrt(dx * dy * np.sum(diffsq))

    def __call__(self, Nx: int, Ny: int, ue: sp.Expr) -> np.ndarray:
        """Solve Poisson's equation with a given manufactured solution

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in x-direction
        Ny : int
            The number of uniform intervals in y-direction
        ue : Sympy expression
            The exact solution

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(Nx, Ny, sp.diff(ue, x, 2) + sp.diff(ue, y, 2), ue)
        return sparse_linalg.spsolve(A, b.ravel()).reshape((Nx + 1, Ny + 1))


def test_poisson2d(sol: Poisson2D, tol:float, u, ue):
    assert sol.l2_error(u, ue) < tol
    return True


if __name__ == "__main__":
    L = 2
    N = 1000
    sol = Poisson2D(Lx=L+2, Ly=L)
    ue = sp.exp(4 * sp.cos(x) + sp.cos(y))
    u = sol(N, N, ue)
    print("Manufactured solution: ", ue)
    print(f"Discretization: Nx = {N}, Ny = {N}")
    print(f"L2-error {sol.l2_error(u, ue)}")
    print("The result of the test is: ", test_poisson2d(sol=sol, tol = 1e-3, u = u, ue=ue))
