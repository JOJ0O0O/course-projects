"""
In this module we study the vibration equation

    u'' + w^2 u = f, t in [0, T]

where w is a constant and f(t) is a source term assumed to be 0.
We use various boundary conditions.

"""

import numpy as np
import sympy as sp
import scipy.sparse as sparse

t = sp.Symbol("t")


class VibSolver:
    """
    Solve vibration equation::

        u'' + w**2 u = f,

    """
    order: int = 0  # The order of the chosen method (VIBSolver is an abstract class, thus order=0)

    def __init__(self, Nt: int, T: float, w: float = 0.35, I: float = 1.0) -> None:
        """
        Parameters
        ----------
        Nt : int
            Number of time steps
        T : float
            End time
        I, w : float, optional
            Model parameters
        """
        self.I = I
        self.w = w
        self.T = T
        self.set_mesh(Nt)

    def set_mesh(self, Nt: int) -> None:
        """Create mesh of chose size

        Parameters
        ----------
        Nt : int
            Number of time steps
        """
        self.Nt = Nt
        self.dt = self.T / Nt
        self.t = np.linspace(0, self.T, Nt + 1)

    def ue(self) -> sp.Expr:
        """Return exact solution as sympy function"""
        return self.I * sp.cos(self.w * t)

    def u_exact(self) -> np.ndarray:
        """Exact solution of the vibration equation

        Returns
        -------
        ue : array_like
            The solution at times n*dt
        """
        return sp.lambdify(t, self.ue())(self.t)

    def l2_error(self) -> float:
        """Compute the l2 error norm of solver

        Returns
        -------
        float
            The l2 error norm
        """
        u = self()
        ue = self.u_exact()
        return np.sqrt(self.dt * np.sum((ue - u) ** 2))

    def convergence_rates(
        self, m: int = 4, N0: int = 32
    ) -> tuple[list[float], np.ndarray, np.ndarray]:
        """
        Compute convergence rate

        Parameters
        ----------
        m : int
            The number of mesh sizes used
        N0 : int
            Initial mesh size

        Returns
        -------
        r : array_like
            The m-1 computed orders
        E : array_like
            The m computed errors
        dt : array_like
            The m time step sizes
        """
        E = []
        dt = []
        self.set_mesh(N0)  # Set initial size of mesh
        for _ in range(m):
            self.set_mesh(self.Nt + 10)
            E.append(self.l2_error())
            dt.append(self.dt)
        r = [
            np.log(E[i - 1] / E[i]) / np.log(dt[i - 1] / dt[i])
            for i in range(1, m, 1)
        ]
        return r, np.array(E), np.array(dt)

    def test_order(self, m: int = 5, N0: int = 100, tol: float = 0.1) -> None:
        r, _, _ = self.convergence_rates(m, N0)
        assert abs(r[-1] - self.order) < tol

    def __call__(self) -> np.ndarray:
        """Solve vibration equation

        Returns
        -------
        u : array_like
            The solution at times n*dt, n=0,1,...,Nt
        """
        raise NotImplementedError


class VibHPL(VibSolver):
    """
    Second order accurate recursive solver

    Boundary conditions u(0)=I and u'(0)=0
    """

    order: int = 2

    def __call__(self) -> np.ndarray:
        u = np.zeros(self.Nt + 1)
        u[0] = self.I
        u[1] = u[0] - 0.5 * self.dt**2 * self.w**2 * u[0]
        for n in range(1, self.Nt):
            u[n + 1] = 2 * u[n] - u[n - 1] - self.dt**2 * self.w**2 * u[n]
        return u


class VibFD2(VibSolver):
    """
    Second order accurate solver using boundary conditions::

        u(0)=u_e(0) and u(T)=u_e(T)

    The boundary values and the right-hand side are taken from the
    manufactured solution.
    """

    order: int = 2

    def __init__(self,Nt: int,T: float,w: float = 0.35,I: float = 1.0,ue: sp.Expr = sp.exp(sp.sin(t))) -> None:
        VibSolver.__init__(self, Nt, T, w, I)
        self.ue_expr = ue

    def ue(self) -> sp.Expr:
        return self.ue_expr

    def __call__(self) -> np.ndarray:
        dt = self.T / self.Nt
        g = 2 - self.w**2 * dt**2

        A = sparse.diags(
            [1, -g, 1],
            [-1, 0, 1],
            shape=(self.Nt + 1, self.Nt + 1),
            format="lil",
        )

        ue = self.ue()
        f = sp.diff(ue, t, 2) + self.w**2 * ue
        f_num = sp.lambdify(t, f, "numpy")

        b = dt**2 * f_num(self.t)

        A[0, :3] = 1, 0, 0
        A[-1, -3:] = 0, 0, 1

        b[0] = float(ue.subs(t, 0))
        b[-1] = float(ue.subs(t, self.T))

        return sparse.linalg.spsolve(A.tocsr(), b)


class VibFD3(VibSolver):
    """
    Second order accurate solver using mixed Dirichlet and Neumann boundary
    conditions::

        u(0)=I and u'(T)=0

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """

    order: int = 2

    def __init__(self, Nt: int, T: float, w: float = 0.35, I: float = 1.0) -> None:
        VibSolver.__init__(self, Nt, T, w, I)
        T = float(T * w / np.pi)
        assert T.is_integer() and T % 2 == 0

    def __call__(self) -> np.ndarray:
        dt = self.T/self.Nt
        g = 2 - self.w**2*dt**2
        A = sparse.diags([1, -g, 1], np.array([-1, 0, 1]), (self.Nt+1, self.Nt+1), 'lil')
        b = np.zeros(self.Nt+1)
        A[0, :3] = 1, 0, 0    # Fix first row
        A[-1, -4:] = -1/3 , 3/2, -3, 11/6  # Fix last row
        b[0], b[-1] = self.I, 0
        u2 = sparse.linalg.spsolve(A.tocsr(), b)
        return u2


class VibFD4(VibFD2):
    """
    Fourth order accurate solver using boundary conditions::

        u(0)=I and u(T)=I

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """

    order: int = 4

    def ue(self) -> sp.Expr:
        return VibSolver.ue(self)

    def __call__(self) -> np.ndarray:
        #setup
        dt = self.T/self.Nt
        v = np.array([-1, 16, -30, 16, -1]) / 12
        A = sparse.diags(v, np.array([-2,-1, 0, 1,2]), (self.Nt+1, self.Nt+1), 'lil')
        b = np.zeros(self.Nt+1)

        #boundary conditions
        A[1, 0:6] = 10/12, -15/12, -4/12, 14/12, -6/12, 1/12
        A[-2, -6:] = np.array([1/12, -6/12, 14/12, -4/12, -15/12, 10/12])
        b[0], b[-1] = self.I, self.I

        B = sparse.diags([self.w**2*dt**2], np.array([0]), (self.Nt+1, self.Nt+1), 'lil')
        C = A+B

        C[0, :5] = 1, 0, 0, 0, 0    # Fix first row
        C[-1, -5:] = 0, 0, 0, 0, 1  # Fix last row

        print(C.todense())
        u2 = sparse.linalg.spsolve(C.tocsr(), b)
        return u2


def test_order():
    w = 0.35
    VibHPL(8, 2 * np.pi / w, w).test_order()
    VibFD2(8, 2 * np.pi, w, ue=sp.exp(sp.sin(t))).test_order()
    polynomial_solver = VibFD2(8, 2 * np.pi, w, ue=t**2)
    assert polynomial_solver.test_order()  # Test with polynomial solution
    VibFD3(8, 2 * np.pi / w, w).test_order()
    VibFD4(8, 2 * np.pi / w, w).test_order(N0=20, tol=0.3)


if __name__ == "__main__":
    test_order()
