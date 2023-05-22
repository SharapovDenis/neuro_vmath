import numpy as np
import scipy


class ODESolver:

    def __init__(self, f) -> None:

        self.f = lambda t, u: np.asarray(f(t, u), float)

    def set_initial_condition(self, u0):

        if isinstance(u0, (float, int)):     # if scalar ODE
            self.neq = 1                     # num of equations
            u0 = float(u0)
        else:
            u0 = np.asarray(u0)
            self.neq = u0.size

        self.u0 = u0

    def solve(self, t_span: tuple, N: int) -> tuple:
        """Compute solution for t_span[0] <= t <= t_span[1], using N steps.

        Args:
            t_span (tuple): solution domain, i. e. t_span = (0, 2*Pi).
            N (int): number of grid points.

        Returns:
            tuple: tuple of arrays t (time grid) and u (solution on the grid).
        """

        t0, T = t_span
        self.dt = T / N
        self.t = np.zeros(N+1)              # N steps ~ N+1 time points

        if self.neq == 1:
            self.u = np.zeros(N+1)
        else:
            self.u = np.zeros((N+1, self.neq))

        self.t[0] = t0
        self.u[0] = self.u0

        for n in range(N):
            self.n = n
            self.t[n+1] = self.t[n] + self.dt
            self.u[n+1] = self.advance()

        return self.t, self.u


class ForwardEuler(ODESolver):
    """The Forward Euler is the first order approximation scheme that defined as \n
        y[n+1] = y[n] + h * f(t[n], y[n])

    Args:
        ODESolver (class): wrapper.
    """

    def advance(self):
        u, f, n, t = self.u, self.f, self.n, self.t
        dt = self.dt
        unew = u[n] + dt * f(t[n], u[n])
        return unew


class ExplicitMidpoint(ODESolver):
    """The Midpoint is the second order approximation scheme that defined as \n
        y[n+1] = y[n] + h * f(t[n] + h/2, y[n] + h/2 * f(t[n], y[n]))

    Args:
        ODESolver (class): wrapper.
    """

    def advance(self):

        u, f, n, t = self.u, self.f, self.n, self.t
        dt = self.dt
        dt2 = dt / 2.0

        k1 = f(t[n], u[n])
        k2 = f(t[n] + dt2, u[n] + dt2 * k1)

        u_next = u[n] + dt * k2

        return u_next


class RungeKutta4(ODESolver):
    """The classic Runge-Kutta method is the fourth order approximation order scheme defined as \n
        y[n+1] = y[n] + h/6 * (k1 + k2 + k3 + k4), \n
        k1 = f(t[n], y[n]), \n
        k2 = f(t[n] + h/2, y[n] + h/2 * k1), \n
        k3 = f(t[n] + h/2, y[n] + h/2 * k2), \n
        k4 = f(t[n] + h, y[n] + h * k3).

    Args:
        ODESolver (class): wrapper.
    """

    def advance(self):

        u, f, n, t = self.u, self.f, self.n, self.t
        dt = self.dt
        dt2 = dt / 2.0

        k1 = f(t[n], u[n],)
        k2 = f(t[n] + dt2, u[n] + dt2 * k1,)
        k3 = f(t[n] + dt2, u[n] + dt2 * k2,)
        k4 = f(t[n] + dt, u[n] + dt * k3,)

        u_next = u[n] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return u_next


class BDF1(ODESolver):
    """The Backward Euler scheme is the second order approximation order define as \n
        y[n+1] = y[n] + h * f(t[n+1], y[n+1])

    Args:
        ODESolver (class): wrapper.
    """

    def stage_eq(self, k):
        u, f, n, t = self.u, self.f, self.n, self.t
        dt = self.dt
        return k - f(t[n] + dt, u[n] + dt * k)

    def solve_stage(self):
        u, f, n, t = self.u, self.f, self.n, self.t
        k0 = f(t[n], u[n])
        sol = scipy.optimize.root(self.stage_eq, k0)
        return sol.x

    def advance(self):
        u, f, n, t = self.u, self.f, self.n, self.t
        dt = self.dt
        k1 = self.solve_stage()
        return u[n] + dt * k1


class CrankNicolson(BDF1):
    """The Crank Nicolson scheme is the second order approximation scheme (RK family) defined as \n
        y[n+1] = y[n] + h/2 * (k1 + k2), \n
        k1 = f(t[n], y[n]), \n
        k2 = f(t[n] + h, y[n] + h * k2).

    Args:
        BDF1 (class): BackwardEuler scheme.
    """

    def advance(self):
        u, f, n, t = self.u, self.f, self.n, self.t
        dt = self.dt

        k1 = f(t[n], u[n])
        k2 = self.solve_stage()

        return u[n] + dt / 2 * (k1 + k2)
