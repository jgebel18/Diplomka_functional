import numpy as np
import os
from scipy.optimize import root_scalar
from PlottingFunctions import PlottingFunctions


class Nonlinear_Equation_Solver:
    """
    This class finds the numerical solution of a nonlinear equation for the
    self-energy values (Sigma) across a range of frequency (omega) values.
    """

    def __init__(self, t, omegavalues):
        self.t_value = t
        self.omega_values = omegavalues
        self.PF = PlottingFunctions()

    def gamma(self, Gamma):
        """
        Calculates the normalized irreducible vertex term (gamma).
        """
        value = -Gamma / (2 * self.t_value) ** 2
        return value

    def W(self, omega):
        """
        Calculates the normalized frequency (W) scaled by the hopping parameter t.
        """
        return omega / (2 * self.t_value)

    def NonlinearEquation(self, s, Gamma, eps, omega):
        """
        Defines the core nonlinear equation f(s) = 0 to be solved for s (Sigma).
        Includes a small imaginary adiabatic factor (eps).
        """
        W = self.W(omega) + eps * 1j
        numerator = self.gamma(Gamma)
        denominator = np.sqrt(1 + (s - (W) * 1j) ** 2)
        return s - numerator / denominator

    def NonlinearEquationDerivative(self, s, Gamma, eps, omega):
        """
        Calculates the analytical first derivative f'(s) of the nonlinear equation.
        This derivative is required for the Newton-Raphson optimization method.
        """
        W = (self.W(omega) + eps * 1j)
        numerator = -self.gamma(Gamma) * (s - 1j * W)
        denominator = (1 + (s - (W) * 1j) ** 2) ** (3 / 2)
        return (1 - numerator / denominator)

    def GiveSolutionofNonlinear(self, Gamma):
        """
        Solves the nonlinear equation for each frequency in omega_values using a fixed epsilon.
        Uses scipy.optimize.root_scalar with Newton's method.
        """
        # Pre-allocate array for complex-valued solutions
        complete_solution = np.empty(len(self.omega_values), dtype=complex)

        # Initial guess for the root solver
        omega_0 = 1.0 + 1.0 * 1j
        eps = 1e-8

        # Loop through each frequency point to find the corresponding root
        for i, omega in enumerate(self.omega_values):
            solution = root_scalar(
                f=self.NonlinearEquation,
                method='newton',
                args=(Gamma, eps, omega),
                fprime=self.NonlinearEquationDerivative,
                x0=omega_0,
                maxiter=1000
            )
            complete_solution[i] = solution.root

        # Optional: Plotting functionality
        # self.PF.PlotRootOfEquation(-1j * complete_solution * 2 * self.t_value, self.omega_values, Gamma)

        return complete_solution

    def GiveSolutionofNonlinearEps(self, Gamma):
        """
        Tests the dependence of the nonlinear equation's solution on different values of epsilon.
        Computes a 2D grid of solutions over all combinations of epsilon and omega.
        """
        # Note: self.epsilon and self.omegavalues need to be defined in __init__ for this method to run safely.
        complete_solution = np.empty((self.epsilon.size, self.omegavalues.size), dtype=complex)
        omega_0 = self.omegavalues[0]

        # Nested loop over all epsilon scaling values and frequencies
        for i, epsilon in enumerate(self.epsilon):
            for j, omega in enumerate(self.omegavalues):
                solution = root_scalar(
                    f=self.NonlinearEquation,
                    method='newton',
                    args=(Gamma, epsilon, omega),
                    fprime=self.NonlinearEquationDerivative,
                    x0=omega_0,
                    maxiter=2000,
                    xtol=1e-8
                )
                complete_solution[i][j] = solution.root

        # Optional: Plotting functionality
        # self.PF.PlotRootOfEquation(-1j * complete_solution * 2 * self.t_value, self.omega_values)

        return complete_solution