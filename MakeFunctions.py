import numpy as np
import scipy.integrate as integrate
from numpy.lib.scimath import sqrt as csqrt
from PlottingFunctions import PlottingFunctions


class MakeFunctions:

    def __init__(self, beta, x_values, t_value, omega_values, U):
        self.beta = beta
        self.x_values = x_values
        self.t_value = t_value
        self.omega_values = omega_values
        self.U = U

        self.PF = PlottingFunctions()
        self.D_Iterations = []
        self.Y_Iterations = []

    # ================= BASIC FUNCTIONS =================

    def FermiFunction(self, x):
        return 1 / (np.exp(self.beta * x) + 1)

    def ComplexFrac_in_Y(self, x, Sigma):
        if np.ndim(Sigma) != 0:
            raise ValueError("Sigma must be scalar")

        denom = csqrt(4 * self.t_value**2 - (-(x) + Sigma)**2)
        return (1 / (denom * (-x + Sigma))).real

    # ================= D RATIOS =================

    def FirstDRatio(self, x, Sigma):
        return 2 * self.ComplexFrac_in_Y(x, Sigma)

    def SecondDRation(self, x, Sigma):
        denom = csqrt(4 * self.t_value**2 - (-(x) + Sigma)**2)**3
        return -2 * ((x - Sigma) / denom).real

    def ThirdDRation(self, x, Sigma):
        fp = (-x + Sigma)
        denom = (fp * csqrt(4 * self.t_value**2 - (-(x) + Sigma)**2))**3
        frac = (8 * self.t_value**2 - 3 * fp**2) / denom
        return 8 * self.t_value**2 * frac.real

    def SumOfComplexRations(self, x, Sigma):
        return (1/16) * (
            self.FirstDRatio(x, Sigma)
            + self.SecondDRation(x, Sigma)
            + self.ThirdDRation(x, Sigma)
        )

    # ================= INTEGRALS (SCALAR ONLY) =================

    def D(self, Sigma):
        def integrand(x):
            sig = Sigma(float(x)).item()
            return (1 / np.pi) * self.FermiFunction(x) * self.SumOfComplexRations(x, sig)

        result, error = integrate.quad_vec(integrand, -np.inf, np.inf)
        return result

    def Y(self, Sigma):
        def integrand(x):
            sig = Sigma(float(x)).item()
            return (1 / np.pi) * self.FermiFunction(x) * self.ComplexFrac_in_Y(x, sig)

        result, error = integrate.quad_vec(integrand, -np.inf, np.inf)
        return result

    # ================= ITERATION DATA =================

    def D_Integrand(self, Sigma):
        values = np.zeros_like(self.x_values, dtype=complex)
        for i, x in enumerate(self.x_values):
            sig = Sigma(float(x)).item()
            values[i] = (1 / np.pi) * self.FermiFunction(x) * self.SumOfComplexRations(x, sig)

        self.D_Iterations.append(values)

    def Y_Integrand(self, Sigma):
        values = np.zeros_like(self.x_values, dtype=complex)
        for i, x in enumerate(self.x_values):
            sig = Sigma((x)).item()
            values[i] = (1 / np.pi) * self.FermiFunction(x) * self.ComplexFrac_in_Y(x, sig)

        self.Y_Iterations.append(values)

    # ================= GREEN FUNCTION =================

    def G(self, omega, Sigma):
        sig = Sigma(float(omega))
        denom = csqrt(4 * self.t_value**2 - ((-omega + sig)**2))
        return (-1j / denom)

    # ================= PLOTTING =================

    def CollectDataAndPlot(self, Sigma):
        self.Y_Integrand(Sigma)
        self.D_Integrand(Sigma)
        D_array = np.vstack(self.D_Iterations)
        Y_array = np.vstack(self.Y_Iterations)
        self.PF.PlotingItegrands(D_array, Y_array, self.x_values)
