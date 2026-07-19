import numpy as np
import scipy.integrate as integrate
from numpy.lib.scimath import sqrt as csqrt
from PlottingFunctions import PlottingFunctions
from scipy.special import expit
from scipy.interpolate import PchipInterpolator
from Nonlinear_Equation_Sigma_Solver import Nonlinear_Equation_Solver


class MakeFunctions_Gamma:
    """
    Constructs, interpolates, and integrates the complex many-body kernel elements
    required to evaluate the irreducible vertex (Gamma) and self-energy (Sigma).
    Provides both robust adaptive numerical quadrature over the Green's function
    cut structure and explicit analytical asymptotic forms.
    """

    def __init__(self, beta, x_values, t_value, omega_values, U):
        self.beta = beta  # Inverse temperature parameter
        self.x_values = x_values  # Spatial or energy coordinate array grid
        self.t_value = t_value  # Electronic hopping integral amplitude
        self.omega_values = omega_values  # Master Matsubara/real-frequency grid
        self.U = U  # Microscopic Coulomb interaction strength
        self.PF = PlottingFunctions()  # Visualization utility handle
        self.D_Iterations = []  # Iteration history tracker for the D-kernel integrands
        self.Y_Iterations = []  # Iteration history tracker for the Y-kernel integrands
        self.Nonlinear_Solver = Nonlinear_Equation_Solver(self.t_value, self.omega_values)

    # ================= BASIC FUNCTIONS =================

    def FermiFunction(self, x):
        """
        Evaluates the Fermi-Dirac distribution function utilizing a numerically stable,
        exponentially bounded logistic sigmoid wrapper.
        """
        return expit(-self.beta * x)

    def Sigma_Interpolation(self, Sigma_values):
        """
        Constructs a shape-preserving, complex interpolation profile for the self-energy
        across the frequency grid using SciPy's Piecewise Cubic Hermite Interpolating Polynomials (PCHIP).
        """
        Sigma_complex_inter = PchipInterpolator(self.omega_values, Sigma_values.imag)
        Sigma_real_inter = PchipInterpolator(self.omega_values, Sigma_values.real)
        Sigma = lambda x: Sigma_real_inter(x) + 1j * Sigma_complex_inter(x)
        return Sigma

    def Sigma(self, Gamma):
        """
        Maps a given irreducible vertex Gamma to its self-consistent self-energy profile
        by solving the underlying nonlinear equations and wrapping the array results
        into a smooth, continuous interpolator.
        """
        values = self.Nonlinear_Solver.GiveSolutionofNonlinear(Gamma)
        Sigma_values = -1j * values * 2 * self.t_value
        Sigma = self.Sigma_Interpolation(Sigma_values)
        return Sigma

    def ComplexFrac_in_Y(self, x, Sigma):
        """
        Evaluates the structural core fraction of the Y-integral kernel.
        Guarantees scalar stability within the square-root branch cut across bands.
        """
        if np.ndim(Sigma) != 0:
            raise ValueError("Sigma must be scalar")

        Sigma_broad = Sigma
        denom = csqrt(4 * self.t_value ** 2 - (-x + Sigma_broad) ** 2)
        return (1 / (denom * (-x + Sigma_broad))).real

    # ================= D RATIOS =================

    def FirstDRatio(self, x, Sigma):
        """
        Computes the first analytical component of the momentum-dependent term,
        derived from the series expansion of the Kondo scale equations.
        """
        return 2 * self.ComplexFrac_in_Y(x, Sigma)

    def SecondDRatio(self, x, Sigma):
        """
        Computes the second analytical component of the momentum-dependent term
        within the series expansion of the reordered Kondo scale parameter 'a'.
        """
        Sigma_broad = Sigma
        denom = csqrt(4 * self.t_value ** 2 - (-x + Sigma_broad) ** 2) ** 3
        return -2 * ((x - Sigma_broad) / denom).real

    def ThirdDRatio(self, x, Sigma):
        """
        Computes the third analytical component of the decomposed D-kernel system.
        Captures high-order electronic scattering processes within the band edge.
        """
        Sigma_broad = Sigma
        fp = (-x + Sigma_broad)
        denom = (fp * csqrt(4 * self.t_value ** 2 - (-x + Sigma_broad) ** 2)) ** 3
        frac = (8 * self.t_value ** 2 - 3 * fp ** 2) / denom
        return 8 * self.t_value ** 2 * frac.real

    def SumOfComplexRatios(self, x, Sigma):
        """
        Combines the analytical expansion ratios to establish the full momentum-dependent
        master integrand within the series expansion of the Kondo scale parameter 'a'.
        """
        return (1 / 16) * (
                self.FirstDRatio(x, Sigma)
                + self.SecondDRatio(x, Sigma)
                + self.ThirdDRatio(x, Sigma)
        )

    # ================= INTEGRATOR=================

    def Integrator(self, Gamma, function):
        """
        Executes a high-precision, vector-adaptive quadrature (quad_vec) over the
        infinite real frequency axis, capturing sharp internal resonance structures.
        """
        Sigma_func = self.Sigma(Gamma)

        def integrand(x):
            sig = Sigma_func(float(x)).item()
            return (1 / np.pi) * self.FermiFunction(x) * function(x, sig)

        result, error = integrate.quad_vec(
            integrand,
            -np.inf,
            np.inf,
            limit=5000,
            epsabs=1e-10,
            epsrel=1e-6
        )
        return result

    # ================= EVALUATION CHANNELS =================

    def D(self, Gamma):
        """
        Evaluates the self-consistent, full-axis integrated value of the momentum-dependent
        renormalization term in the Kondo scale 'a' equations.
        """
        return self.Integrator(Gamma, self.SumOfComplexRatios)

    def Y(self, Gamma):
        """
        Evaluates the self-consistent, full-axis frequency integral of the
        standard electron-electron polarization bubble.
        """
        return self.Integrator(Gamma, self.ComplexFrac_in_Y)

    def Y_approx(self, Gamma):
        """
        Evaluates the clean analytical approximation for the Y-parameter,
        serving as a verification anchor against numerical solutions.
        """
        denom = -Gamma / (2 * self.t_value) ** 2
        first_term = 1 / (2 * np.pi * self.t_value)
        numerator = np.sqrt(1 + (denom ** 2)) + 1
        return first_term * np.log(numerator / denom)

    def D_approx(self, Gamma):
        """
        Evaluates the analytical high-order approximation for the D-parameter,
        capturing the essential logarithmic divergences.
        """
        const_in_front = 1 / (16 * np.pi * self.t_value)
        denom = -Gamma / (2 * self.t_value) ** 2
        first_term = np.log((np.sqrt(1 + (Gamma / (2 * self.t_value) ** 2) ** 2) + 1) / (denom))
        second_term = np.sqrt(1 + (Gamma / (2 * self.t_value) ** 2) ** 2) / denom ** 2
        result = const_in_front * (first_term - second_term)
        return result * Gamma ** 2

    def D_Integrand(self, Gamma):
        """
        Computes and caches the discrete D-channel integrand profile across the spatial/energy grid
        for visualization and diagnostic mapping.
        """
        Sigma = self.Sigma(Gamma)
        values = np.zeros_like(self.x_values, dtype=complex)
        for i, x in enumerate(self.x_values):
            sig = Sigma(float(x)).item()
            values[i] = (1 / np.pi) * self.FermiFunction(x) * self.SumOfComplexRatios(x, sig)
        self.D_Iterations.append(values)

    def Y_Integrand(self, Gamma):
        """
        Computes and caches the discrete Y-channel integrand profile across the spatial/energy grid
        for visualization and diagnostic mapping.
        """
        Sigma = self.Sigma(Gamma)
        values = np.zeros_like(self.x_values, dtype=complex)
        for i, x in enumerate(self.x_values):
            sig = Sigma(x).item()
            values[i] = (1 / np.pi) * self.FermiFunction(x) * self.ComplexFrac_in_Y(x, sig)
        self.Y_Iterations.append(values)

    def G(self, omega, Sigma):
        """
        Evaluates the standalone single-particle Green's function propagator.
        """
        sig = Sigma(float(omega))
        denom = csqrt(4 * self.t_value ** 2 - ((-omega + sig) ** 2))
        return (-1j / denom)

    def CollectDataAndPlot(self, Gamma):
        """
        Triggers explicit evaluation of internal integrands and passes compiled matrix
        profiles directly into the visualization suite.
        """
        self.Y_Integrand(Gamma)
        self.D_Integrand(Gamma)
        D_array = np.vstack(self.D_Iterations)
        Y_array = np.vstack(self.Y_Iterations)
        self.PF.PlottingItegrands(D_array, Y_array, self.x_values)

    def Y_Gamma_approx_integrand(self, Gamma):
        """
        Asymptotic scaling form of the Y-integrand core under high-vertex constraints.
        """
        return 4 / (3 * np.pi * np.sqrt(-Gamma))

    def D_Gamma_approx_integrand(self, Gamma):
        """
        Asymptotic scaling form of the D-integrand core under high-vertex constraints.
        """
        return -8 * self.t_value ** 2 / (15 * np.pi) * (1 / (np.sqrt(-Gamma)) ** 3)