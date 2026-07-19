import numpy as np
from numpy.lib.scimath import sqrt as csqrt

# This class initializes the key equations as functions of the irreducible vertex.
# It bridges the numerical integrals and the self-energy, and serves to solve the final non-linear equation.
class Main_Terms_Equations:


    def __init__(self, beta, x_values, t_value, omega_values, U,
                 NumIterations, Tolerance, Rezolution):
        self.beta = beta  # Inverse temperature (\beta = 1/k_B T)
        self.x_values = x_values  # Spatial or energy coordinate array
        self.t_value = t_value  # Electronic hopping parameter
        self.omegavalues = omega_values  #  real-frequency grid for Green's functions
        self.U = U  # Coulomb interaction strength (U < 0 for attractive case)
        self.NumIteration = NumIterations  # Maximum allowed self-consistency loops
        self.Rezolution = Rezolution  #  Num of values
        self.Tolerance = Tolerance  # Convergence threshold

    @staticmethod
    def FermiFunction(x, beta):
        """
        Evaluates the standard Fermi-Dirac distribution function.
        Constructed exponentially to ensure numerical stability at low temperatures (high beta).
        """
        return 1 / (np.exp(beta * x) + 1)

    def a(self, Gamma, Mkf):
        """
        Calculates the reordered Kondo scale parameter 'a', which measures the distance from the superconducting state to the irreducible vertex via the function Y(Gamma).
        """
        return 1.0 + self.U * Mkf.Y(Gamma)

    def Gamma_function_a_D(self, beta, D, a):
        """
        Computes the updated value of the irreducible vertex Gamma using
        explicit, pre-calculated values of the D and a parameters.
        """
        DotofItegrals = csqrt(2 / (D * self.U))
        Gamma_function = (self.U / (beta * np.pi)) * DotofItegrals * (1 / csqrt(a))
        return Gamma_function

    def GammaNonlinearFunction(self, Gamma, beta, Mkf):
        """
        Evaluates the primary non-linear objective equation for the root-finding routine.
        """
        Gamma_function = self.Gamma_function(Gamma, beta, Mkf)
        return Gamma - Gamma_function

    def Gamma_function(self, Gamma, beta, Mkf):
        """
        Evaluates the full self-consistent non-linear functional mapping for Gamma
        by pulling current values of D(Gamma) and a(Gamma) dynamically.
        """
        DotofItegrals = csqrt(2 / (Mkf.D(Gamma) * self.U))
        Gamma_function = (self.U / (beta * np.pi)) * DotofItegrals * (1 / csqrt(self.a(Gamma, Mkf)))
        return Gamma_function

    def ZeroOmegaCase(self, Gamma):
        """
        Analytically determines the exact asymptotic roots of the self-energy
        equations specifically evaluated at the zero-frequency limit (\omega = 0).
        """
        values_real_1 = +csqrt(-0.5 + csqrt((Gamma / (2 * self.t_value) ** 2) ** 2 + 0.25))
        values_real_2 = -csqrt(-0.5 + csqrt((Gamma / (2 * self.t_value) ** 2) ** 2 + 0.25))
        values_imag_1 = +1j * csqrt(0.5 + csqrt((Gamma / (2 * self.t_value) ** 2) ** 2 + 0.25))
        values_imag_2 = -1j * csqrt(0.5 + csqrt((Gamma / (2 * self.t_value) ** 2) ** 2 + 0.25))

        result_array = np.array([values_real_1, values_real_2, values_imag_1, values_imag_2])
        print(result_array)
        return result_array

    def PrintEveryParameter(self, i, Gamma, Mkf, Sigma_0, Gamma_0):
        """
        Diagnostic utility to monitor and print out the evolution of physical parameters
        during active self-consistent iterations.
        """
        print(f"""Values of parameters in {i}-Iteration
              β = {self.beta}
              U = {self.U}
              a = {self.a(Gamma, Mkf)}
              Γ = {Gamma}
              D = {Mkf.D(Gamma)}
              Y = {Mkf.Y(Gamma)}   
              Σ_0 = {Sigma_0}
              Γ_0 = {Gamma_0}""")

    def NonlinearEquationa(self, Gamma, beta, Y, D):
        """
        Second formulation of the final non-linear equation for Gamma, utilizing
        static initial parameters to optimize numerical accessibility.
        """
        DotsOfIntegrals = 2 / (((np.pi * beta) ** 2) * D * Gamma ** 2)
        return -(1 / self.U) - Y + DotsOfIntegrals

    def NonlinearEquationA(self, Gamma, beta, Mkf):
        """
        Second formulation of the final non-linear equation as a direct function of the irreducible vertex Gamma.
        """
        DotsOfIntegrals = 2 / (((np.pi * beta) ** 2) * Mkf.D(Gamma) * Gamma ** 2)
        return -(1 / self.U) - Mkf.Y(Gamma) + DotsOfIntegrals

    def SqrtGamma_ideal(self):
        """
        Analytical high-beta (low temperature) expansion model for the
        behavior of the square root of the vertex function Gamma.
        """
        return -self.U * ((4 / (3 * np.pi)) + (15 / (4 * np.pi * self.beta ** 2 * self.t_value ** 2)))

    def a_ideal(self):
        """
        Idealized scaling behavior of parameter 'a' derived analytically
        under the strict high-interaction limit.
        """
        result = 1 - (1 / (1 + (45 / (16 * self.beta ** 2 * self.t_value ** 2))))
        return result * self.beta ** 2

    def a_Cooper(self):
        """
        Calculates the renormalization tracking factor 'a' corresponding
        to the standard Cooper instability channel.
        """
        return 1 + self.U * (1 / (2 * np.pi * self.t_value)) * np.log(self.beta * self.t_value)

    def a_approx_BCS(self):
        """
        Asymptotic approximation formula for parameter 'a' matching the
        traditional BCS thermal fluctuations regime.
        """
        return 1 + self.U * (1 / (2 * np.pi * self.t_value)) * (np.log(self.beta * self.t_value) + 1 + np.log(2))