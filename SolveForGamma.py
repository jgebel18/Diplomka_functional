import numpy as np
from scipy.optimize import root_scalar

from Main_Terms_In_Equations import Main_Terms_Equations
from MakeFunction_Gamma import MakeFunctions_Gamma
from Operation_with_files import Operation_with_files
from PlottingFunctions import PlottingFunctions
import os


class MakeIterationProcessforGamma:
    """
    This class manages the main iterative self-consistency process for solving
    equations involving the irreducible vertex (Gamma) and self-energy (Sigma)
    at a given inverse temperature (beta) and interaction strength (U).
    """

    def __init__(self, beta, x_values, t_value, omega_values, U, U_values,
                 NumIterations, Tolerance, Rezolution, Gamma_0, Sigma_0):

        # Core physical parameters
        self.beta = beta  # Inverse temperature (1/k_B*T)
        self.x_values = x_values  # Grid coordinates or spatial/momentum values
        self.t_value = t_value  # Quantum mechanical hopping parameter
        self.omegavalues = omega_values  # Matsubara or real frequency values for Green's functions
        self.U = U  # On-site electron interaction strength (current)
        self.U_values = U_values  # Array of various interaction strengths for scaling analysis

        # Iteration and numerical controls
        self.NumIteration = NumIterations
        self.Rezolution = Rezolution
        self.Tolerance = Tolerance

        # Visualization class
        self.PF = PlottingFunctions()

        # File paths, names, and limits for data export
        self.Gamma_0 = 0
        self.step = 0.01
        self.num_steps = 50
        self.h5py_filename = 'Data.h5'
        self.h5py_solution_filename = 'Solutions.h5'
        self.Path_Files = 'Files'
        self.Path_Images = 'Images'

        # Initial states and grids
        self.Sigma_0 = Sigma_0
        self.Gamma_values = np.linspace(-2, -0.001, self.num_steps + 1)


        self.D2_values_Sigma = None

    def Iterationprocess(self):
        """
        Executes the main self-consistent loop to find converged Self-Energy (Sigma) values.
        Stops prematurely if the convergence criteria (Tolerance) is reached.
        """
        Sigma = self.Sigma_0
        SigmaIterations = []
        SigmaValues = np.full(len(self.omegavalues), Sigma, dtype=complex)
        NewSigma = None

        for i in range(self.NumIteration):
            # Interpolate and update current state of Sigma
            Sigma = self.InterpolateOfSigmaValues(SigmaValues, self.omegavalues)
            self.Mkf.CollectDataAndPlot(Sigma)
            self.PF.PlotSigmaFunction(self.omegavalues, Sigma, SigmaValues)
            self.PrintEveryParameter(i, Sigma)
            SigmaIterations.append(Sigma)

            # Map Sigma to the vertex Gamma and generate the updated solution for Sigma
            Gamma = self.Gamma(Sigma)
            NewSigma = self.GenerateSolution(Gamma)

            # Convergence check: break if the difference falls below tolerance
            if self.CheckingforContinue(SigmaValues, NewSigma) == True:
                return NewSigma, np.array(SigmaIterations)
            else:
                SigmaValues = NewSigma  # Update target values for the next step

        return NewSigma, np.array(SigmaIterations)

    def CheckingforContinue(self, SigmaValues, NewSigmaValues):
        """
        Compares the absolute standard deviation between the current and previous
        iterations of Sigma (splitting real and imaginary paths) to evaluate convergence.
        """
        diff_real = np.std((NewSigmaValues.real - SigmaValues.real))
        diff_imag = np.std((NewSigmaValues.imag - SigmaValues.imag))
        print(f"std_real={diff_real}")
        print(f"std_imag={diff_imag}")

        diff = np.std(np.abs(NewSigmaValues - SigmaValues))
        print(f"std={diff}")

        # Check if both real and imaginary variances are within acceptable limits
        if diff_real < self.Tolerance and diff_imag < self.Tolerance:
            return True
        else:
            return False

    def SpecificateInterval(self, f, x0, x1):
        """
        Adjusts boundary limits dynamically to capture the core bracket interval
        where the target function changes signs f(x0)*f(x1) <= 0. Required for root bracket methods.
        """
        step = 100
        max_kroku = 500  # Safeguard against infinite loops

        # Happy path: The initial interval already brackets the root
        if f(x0) * f(x1) <= 0:
            return min(x0, x1), max(x0, x1)

        pocitadlo = 0

        # Scenario A: Both boundaries evaluate to negative values
        if f(x0) < 0 and f(x1) < 0:
            if abs(f(x1)) > abs(f(x0)):
                while f(x1) < 0 and pocitadlo < max_kroku:
                    x1 += step
                    pocitadlo += 1
            else:
                while f(x0) < 0 and pocitadlo < max_kroku:
                    x0 -= step
                    pocitadlo += 1

        # Scenario B: Both boundaries evaluate to positive values
        else:
            if abs(f(x1)) > abs(f(x0)):
                while f(x0) > 0 and pocitadlo < max_kroku:
                    x0 -= step
                    pocitadlo += 1
            else:
                while f(x1) > 0 and pocitadlo < max_kroku:
                    x1 += step
                    pocitadlo += 1

        # Warning fallback check if search constraints are exhausted
        if pocitadlo >= max_kroku:
            print("Warning: Interval not found, reached execution step limit.")
            return None

        return min(x0, x1), max(x0, x1)

    def GiveFinalG(self):
        """
        Computes crucial equation component metrics over a grid of beta and Gamma values,
        storing the evaluated outputs in a multi-dimensional 3D array and exporting to HDF5.
        """
        Ow_files = Operation_with_files(self.h5py_filename)
        cals = np.array(['Y(Γ)', 'D(Γ)', 'a(Γ)', 'a_Nonliear(Γ)'])

        # 3D Result array format: (4 Metrics, Size of Beta Array, Size of Gamma Array)
        res = np.empty((4, self.beta.size, self.Gamma_values.size))

        for i, beta in enumerate(self.beta):
            Terms = Main_Terms_Equations(self.beta, self.x_values, self.t_value, self.omegavalues, self.U,
                                         self.NumIteration, self.Tolerance, self.Rezolution)
            mkf = MakeFunctions_Gamma(beta, self.x_values, self.t_value, self.omegavalues, self.U)

            for j, gamma in enumerate(self.Gamma_values):
                res[0, i, j] = mkf.Y(gamma)
                res[1, i, j] = mkf.D(gamma) * gamma ** 2
                res[2, i, j] = 1 + self.U * res[0, i, j]
                res[3, i, j] = Terms.NonlinearEquationa(gamma, beta, res[0, i, j], res[1, i, j] / gamma ** 2)

        # Write each calculated matrix context out to HDF5 storage tracking keys
        for i, nazev in enumerate(cals):
            Ow_files.Write_to_file(nazev, res[i])

    def Read_Data_and_plot(self):
        """
        Reads saved data matrices back from HDF5 tracking files and triggers
        visual plotting routines to observe scaling trends.
        """
        names = np.array(['Y(Γ)', 'D(Γ)', 'a(Γ)', 'a_Nonliear(Γ)'])
        res = np.empty((4, self.beta.size, self.Gamma_values.size))
        Ow_files = Operation_with_files(self.h5py_filename)

        for i, name in enumerate(names):
            res[i] = Ow_files.Read_file(name)

        self.PF.Plot_Beta_Gamma_Dependence(res, self.beta, self.Gamma_values, self.U)
        return res

    def AnalyzeTheSolution_read(self):
        """
        Reads boundary limits and solved coefficients from HDF5 files, checks
        root solving behavior via Brent's method, and plots diagnostic results.
        """
        x0, x1, x1_new = (np.full(self.beta.size, self.Gamma_values[0], dtype=float),
                          np.full((self.beta.size), self.Gamma_values[-1], dtype=float),
                          np.empty(self.beta.size))

        for i, beta in enumerate(self.beta):
            mkf = MakeFunctions_Gamma(beta, self.x_values, self.t_value, self.omegavalues, self.U)
            terms = Main_Terms_Equations(self.beta, self.x_values, self.t_value, self.omegavalues, self.U,
                                         self.NumIteration, self.Tolerance, self.Rezolution)
            solution_1 = root_scalar(f=terms.NonlinearEquationA, args=(beta, mkf), method='brenth',
                                     maxiter=1000, xtol=1e-4, bracket=(x0[i], x1[i]))
            x1_new[i] = solution_1.root

        Ow_files = Operation_with_files(self.h5py_filename)
        Key_Values_a = Ow_files.Read_file('a_limit_values')
        Key_Values_gamma = Ow_files.Read_file('Gamma_limit_values')

        values_complete = np.vstack((Key_Values_a, Key_Values_gamma, x0, x1, x1_new))
        Result_values = self.Read_Data_and_plot()

        self.PF.Plot_Beta_Gamma_Dependence(Result_values, self.beta, self.Gamma_values, values_complete, self.U)
        return x0, x1, x1_new

    def GenerateIntervalMonteCarlo(self, function, search_interval=(-100, 0)):
        """
        A targeted stochastic Monte Carlo algorithm designed to quickly find valid initial
        boundary parameters satisfying f(x0)*f(x1) < 0 under proximity conditions.
        """
        Num_Iteration = int(1e4)
        for _ in range(Num_Iteration):
            x0, x1 = np.random.uniform(search_interval[0], search_interval[1], 2)
            print(f'Random pairs chosen:', (x0, x1))
            if x0 > x1:
                x0, x1 = x1, x0
            if (function(x0) * function(x1) < 0 and np.abs(x0 - x1) <= 50):
                print(f'Success: Root-bracketing boundaries located!')
                return x0, x1
        print('Error: Failed to isolate matching boundaries.')
        return None

    def AnalyzeTheSolution_write(self):
        """
        Solves the final non-linear system via dynamic bracketing, extracts key physical parameters
        at solved limits, and exports the structural arrays to HDF5.
        """
        x0, x1, x2 = (np.full(self.beta.size, self.Gamma_values[0], dtype=float),
                      np.full((self.beta.size), self.Gamma_values[-1], dtype=float),
                      np.empty(self.beta.size))

        Key_Values_a = np.empty((3, self.beta.size))
        Key_Values_gamma = np.empty((3, self.beta.size))

        for i, beta in enumerate(self.beta):
            mkf = MakeFunctions_Gamma(beta, self.x_values, self.t_value, self.omegavalues, self.U)
            terms = Main_Terms_Equations(self.beta, self.x_values, self.t_value, self.omegavalues, self.U,
                                         self.NumIteration, self.Tolerance, self.Rezolution)
            solution_1 = root_scalar(f=terms.a, args=(mkf), method='brenth', maxiter=1000, xtol=1e-4,
                                     bracket=(x0[i], x1[i]))
            x2[i] = solution_1.root
            print(solution_1.root)

            Key_Values_a.T[i] = np.array([terms.a(x0[i], mkf), terms.a(x1[i], mkf), terms.a(x2[i], mkf)])
            Key_Values_gamma.T[i] = np.array([terms.Gamma_function(x0[i], beta, mkf),
                                              terms.Gamma_function(x1[i], beta, mkf),
                                              terms.Gamma_function(x2[i], beta, mkf)])

        values_complete = np.vstack((Key_Values_a, Key_Values_gamma, x0, x1, x2))
        Ow_files = Operation_with_files(self.h5py_filename)
        Ow_files.Write_to_file('a_limit_values', Key_Values_a)
        Ow_files.Write_to_file('Gamma_limit_values', Key_Values_gamma)

        Result_values = self.Read_Data_and_plot()
        self.PF.Plot_Beta_Gamma_Dependence(Result_values, self.beta, self.Gamma_values, values_complete)
        return x0, x1, x2

    def SolveFinalEquation(self):
        """
        Solves the overarching final p non-linear equation  as a function of irreducible vertex using the TOMS 748 algorithm across
        combinations of temperature (beta) and interactions (U_values). Initial parameters are better to control from user
        """
        solutions = np.empty((self.beta.size, self.U_values.size))
        current_U = {}
        Op_files = Operation_with_files(self.h5py_filename)

        for i, beta in enumerate(self.beta):
            for j, U in enumerate(self.U_values):
                mkf = MakeFunctions_Gamma(beta, self.x_values, self.t_value, self.omegavalues, U)
                terms = Main_Terms_Equations(beta, self.x_values, self.t_value, self.omegavalues, U,
                                             self.NumIteration, self.Tolerance, self.Rezolution)

                x0, x1 = -4500000, -1e-5  # this is necceary to test these parameters in some first iterations
                # for higer U- depnendence rgime was these parameters  x0, x1 = (-5000, -1e-91 )
                f = lambda gamma: terms.NonlinearEquationA(gamma, beta, mkf)

                print(('x0,x1 limits of interval'),(f(x0), f(x1)))
                if f(x0) * f(x1) < 0:
                    # Optimize using TOMS 748 method for highly stable bracketing root search
                    solution_1 = root_scalar(f=terms.NonlinearEquationA, args=(beta, mkf), method='toms748',
                                             maxiter=1500, rtol=2.22045e-16, bracket=(x0, x1))
                    solutions[i][j] = solution_1.root
                    # These printing values are in super comuper saved in txt. files some of these are comented

                    # for specifical purpose of plotted variable.
                    #print(f'a-solution-for-U={U}-beta={beta}', terms.a(solutions[i][j] , mkf)*beta**2)
                    print(f'Gamma-final-for-U={U}-beta={beta}', terms.Gamma_function(solutions[i][j], beta, mkf))
                    print(f'Gamma-solution-U={U}-beta={beta}', solution_1.root)

                    print(f'U={U}-beta={beta}', solution_1.iterations)
                # for case if the requirements of solving methods are not satisfied
                else:
                    solutions[i][j] = np.nan

                current_U[f'for-U:{U}-beta-{beta}'] = solutions[i][j]
            print(f'Solutions-for-beta={beta}:', solutions[i])

        Op_files.Write_to_file(
            f'solutions-{(self.U_values.min(), self.U_values.max())}-{(self.beta.min(), self.beta.max())}', solutions)

    def PlotIntergrals(self):
        """
        Evaluates exact vs approximated forms of the core Y and D integrals
        across a tracking range of Gamma to verify analytical frameworks.
        """
        filtered = []
        Gamma_2 = np.linspace(-5, -0.1, 50)
        Mkf = MakeFunctions_Gamma(10, self.x_values, self.t_value, self.omegavalues, self.U)

        for i, gamma in enumerate(Gamma_2):
            integral_Y = Mkf.Y(gamma)
            integral_D = Mkf.D(gamma)
            integral_Y_approx = Mkf.Y_Gamma_approx_integrand(gamma)
            integral_D_approx = Mkf.D_Gamma_approx_integrand(gamma)

            print(f'Integrals Y and D for gamma {gamma}', np.array([integral_Y, integral_D]))
            print(f'Integrals Y_approx and D_approx for gamma {gamma}',
                  np.array([integral_Y_approx, integral_D_approx]))








