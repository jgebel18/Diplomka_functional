import numpy as np
import matplotlib

import Read_files_SuperComputer
from SolveForGamma import MakeIterationProcessforGamma
from Read_files_SuperComputer import ReadDataFilesFromSuperComputer
from multiprocessing import Pool


# Zero iteration of self-energy
def ZerothIterationofSelfEnergy(U, t):
    Sigma = 1j * 3 * Gamma_0(U, t) / (2 * t)
    return Sigma


# Gamma_0 (Zeroth iteration of Gamma)
def Gamma_0(U, t):
    parameter = (2 * np.pi * t) / U
    gamma_0 = -(2 * t) ** 2 * np.exp(parameter)
    # print(gamma_0)
    return gamma_0


def NewGamma_0(U, t):
    return 1 / np.sinh((2 * np.pi * t) / (-U)) * (-(2 * t) ** 2)


# This is the main script for the initialization of the initial simulation parameters,
# where the chosen values are key to ensuring the correct behavior of the simulation.
Image_dir = 'Images'
File_dir = 'Files'

# Simulation parameters
t = 1.0  # Hopping parameter
U = -2.0  # Coulomb Interaction
beta_element = 1 / (np.abs(ZerothIterationofSelfEnergy(U, t)))

# Grid configurations for different simulation modes:
# - np.linspace(-5, -0.1, 120) or np.linspace(-25, -0.1, 95) for the interaction dependence of 'a' (uncomment ReadFileWithData_a())
# - np.linspace(-2, -0.5, 4) for the temperature dependence of 'a' and 'gamma'
U_values = np.linspace(-2, -0.5, 4)

# - np.logspace(2, 6, 9) for the interaction dependence of 'a' (uncomment ReadFileWithData_a())
# - np.logspace(-1.5, 2, 50) for the temperature dependence of 'a' and 'gamma'
beta = np.logspace(-1.5, 2, 50)

ScaleFactor = 1.5

# For better orientation and troubleshooting, these exceptions have been commented out
# to find the main source of the errors behind the "Wrong Value Try it again." message.

# Resolution suggestion (Exception handling to be implemented in the future)
Rezolution = int(input("Enter the resolution: "))

if Rezolution >= 0:
    x_values = np.linspace(-(Gamma_0(U, t) / (2 * t)) * ScaleFactor, (Gamma_0(U, t) / (2 * t)) * ScaleFactor,
                           Rezolution)
    # x_values = np.linspace(-0.5, 0.5, Rezolution)

    # The key factor for the convergence of numerical integrals was the quality of the interpolation,
    # the self-energy, and the density of points.
    omega_values = np.linspace(-1250, 1250, Rezolution)  # Typical calculation range is 5000-10000

    NumIteration = 100
    Tolerance = 1e-12

    # Class initialization
    # IP = MakeIterationFuntions(beta, x_values, t, omega_values, U, NumIteration,
    #                            Tolerance, Rezolution,
    #                            Gamma_0(U, t), ZerothIterationofSelfEnergy(U, t))
    IP = MakeIterationProcessforGamma(beta, x_values, t, omega_values, U, U_values, NumIteration,
                                      Tolerance, Rezolution,
                                      Gamma_0(U, t), ZerothIterationofSelfEnergy(U, t))
    IP.SolveFinalEquation() # THis is to show the complete interion process working
    Supercomputer = ReadDataFilesFromSuperComputer(beta, x_values, t, omega_values, U, U_values, NumIteration, Tolerance,
                                                  Rezolution)

    # One of these methods must be commented out to prevent collisions between them.
    #Supercomputer.ReadFileWithData_a()
    Supercomputer.ReadFileWithData_Gamma()

    # IP.SolveFinalEquation()
    # IP.GiveFinalG()
    # IP.Read_Data_and_plot()
    # IP.GiveFinalG()
    # IP.AnalyzeTheSolution_write()
    # IP.AnalyzeTheSolution_read()
# else:
#     print("Resolution must be positive.")