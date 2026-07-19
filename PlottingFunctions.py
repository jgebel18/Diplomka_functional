import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from rich import color

from Main_Terms_In_Equations import Main_Terms_Equations

matplotlib.use("TkAgg")

import os




class PlottingFunctions:
    def __init__(self):
        """Initializes the plotting class with default directories for saving data and images."""
        self.Path_Files = 'Files'
        self.Path_Images = 'Images'

    def PlottingItegrands(self, D_array, Y_array, x_values):
        """
        Plots the D and Y integrands across multiple iterations in a 2x1 subplot grid.
        Saves the resulting plot as 'Integrands_for_Interpolated_Values.png'.
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        for i in range(D_array.shape[0]):
            axes[0].plot(x_values, D_array[i], label=f"{i}-Iteration")
            axes[1].plot(x_values, Y_array[i], label=f"{i}-Iteration")
            axes[0].set_title("D-Integrands-Iterations")
            axes[1].set_title("Y-Integrands-Iterations")
            axes[0].set_xlabel('x')
            axes[1].set_xlabel('x')
            axes[0].set_ylabel('D-Itegrands')
            axes[1].set_ylabel('Y-Itegrands')
            axes[0].legend()
            axes[1].legend()
            axes[0].grid(True)
            axes[1].grid(True)
        plt.savefig('Integrands_for_Interpolated_Values.png')
        plt.show()

    def PlotIntegrandsfordifferentbetavalues(self, beta, x_values, integrand_2, integrand_3):
        """
        Plots left-hand side (lhs) and right-hand side (rhs) integrands as a function of epsilon
        for various values of beta. Saves the plot as 'Integrands_for_beta_values.png'.
        """
        fig, ax = plt.subplots()
        for i in range(len(beta)):
            beta_val = beta[i]
            ax.plot(x_values, integrand_2[i], label=f"Integrand_lhs (β={beta_val})")
            ax.plot(x_values, integrand_3[i], label=f"Integrand_rhs (β={beta_val})")
        ax.set_title("Integrands for Different eslion Values")
        ax.set_xlabel('ε')
        ax.set_ylabel('Integrand Value')
        ax.legend()
        plt.tight_layout()
        plt.savefig('Integrands_for_beta_values.png')
        plt.show()

    def PlotResultsofIntegrals(self, X, Y, Y_2):
        """
        Plots calculated D integral results across different epsilon values and compares
        them to the ideal infinite beta limit scenario. Saves the plot as 'Results_for_epsilon_values.png'.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(X, Y, label="D Integral Results for different epsilon values")
        plt.plot(X, Y_2, label="Ideal beta =inf")
        plt.xlabel('ε')
        plt.ylabel('Integral results')
        plt.legend()
        plt.savefig('Results_for_epsilon_values.png')
        plt.show()

    def PlotIntegrands(self, nazev, DataX, DataY, nazev1, DataY2, nazev2, DataY3, nazev3):
        """
        Plots three distinct integrands into a vertical 3x1 grid sharing the same X-axis.
        Useful for dissecting different components that define the D function.
        Saves the file dynamically based on the 'nazev' parameter.
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        # First subplot
        axes[0].plot(DataX, DataY, label=nazev1)
        axes[0].set_title(nazev1)
        axes[0].set_ylabel('Integrand')
        axes[0].legend()

        # Second subplot
        axes[1].plot(DataX, DataY2, label=nazev2, color='orange')
        axes[1].set_title(nazev2)
        axes[1].set_ylabel('Integrand')
        axes[1].legend()

        # Third subplot
        axes[2].plot(DataX, DataY3, label=nazev3, color='green')
        axes[2].set_title(nazev3)
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('Integrand')
        axes[2].legend()
        fig.suptitle(nazev)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(nazev + '_subplots.png')
        plt.show()

    def PlotData(self, dataX, dataY):
        """
        Generates a standard standalone 2D plot for the Green's function G(omega) vs omega.
        Saves the graphic as 'Integrands.png'.
        """
        plt.figure(figsize=(10, 10))
        plt.plot(dataX, dataY)
        plt.xlabel('ω')
        plt.ylabel('G(ω)')
        plt.savefig('Integrands.png')
        plt.show()

    def PlotTwoSubplots(self, Nazev1, Nazev2, dataX, dataY1, dataY2, Nazev_X, Nazev_Y, Title):
        """
        Generates two subplots side-by-side (1x2 grid) comparing two different datasets
        sharing identical X and Y labels. Saves the file under the provided 'Title' name.
        """
        plt.figure(figsize=(12, 6))

        # Left subplot
        plt.subplot(1, 2, 1)
        plt.plot(dataX, dataY1, label=Nazev1, color='blue')
        plt.xlabel(Nazev_X)
        plt.ylabel(Nazev_Y)
        plt.title(Nazev1)
        plt.grid(True)

        # Right subplot
        plt.subplot(1, 2, 2)
        plt.plot(dataX, dataY2, label=Nazev2, color='red')
        plt.xlabel(Nazev_X)
        plt.ylabel(Nazev_Y)
        plt.title(Nazev2)
        plt.grid(True)
        plt.suptitle(Title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(Title + '.png')
        plt.show()

    def PlotGreenFunction(self, omega, Function, Sigma, NumberIterationToPlot):
        """
        Iterates over a specific count of simulation runs to plot both the Real and Imaginary
        parts of the Green's Function G(omega). Saves the plot as 'Greens_function_omega.png'.
        """
        fig, ax = plt.subplots()
        for i in range(NumberIterationToPlot):
            data = Function(omega, Sigma[i])
            ax.plot(omega, data.real, label=f"{i}-iterace funkce G(ω) reálná část")
            ax.plot(omega, data.imag, label=f"{i}-iterace funkce G(ω) imaginární část")
        ax.set_xlabel("ω")
        ax.set_ylabel("G(ω)")
        ax.legend()
        plt.grid(True)
        plt.savefig('Greens_function_omega.png')
        plt.show()

    def PlotSigmaFunction(self, omega, SigmaInterpolated, SigmaOriginal):
        """
        Provides a scatter plot validation to check the quality of interpolation for the self-energy
        Sigma(omega) by comparing the original data points directly against a highly sampled
        interpolated line (100,000 points). Saves as 'Interpolated_Sigma.png'.
        """
        fig, ax = plt.subplots()
        newomega = np.linspace(omega.min(), omega.max(), num=100000)
        ax.scatter(newomega, SigmaInterpolated(newomega).real, label=f"Interpolated  Σ(ω) reálná část")
        ax.scatter(newomega, SigmaInterpolated(newomega).imag, label=f"Interpolated Σ(ω) imaginární část")
        ax.scatter(omega, SigmaOriginal.real, label=f"Original  Σ(ω) reálná část")
        ax.scatter(omega, SigmaOriginal.imag, label=f"Original  Σ(ω) imaginární část")
        ax.set_xlabel("ω")
        ax.set_ylabel("Σ(ω)")
        ax.legend()
        plt.savefig("Interpolated_Sigma.png")
        plt.show()

    def PlotRootsofQuarticEquation(self, roots, omega):
        """
        Splits and maps all four distinct complex roots of the quartic equation into a 2x2 grid
        of scatter plots against omega, visualizing both Real and Imaginary parts.
        Saves the file into the predefined 'Files' directory.
        """
        nazev = f'Self-Energy for every root of quartic equation'
        fig, axes = plt.subplots(2, 2, figsize=(10, 12), sharex=True)
        axes = axes.flatten()
        for i in range(len(roots)):
            axes[i].scatter(omega, roots[i].real, label=f"{i}-th Root (Re)")
            axes[i].scatter(omega, roots[i].imag, label=f"{i}-th Root (Im)")
            axes[i].set_xlabel('ω')
            axes[i].set_ylabel('Σ(ω)')
            axes[i].set_title(f'{i}-th Root of Equation')
            axes[i].legend()
            axes[i].grid(True)
        fig.suptitle(nazev)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.Path_Files, nazev + f'_subplots.png'), dpi=300)
        plt.show()

    def PlotRootsOfEquation(self, roots, omega):
        """
        Displays a comparative 3x2 grid of scatter plots charting out the roots of both
        the system's Nonlinear equation (index 0) and the remaining Quartic equations.
        Saves the plot as 'Self-Energy for every root of both equation_subplots.png'.
        """
        nazev = 'Self-Energy for every root of both equation'
        fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=True)
        axes = axes.flatten()
        for i in range(len(roots)):
            if i == 0:
                axes[i].scatter(omega, roots[i].real, label=f"Nonlinear equation Root (Re)")
                axes[i].scatter(omega, roots[i].imag, label=f"Nonlinear equation Root (Im)")
                axes[i].legend()
                axes[i].grid(True)
                axes[i].set_xlabel('ω')
                axes[i].set_ylabel('Σ(ω)')
            else:
                axes[i].scatter(omega, roots[i].real, label=f"{i}-th Root (Re)")
                axes[i].scatter(omega, roots[i].imag, label=f"{i}-th Root (Im)")
                axes[i].legend()
                axes[i].grid(True)
                axes[i].set_xlabel('ω')
                axes[i].set_ylabel('Σ(ω)')
        fig.suptitle(nazev)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(nazev + '_subplots.png', dpi=300)
        plt.show()

    def PlotBothpartsofSigma(self, Sigma_values, omega):
        """
        Plots the entire aggregated array of complete roots for the quartic equation
        (Real vs Imaginary parts) together inside a single scatter chart.
        """
        plt.figure(figsize=(10, 10))
        plt.scatter(omega, Sigma_values.real, label='Real Part')
        plt.scatter(omega, Sigma_values.imag, label='Imaginary Part')
        plt.legend()
        plt.grid(True)
        plt.xlabel('ω')
        plt.ylabel('Σ(ω)')
        plt.title('Complete Roots of Quartic equation')
        plt.show()

    def PlotRootOfEquation(self, root, omega, Gamma):
        """
        Visualizes a single chosen root isolated from the nonlinear method for a specific
        Gamma value, plotting Real and Imaginary parts. Saves into the 'Files' folder.
        """
        nazev = f'Self-Energy for root of nonlinear equation'
        plt.figure(figsize=(10, 10))
        plt.scatter(omega, root.real, label=f" Root (Re)")
        plt.scatter(omega, root.imag, label=f" Root (Im)")
        plt.xlabel('ω')
        plt.ylabel('Σ(ω)')
        plt.title(f'Root of Equation')
        plt.legend()
        plt.grid(True)
        plt.title(nazev)
        plt.savefig(os.path.join(self.Path_Files, nazev + f'_subplots-{Gamma}.png'), dpi=300)
        plt.show()

    def PlotRootOfEquationEps(self, root, omega, epsilon):
        """
        Plots the chosen nonlinear solution root across an iterable list of different
        epsilon values to map variance. Saves the plot as a subplot image format.
        """
        nazev = 'Self-Energy for root of nonlinear equation for epsilon'
        plt.figure(figsize=(10, 12))
        for i, eps_val in enumerate(epsilon):
            plt.scatter(omega, root[i].real, label=f" Root (Re)-epsilon-{eps_val}")
            plt.scatter(omega, root[i].imag, label=f" Root (Im)-epsilon-{eps_val}")
        plt.xlabel('ω')
        plt.ylabel('Σ(ω)')
        plt.title(f'Root of Equation')
        plt.legend()
        plt.grid(True)
        plt.title(nazev)
        plt.savefig(nazev + '_subplots.png', dpi=300)
        plt.show()

    def Plot_Values_of_a_and_D(self, values_Y, Values_D, values_Y_approx, values_D_approx, Gamma_Values, beta):
        """
        Plots calculated values of numerical integrals D and Y alongside their approximations across
        various Gamma configurations for a set beta parameter. Saves the breakdown under the 'Images' folder.
        """
        Images_path = 'Images'
        D_control = Values_D * Gamma_Values ** 2
        D_control_approx = values_D_approx
        Y_control = values_Y
        Y_control_approx = values_Y_approx

        nazev = os.path.join(Images_path, f'Values_of_Y_D_beta_{beta}.png')
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        axes[0].scatter(Gamma_Values, D_control, label='D')
        axes[0].scatter(Gamma_Values, D_control_approx, label='D_approx')
        axes[0].plot(Gamma_Values, D_control - D_control_approx, label='D_diff')
        axes[0].set_xlabel('Gamma')
        axes[0].set_ylabel('D')
        axes[0].legend()
        axes[0].grid(True)
        axes[1].scatter(Gamma_Values, Y_control, label='Y')
        axes[1].scatter(Gamma_Values, Y_control_approx, label='Y_approx')
        axes[1].plot(Gamma_Values, Y_control - Y_control_approx, label='Y_diff')
        axes[1].set_xlabel('Gamma')
        axes[1].set_ylabel('Y(0,0)')
        axes[1].legend()
        axes[1].grid(True)
        fig.suptitle(f'Values of Y and D for beta={beta}')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.title(f'Values of Y and D for beta={beta}')
        plt.savefig(nazev)
        plt.show()

    def Plot_Beta_Gamma_Dependence(self, Variables, beta_value, Gamma_Values, U):
        """
        Generates a 2x2 grid dashboard mapping out the core variables (Y, D, a, and nonlinear a)
        relative to the Gamma scales across multiple beta parameter lines. Utilizes LaTeX text rendering.
        """
        Axes_y = np.array([r'$Y(\Gamma)$', r'$D(\Gamma)*\Gamma^2$', r'$a(\Gamma)$', r'$a_{Nonlinear}(\Gamma)$'])
        Names = self.TeXEquations()
        Axes_x = r'$\Gamma$'
        Nazev = f'Values_of_all_important_terms_in_equations_for_different_Gamma_U:{U}-Test'
        plt.rcParams['text.usetex'] = True

        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()

        for i in range(Axes_y.size):
            for j, beta in enumerate(beta_value):
                axes[i].scatter(Gamma_Values, Variables[i][j], marker='v', s=20, alpha=0.7,
                                label=rf'$\beta={beta:.2f}$')

            axes[i].set_xlabel(Axes_x)
            axes[i].set_ylabel(Axes_y[i])
            axes[i].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=9)
            axes[i].text(0.6, 0.6, Names[i],
                         fontsize=16,
                         transform=axes[i].transAxes,
                         verticalalignment='top',
                         horizontalalignment='left')
            axes[i].grid(True)
            axes[i].set_title(rf'{Axes_y[i]} - Values for different Gamma and beta')

        fig.tight_layout()
        os.makedirs('Images', exist_ok=True)
        plt.savefig(os.path.join('Images', Nazev + '.png'), bbox_inches='tight', dpi=300)
        plt.show()

    def AddLimitsofIntervals(self, i, j, axes, limits_unpacked, beta):
        """
        Overlays threshold limits (x0, x1, x2 boundary points) on specific subplots (indices 2 and 3)
        to trace convergence boundaries across varying levels of beta.
        """
        (Key_a_x0, Key_a_x1, Key_a_x1_new,
         Key_gamma_x0, Key_gamma_x1, Key_gamma_x1_new,
         x0, x1, x1_new) = limits_unpacked

        if i == 2:
            axes[i].scatter(x0.flatten()[j], Key_a_x0.flatten()[j], s=40, marker='o',
                            label=rf'$x_0 (\beta={beta:.2f})$')
            axes[i].scatter(x1.flatten()[j], Key_a_x1.flatten()[j], s=40, marker='s',
                            label=rf'$x_1 (\beta={beta:.2f})$')
            axes[i].scatter(x1_new.flatten()[j], Key_a_x1_new.flatten()[j], s=40, marker='D',
                            label=rf'$x_2 (\beta={beta:.2f})$')
        elif i == 3:
            axes[i].scatter(x0.flatten()[j], Key_gamma_x0.flatten()[j], s=40, marker='o',
                            label=rf'$x_0 (\beta={beta:.2f})$')
            axes[i].scatter(x1.flatten()[j], Key_gamma_x1.flatten()[j], s=40, marker='s',
                            label=rf'$x_1 (\beta={beta:.2f})$')
            axes[i].scatter(x1_new.flatten()[j], Key_gamma_x1_new.flatten()[j], s=40, marker='D',
                            label=rf'$x_2 (\beta={beta:.2f})$')

    def PlotaNonlinear(self, function, GammaValues, beta):
        """
        Plots the functional output behavior of the non-linear equation system against
        Gamma settings for an individual beta configuration.
        """
        plt.figure(figsize=(20, 20))
        plt.plot(GammaValues, function)
        plt.xlabel('Γ')
        plt.ylabel('f(Γ)')
        plt.savefig(f'Gamma_NonlinearEQ-{beta}')
        plt.show()

    def TeXEquations(self):
        """
        Helper method returning an array of raw LaTeX equation strings used for labeling
        and documentation purposes within generated plot figures.
        """
        Titles = np.array([
            r"$Y(0,0)=\int_{-\infty}^{+\infty} \frac{dx}{\pi} \cdots$",
            r"$D = \int_{-\infty}^{+\infty} \frac{dx}{\pi} \cdots$",
            r"$a=1+UY(0,0)$",
            r"$ -\frac{1}{U}-Y(0,0)+\frac{2}{\Gamma^{2}(\beta\pi)^{2} D}=0$"
        ])
        return Titles

    def Plot_3D_solution(self, beta_values, U_values, solution):
        """
        Generates a comparative dual-view figure showing the optimization path of parameter 'a':
        Left side presents a 3D surface plot over beta and U spaces; Right side draws the corresponding 2D contour lines.
        """
        plt.rcParams['text.usetex'] = True
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)

        # 3D plot
        surf = ax1.plot_surface(beta_values, U_values, solution, cmap='viridis', alpha=0.8)
        ax1.set_xlabel(rf'$\beta$')
        ax1.set_ylabel(rf'$U$')
        ax1.set_zlabel(rf'Solutions of nonlinear $a$')
        ax1.view_init(elev=45, azim=45)

        # 2D contour lines
        ax2.contour(beta_values, U_values, solution, levels=30, cmap='viridis')
        ax2.set_xlabel(rf'$\beta$')
        ax2.set_ylabel(rf'$U$')

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title('Hladiny funkce a průběh optimalizace pro obě metody')
        plt.tight_layout()
        plt.savefig(f'Solution-of-nonlinear-equation-for-beta-{beta_values.size}-U-{U_values.size}.png ')
        plt.show()

    def Plot_Fermi_function(self, x_values, beta_values):
        """
        Plots standard Fermi-Dirac distribution functions evaluated across multiple different
        temperatures/beta bounds to inspect scaling setups. Saves in 'Images'.
        """
        plt.figure(figsize=(10, 10))
        for beta in beta_values:
            function = lambda x: Main_Terms_Equations.FermiFunction(x, beta)
            plt.plot(x_values, function(x_values), label=f'Fermi-function-beta-{beta}')
        plt.xlabel('x')
        plt.ylabel('Fermi function(x)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.Path_Images, 'Fermi_function_for_different_beta_for.png'))
        plt.show()

    def Plot_Final_Dependence(self, beta_values, U_values, a_beta_crtitical):
        """
        Tracks critical scaling lines for variable 'a' across changing interaction potentials (U/t ratio)
        for an array of beta environments. Saves as 'Critical_Behaviour-Attempt-beta-U-BCS.png'.
        """
        Name = f'Critical_Behaviour-Attempt-beta-U-BCS.png'
        plt.rcParams['text.usetex'] = True
        plt.figure(figsize=(10, 10))
        for i, beta in enumerate(beta_values):
            plt.plot(U_values, a_beta_crtitical[i], label=rf'$\beta t$-{beta}')
        plt.grid(True)
        plt.xlabel(r'$\frac{U}{t}$', fontsize=20)
        plt.ylabel(rf'$a(\beta t)^2$', fontsize=20)
        plt.ylim(0, 3)
        plt.xlim(-20, 0)
        plt.legend()
        plt.savefig(os.path.join(self.Path_Images, Name))
        plt.show()

    def Plot_Final_Dependence_BCS(self, beta_values, U_values, a_beta_crtitical, a_BCS):
        """
        Compares computed critical state tracking data explicitly against formal
        Bardeen-Cooper-Schrieffer (BCS) theoretical approximations  from thermal fluctuations regime (-U/t).
        """
        Name = f'Critical_Behaviour-Attempt-BCS-U-low_range.png'
        plt.rcParams['text.usetex'] = True
        plt.figure(figsize=(10, 10))
        barvy = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#17becf"]

        for i, beta in enumerate(beta_values):
            plt.plot(-U_values, a_beta_crtitical[i], label=rf'$\beta t$-{beta}', color=barvy[i])
            plt.plot(-U_values, a_BCS[i], '--', linewidth=2.6, label=rf'BCS-approx:$\beta$-{beta}', color=barvy[i])
        plt.grid(True)
        plt.xlabel(r'$\frac{-U}{t}$', fontsize=20)
        plt.ylabel(rf'$a$', fontsize=20)
        plt.xlim(0, 2)
        plt.ylim(-0.1, 1.2)
        plt.legend()
        plt.savefig(os.path.join(self.Path_Images, Name))
        plt.show()

    def Plot_Final_Dependence_Transposed_BCS(self, beta, U_values, a_beta_crtitical, a_BCS):
        """
        Plots a transposed interpretation displaying the behavior of 'a' plotted directly in dependence of
        inverse temperature parameters (1 / beta*t) for discrete interaction constants (U/t).
        """
        Name = f'Critical_Behaviour-with-BCS-Transpose_low_Unstability.png'
        plt.rcParams['text.usetex'] = True
        plt.figure(figsize=(10, 10))
        barvy = ["#1f77b4", "#2ca02c", "#9467bd", "#e377c2"]
        for i in range(U_values.size):
            plt.plot(1 / beta, a_beta_crtitical.T[i], label=r'$\frac{-U}{t}$='f'{-U_values[i]}', color=barvy[i])

        plt.grid(True)
        plt.xlabel(r'$\frac{1}{\beta t}$', fontsize=20)
        plt.ylabel(rf'$a$', fontsize=20)
        #This parameters requires the user correction.
        #plt.xlim(0, 20)
        plt.legend()
        plt.savefig(os.path.join(self.Path_Images, Name))
        plt.show()

    def Plot_Final_Dependence_BCS_Gamma(self, beta_values, U_values, Gamma):
        """
        Plots the logarithmic scale values of irreducible vertex Gamma against inverse interaction ratios (-t/U).
        Helpful for mapping exponential properties typical in superconducting gap developments.
        """
        Name = f'Critical_Behaviour-Gamma-Attempt-U-log.png'
        plt.rcParams['text.usetex'] = True
        plt.figure(figsize=(10, 10))
        barvy = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#17becf"]

        for i, beta in enumerate(beta_values):
            plt.plot(-1 / U_values, np.log(-Gamma[i]), label=rf'$\beta t$-{beta}', color=barvy[i])
        plt.grid(True)
        plt.xlabel(r'$-\frac{t}{U}$', fontsize=20)
        plt.ylabel(r'$ln(-\frac{\Gamma}{(2t)^2})$', fontsize=20)
        plt.legend()
        plt.savefig(os.path.join(self.Path_Images, Name))
        plt.show()

    def Plot_Final_Dependence_Transposed_BCS_Gamma(self, beta, U_values, Gamma_Critical):
        """
        Plots a transposed graph displaying the direct scale of irreducible vertex Gamma
        versus the inverse temperature metrics (1 / beta*t) for explicit constant values of potential U.
        """
        Name = f'Critical_Behaviour-Transpose_Gamma.png'
        plt.rcParams['text.usetex'] = True
        plt.figure(figsize=(10, 10))
        barvy = ["#1f77b4", "#2ca02c", "#9467bd", "#e377c2"]
        for i in range(U_values.size):
            plt.plot(1 / beta, (-Gamma_Critical.T[i]), label=r'$\frac{-U}{t}=$'f'${-U_values[i]} $', color=barvy[i])

        plt.grid(True)
        plt.xlabel(r'$\frac{1}{\beta t}$', fontsize=20)
        plt.ylabel(r'$\frac{-\Gamma}{(2t)^2}$', fontsize=20)
        #These parameters requires the user correction
        #plt.xlim(0, 0.5)
        #plt.ylim(0, 1)
        plt.legend()
        plt.savefig(os.path.join(self.Path_Images, Name))
        plt.show()