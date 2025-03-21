import os

import matplotlib.pyplot as mpl
import numpy as np
from tabulate import tabulate
from assimulo.solvers import IDA, RungeKutta4

from squeezer import Seven_bar_mechanism_indx3, Seven_bar_mechanism_indx2, Seven_bar_mechanism_indx1, \
    Seven_bar_mechanism_expl, compute_consistent_initial

angles = [r'$\beta$', r'$\Theta$', r'$\gamma$', r'$\phi$', r'$\delta$', r'$\Omega$', r'$\epsilon$']
velocities = [r'$\dot{\beta}$', r'$\dot{\Theta}$', r'$\dot{\phi}$', r'$\dot{\delta}$',
                 r'$\dot{\omega}$', r'$\dot{\Omega}$', r'$\dot{\epsilon}$']
accelerations = [r'$\ddot{\beta}$', r'$\ddot{\Theta}$', r'$\ddot{\phi}$', r'$\ddot{\delta}$',
                 r'$\ddot{\omega}$', r'$\ddot{\Omega}$', r'$\ddot{\epsilon}$']
lambdas = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$', r'$\lambda_5$', r'$\lambda_6$',
           r'$\lambda_7$']
var_labels = angles + velocities + lambdas

tab_headers = ['experiment', 'index', 'atol_v', 'atol_lambda', 'algvar_v', 'algvar_lambda', 'suppress_alg']

output_dir = './Plots/Tables/'
os.makedirs(output_dir, exist_ok=True)


def run_seven_bar_problem(with_plots=True, problem_index=3, atol_v=1E5, atol_lambda=1E5,
                          algvar_v=False, algvar_lambda=False, suppress_alg=True, step=0):
    """
    """
    tfinal = 0.03  # Specify the final time

    if problem_index <= 0:
        mod = Seven_bar_mechanism_expl()
    elif problem_index == 1:
        mod = Seven_bar_mechanism_indx1()
    elif problem_index == 2:
        mod = Seven_bar_mechanism_indx2()
    else:
        mod = Seven_bar_mechanism_indx3()

    if problem_index <= 0:
        # Define an explicit solver
        sim = RungeKutta4(mod)

        if step:
            sim.h = step

        # Set the parameters

        # Simulate
        t, y = sim.simulate(tfinal)
    else:
        # Define an implicit solver
        sim = IDA(mod)

        # Set the parameters
        atol = 1E-6 * np.ones((20,))
        atol[7:14] = atol_v
        atol[14:20] = atol_lambda
        sim.atol = atol
        algvar = np.ones((20,))
        algvar[7:14] = algvar_v
        algvar[14:20] = algvar_lambda
        sim.algvar = algvar
        sim.suppress_alg = suppress_alg

        # Simulate
        t, y, yd = sim.simulate(tfinal)

    # Plot
    if with_plots:
        plot_soln(t, y)
        mpl.show()
    return mod, sim, [t, y]


def plot_soln(t, y, plotnumber=500):
    # do some plotting
    mpl.figure(plotnumber, clear=False)
    for i in range(7):
        mpl.plot(t, y[:, i])
    mpl.legend(var_labels[:7])
    mpl.title('Time Evolution of Angles')
    mpl.xlabel('Time (s)')
    mpl.ylabel('Angles (rad)')
    mpl.savefig(f'report/plots/angles.pdf')
    
    mpl.figure(plotnumber + 1, clear=False)
    for i in range(7, 14):
        mpl.plot(t, y[:, i])
    mpl.legend(var_labels[7:14])
    mpl.title('Time Evolution of Angular Velocities')
    mpl.xlabel('Time (s)')
    mpl.ylabel('Angular Velocities (rad/s)')
    mpl.savefig(f'report/plots/velocities.pdf')
    
    if y.shape[1] > 14:
        mpl.figure(plotnumber + 2, clear=False)
        for i in range(14, 20):
            mpl.plot(t, y[:, i])
        mpl.legend(var_labels[14:])
        mpl.title('Time Evolution of Lagrange Multipliers')
        mpl.xlabel('Time (s)')
        mpl.ylabel('Lagrange Multipliers (N⋅m)')
        mpl.savefig(f'report/plots/lagrange_multipliers.pdf')


def plot_stats(xdata, ydata, plotnumber=500, xlabel='', figsize=(6.4,4.8)):
    ylabels = ['nsteps', 'nfcns', 'njacs', 'nerrfails']
    titles = ['Number of Integration Steps', 'Number of Function Evaluations', 
              'Number of Jacobian Evaluations', 'Number of Error Test Failures']
    for i in range(4):
        mpl.figure(plotnumber + i, figsize, clear=False)
        mpl.bar(xdata, ydata[i])
        mpl.xlabel(xlabel)
        mpl.ylabel(ylabels[i])
        mpl.title(titles[i])
        mpl.tight_layout()
        mpl.savefig(f'report/plots/stats_{i}.pdf')
    
    ylabels = ['nfcns / nsteps', 'njacs / nsteps', 'nerrfails / nsteps']
    titles = ['Function Evaluations per Step', 'Jacobian Evaluations per Step', 
              'Error Test Failures per Step']
    for i in range(3):
        mpl.figure(plotnumber + 10 + i, figsize, clear=False)
        mpl.bar(xdata, np.asarray(ydata[i + 1]) / np.asarray(ydata[0]))
        mpl.xlabel(xlabel)
        mpl.ylabel(ylabels[i])
        mpl.title(titles[i])
        mpl.tight_layout()
        mpl.savefig(f'report/plots/stats_{i + 10}.pdf')


if __name__ == '__main__':
    run_seven_bar_problem(True, 1, 1E5, 1E5, False, False, True)

    if True:
        # This plots comparisons of the index 1,2,3 formulations
        all_solns = []
        for i in range(4):
            _, _, soln = run_seven_bar_problem(False, i)
            if i <= 0:
                # Do some padding in the case of the explicit problem
                soln[1] = np.hstack((soln[1], np.zeros((len(soln[0]), 6))))
            all_solns.append(soln)

        t = np.linspace(0, 0.03, 500)
        all_solns_interp = np.zeros((4, t.size, 20))
        for i, soln in enumerate(all_solns):
            for j in range(20):
                all_solns_interp[i, :, j] = np.interp(t, soln[0], soln[1][:, j])

        # Plot soln
        plot_soln(all_solns[1][0], all_solns[1][1], plotnumber=510)
        plot_soln(all_solns[2][0], all_solns[2][1], plotnumber=513)
        plot_soln(all_solns[3][0], all_solns[3][1], plotnumber=516)
        plot_soln(t, all_solns_interp[3, :, :] - all_solns_interp[0, :, :], plotnumber=520)
        plot_soln(t, all_solns_interp[3, :, :] - all_solns_interp[1, :, :], plotnumber=530)
        plot_soln(t, all_solns_interp[3, :, :] - all_solns_interp[2, :, :], plotnumber=540)
        # mpl.show()

    if True:
        # This compares the index=1,2,3 formulations
        # list of experiments in the form [problem_index, atol_v, atol_lambda, algvar_v, algvar_lambda, suppress_alg]
        experiments = [[1, 1E5, 1E5, False, False, True],
                       [2, 1E5, 1E5, False, False, True],
                       [3, 1E5, 1E5, False, False, True]]

        nsteps = []
        nfcns = []
        njacs = []
        nerrfails = []
        xdata = []
        for counter, exp in enumerate(experiments):
            try:
                mod, sim, _ = run_seven_bar_problem(False, *exp)

                stats = sim.get_statistics()
                xdata.append(f'{exp[0]}')
                nsteps.append(stats.__getitem__('nsteps'))
                nfcns.append(stats.__getitem__('nfcns'))
                njacs.append(stats.__getitem__('njacs'))
                nerrfails.append(stats.__getitem__('nerrfails'))
            except:
                print(f'There seems to be a problem in the experiment {exp}')

        plot_stats(xdata, [nsteps, nfcns, njacs, nerrfails], plotnumber=600, xlabel='index', figsize=(2, 2))

    if True:
        # This tests the index=1 problem
        # list of experiments in the form [problem_index, atol_v, atol_lambda, algvar_v, algvar_lambda, suppress_alg]
        experiments = [[1, 1E5, 1E5, False, False, True],
                       [1, 1E-6, 1E5, False, True, True],
                       [1, 1E-6, 1E5, True, False, True],
                       [1, 1E-6, 1E5, True, True, False],
                       [1, 1E-6, 1E-6, False, False, True]]


        # print(tabulate(experiments, headers=tab_headers, showindex='always', tablefmt='fancy_grid'))
        with open('./Plots/Tables/Overview_Index1Experiment.tex', 'w') as output:
            output.write(tabulate(experiments, headers=tab_headers, showindex='always', tablefmt='latex'))
            output.close()

        nsteps = []
        nfcns = []
        njacs = []
        nerrfails = []
        xdata = []
        for counter, exp in enumerate(experiments):
            try:
                mod, sim, _ = run_seven_bar_problem(False, *exp)

                stats = sim.get_statistics()
                xdata.append(f'{counter}')
                nsteps.append(stats.__getitem__('nsteps'))
                nfcns.append(stats.__getitem__('nfcns'))
                njacs.append(stats.__getitem__('njacs'))
                nerrfails.append(stats.__getitem__('nerrfails'))
            except:
                print(f'There seems to be a problem in the experiment {exp}')

        plot_stats(xdata, [nsteps, nfcns, njacs, nerrfails], plotnumber=700, xlabel='experiment', figsize=(2, 2))

    if True:
        # This tests the index=2 problem
        # list of experiments in the form [problem_index, atol_v, atol_lambda, algvar_v, algvar_lambda, suppress_alg]
        experiments = [[2, 1E-6, 1E5, False, False, True],
                       [2, 1E-6, 1E5, False, True, True],
                       [2, 1E-6, 1E5, False, True, True]]

        nsteps = []
        nfcns = []
        njacs = []
        nerrfails = []
        xdata = []
        for counter, exp in enumerate(experiments):
            try:
                mod, sim, _ = run_seven_bar_problem(False, *exp)

                stats = sim.get_statistics()
                xdata.append(f'prblm {counter}')
                nsteps.append(stats.__getitem__('nsteps'))
                nfcns.append(stats.__getitem__('nfcns'))
                njacs.append(stats.__getitem__('njacs'))
                nerrfails.append(stats.__getitem__('nerrfails'))
            except:
                print(f'There seems to be a problem in the experiment {exp}')

        plot_stats(xdata, [nsteps, nfcns, njacs, nerrfails], plotnumber=800, figsize=(2, 2))
    # mpl.show()

    if True:
        # This exports the experiment configuration as a latex table
        # tab_headers = ['experiment', r'\pyth{problem_index}', r'\pyth{atol_v}', r'\pyth{atol_lambda}', r'\pyth{algvar_v}',
        #                r'\pyth{algvar_lambda}', r'\pyth{suppress_alg}']
        tab_headers = ['experiment', 'index', 'atol_v', 'atol_lambda', 'algvar_v', 'algvar_lambda', 'suppress_alg']
        experiments = [[1, 1.E5, 1.E5, False, False, True],
                       [1, 1E-6, 1.E5, False, True, True],
                       [1, 1E-6, 1.E5, True, False, True],
                       [1, 1E-6, 1.E5, True, True, False],
                       [1, 1E-6, 1E-6, False, False, True]]
        # for i, line in enumerate(experiments):
        #    for j, obj in enumerate(line):
        #        experiments[i][j] = f'\pyth{{{obj}}}'
        print(tabulate(experiments, headers=tab_headers, showindex='always', tablefmt='fancy_grid'))
        with open('./Plots/Tables/Overview_Index1Experiment.tex', 'w') as output:
            output.write(tabulate(experiments, headers=tab_headers, showindex='always', tablefmt='latex'))
            output.close()

    if True:
        # This tests the RK4 method for different values of h
        exp = [0, 1E-6, 1E-6, False, False, False]

        norms = []
        t = np.array(np.arange(0.001, 0.002, 0.000005))

        for step in t:
            try:
                mod, sim, [_, y] = run_seven_bar_problem(False, *exp, step)
                norms.append(np.linalg.norm(np.array(y)[:, :7], axis=0))
            except:
                print(f'There seems to be a problem in the experiment {exp} with step {step}')

        mpl.plot(t, norms)
        mpl.legend(var_labels[:7])
        mpl.xlabel('h')
        mpl.ylabel('L2')
        mpl.show()

    if True:
        # This tests the RK4 method
        exp = [0, 1E-6, 1E-6, False, False, False]

        try:
            mod, sim, _= run_seven_bar_problem(True, *exp, step=0.00195)
        except:
            print(f'There seems to be a problem in the experiment {exp}')


    if True:
        # This exports the generated initial values as a latex table

        # Compute consistent initial values
        y = compute_consistent_initial()

        print(tabulate(list(zip(angles, y[:7])), headers='firstrow', tablefmt='fancy_grid'))
        print(tabulate(list(zip(accelerations, y[7:14])), headers='firstrow', tablefmt='fancy_grid'))
        print(tabulate(list(zip(lambdas, y[14:])), headers='firstrow', tablefmt='fancy_grid'))
        with open('./Plots/Tables/Initial_Angles.tex', 'w') as output:
            output.write(tabulate(list(zip(angles, y[:7])), tablefmt='latex_raw'))
            output.close()
        with open('./Plots/Tables/Initial_Accelerations.tex', 'w') as output:
            output.write(tabulate(list(zip(accelerations, y[7:14])), tablefmt='latex_raw'))
            output.close()
        with open('./Plots/Tables/Initial_Lambdas.tex', 'w') as output:
            output.write(tabulate(list(zip(lambdas, y[14:])), tablefmt='latex_raw'))
            output.close()