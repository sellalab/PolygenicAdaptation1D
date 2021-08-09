"""
Simulate a full population (either using the computationally expensive full model, or the faster approximation
which just simulates a population by tracking all the alles segregating in the population).
The program can be run on the command line
"""
from simulate_populations_class import SimulatePopulations
import argparse
import os

# first set up some default parameters for convenience
default_params = dict()

# Change my_base_directory_local to be the in the full directory address of PolygenicAdpation1D"
my_base_directory_local = '/home/ubuntu/PolygenicAdpation1D'

# default pop size N and mutation rate U
default_params['N'], default_params['U'] = 5000, 0.01
# if Vs < 0, then Vs will be set as 2N
default_params['Vs'], default_params['shift_s0'] = -1, 2 #shift in units of sigma_0

#for gamma
# if V2Ns < 0, then V2Ns will be set as E2Ns^2
default_params['E2Ns'], default_params['V2Ns'] = 10, -1 #exponential dist. <0

default_params['t_freeze_new_muts'] = [0]

default_params['hi'], default_params['fT'] = True, 0

default_params['sD'] = os.path.join(my_base_directory_local,'results_local_test')

# bTN should be at least 5. 10 for simulations for final conclusions.
# lTN can be 1 for rough simulations. But for final conclusion sims you want
# 10 as well
default_params['bTN'], default_params['lTN'] = 5, 1

# Simulation 5 populations
default_params['r'] = 5


# specify flags & usage
parser = argparse.ArgumentParser(description='Simulate the evolution of a quantative trait '
                                             'before and after a shift in optimum phenotype.Vital to specify file.')
parser.add_argument('-sD', '--save_directory', type=str, default = default_params['sD'],
                    help='The directory in which the results are saved')
parser.add_argument('-U', type=float, default = default_params['U'],
                    help='The mutation rate. The default is ' +str(default_params['U']))
parser.add_argument('-Vs', type=float, default = default_params['Vs'],
                    help='Free parameter in fitness function. NOTE that this choice is irrelevant, provided you look at'
                         ' the results in units of trait value that that scale with root(Vs). For example, read_data '
                         'will present results in units of initial phenotypic SDs or in units of delta =root(Vs/(2N)), '
                         'both of which scale with root(Vs).If, for example, you choose '
                         'Vs= 1, then simulations run and results are saved in units of Vs=1. But it is still better to '
                         'look at results in units of Vs=2N (i.e. delta^2 =Vs/(2N) =1). If you choose Vs< 0, '
                         'the simulator will use Vs = 2N. We use as default ' +str(default_params['Vs']))
parser.add_argument('-N', type=int, default = default_params['N'],
                    help='The population size. Default is ' +str(default_params['N']))
parser.add_argument('-E2Ns', type=float, default = default_params['E2Ns'],
                    help='The expected steady-state scaled selection co-efficient of a new mutation (or E[a^2] in units'
                         ' of delta^2=Vs/(2N)).The scaled selection coefficients of new muations are gamma distributed.'
                         ' Default is ' + str(default_params['E2Ns']))
parser.add_argument('-V2Ns', type=float, default = default_params['V2Ns'],
                    help='The variance in steady-state scaled selection co-efficient of a new mutation. When this '
                         'negative then V_2Ns = E_2Ns**2, implying the distribution of scaled selection coefficients '
                         'of new mutations is exponentially distributed. Default is ' + str(default_params['V2Ns']))
parser.add_argument('-fT','--freeze_time', type=int, default = default_params['fT'],
                    help='The freeze my_time - my_time after the shift when variants frozen. Have only ever used time 0. Default is '
                         + str(default_params['fT']))
parser.add_argument('-bTN','--burn_time_N', type=int, default = default_params['bTN'],
                    help='The burntime in units of pop size - the burn time before the shift in optimum. Default is '
                         + str(default_params['bTN']))
parser.add_argument('-lTN','--lag_time_N', type=int, default = default_params['lTN'],
                    help='The lag time in units of pop size - the burn time between runs. default is '+str(default_params['lTN']))

parser.add_argument('-D_s0','--shift_s0', type=float, default = default_params['shift_s0'],
                    help='The shift in terms of sigma_0. The default is ' + str(default_params['shift_s0']))
parser.add_argument('-a', '--algorithm', choices = ['approx','exact'], default = 'approx',
                    help="use the approximation instead by a Wright-Fisher process."
                         "Default is to use it.")
parser.add_argument('-r', '--runs', type=int, default = default_params['r'],
                    help='The number of runs of the simulation implemented by a single .py. '
                         'The default is ' + str(default_params['r']))
parser.add_argument('-hi', '--histograms', help='Causes us to save data for histograms on segregating'
                          'mutations. Default is not to do so', action="store_true")

parser.add_argument('-t_freeze_new_muts', type=int, default = default_params['t_freeze_new_muts'],
                    help='List with the time at which we start freezing new mutations and the period that we do it for. '
                         'Default is to start at time of shift and stop after N generations ' +str(default_params['t_freeze_new_muts']))

parser.add_argument('-t_f','--t_freeze_new_muts',  type=int, nargs='+', default = default_params['t_freeze_new_muts'],
                    help="List of the time to begin collecting new muts and the time to end. Default is " +str(default_params['t_freeze_new_muts']))

parser.add_argument('-o', '--offspring', help='only use if NOT using approx. Causes us to sample '
                        'by offspring fitness instead of parent. Default is parent, which is faster', action="store_true")

# save the arguments entered in the terminal
args = parser.parse_args()

# if args.V2Ns<0, make effect dist exponential
if args.V2Ns < 0:
    args.V2Ns = args.E2Ns**2
# if args.Vs<0, set it to be 2N
if args.Vs < 0:
    args.Vs = float(2*args.N)

# Set the population simulator to have specific mutational and demographic parameters
popSimulator = SimulatePopulations(shift_s0=args.shift_s0, E2Ns=args.E2Ns, V2Ns=args.V2Ns, N=args.N, U=args.U, Vs=args.Vs,
                                   lag_time_N=args.lag_time_N, histograms=args.histograms)

# set up a full population if we are using the exact model
if args.algorithm == 'exact':
    # Sample by offspring fitness, if args.offspring is True <- don't recommend. Slow
    popSimulator.initiate_basic_population_classExact(selection_mode_offspring=args.offspring)
else:
    # set up for wright-fisher sims if args.algorithm == 'approx'. Recommend this for speed
    popSimulator.initiate_basic_population_classWF()

# If we are saving histogram data, set up the histograms
if args.histograms:
    # Set up the histograms for all the new mutations that arise during certain period
    # I always start at time 0, which is time of shift, and to 5N usually)
    if len(args.t_freeze_new_muts)>0:
        t_freeze_new_muts_start = args.t_freeze_new_muts[0]
        if len(args.t_freeze_new_muts)>1:
            t_length_freeze_new_muts= args.t_freeze_new_muts[1]
            popSimulator.initiate_histograms_new_muts(t_freeze_new_muts_start,t_length_freeze_new_muts)
        else:
            popSimulator.initiate_histograms_new_muts(t_freeze_new_muts_start)
    else:
        popSimulator.initiate_histograms_new_muts()
    # Set up the histograms for mutations frozen at freeze time
    # (I always use time 0 for the freeze time, which is the time of the shift)
    popSimulator.initiate_histograms_frozen_muts(args.freeze_time)

# We advance many generations. MANY generations, to burn the population till it reaches
# steady-state under stabilizing selection
popSimulator.burn_basic_population_to_reach_steady_state(burn_time_N=args.burn_time_N)

# I run more than one run with a single job to make it less parallel
# in cases where the sims very fast.
# If you are using simulation parameters that take a long time to run
# you probably want runs=1
for _ in range(args.runs):
    # For each run of these sim params we need a new current population
    popSimulator.refresh_current_population(save_folder=args.save_directory)
    # We burn that current population in, to make it different from the basic
    popSimulator.burn_current_pop()
    # Shift the optimum and run for many generations
    popSimulator.shift_opt_and_run()
    # Save the statistics collected in this run
    popSimulator.save_stats_for_current_population()