"""
Simulates allele trajectories with specific initial squared effect sizes (steady-state scaled selection coefficients).
The program can be run on the command line.
"""
import argparse
import os
from simulate_trajectories_class import SimulateTrajectories
from read_data import find_name_of_sim_pD_folder

# first set up some default parameters for convenience
default_params = dict()

run_folder_local = 'results_local_test/51f9e516daf6c4db153b_pD'

# Change my_base_directory_local to be the in the full directory address of PolygenicAdpation1D"
my_base_directory_local = '/home/ubuntu/PolygenicAdpation1D'
result_file_local  = 'results_local_test'
result_file_local_lande  = 'results_local_test_lande'
local_folder = os.path.join(my_base_directory_local, result_file_local)
local_folder_lande = os.path.join(my_base_directory_local, result_file_local_lande)

default_params['s_min'], default_params['s_max'] = 0.09, 101
# if Vs < 0, then Vs will be set as 2N
default_params['Vs'],default_params['shift_s0'],default_params['sigma_0_del'] = -1.0, 2.0, 20.0
default_params['start_time'], default_params['n'] = 0, 1
default_params['nT'] = 0
default_params['nXi_s'],default_params['nXi_l'] = 0, 0

# IF (and only if) we are reading results from a parent simulation directory, we need a mutation rate per gamete
default_params['U'] = 0.01
#and the expected value and variance for gamma distribution of squared effects,
default_params['E2Ns'], default_params['V2Ns'] = 10, -1 #exponential dist. <0

default_params['sF'] = os.path.join(local_folder,run_folder_local)
default_params['lF'] = local_folder_lande
default_params['psF'] = ''
default_params['N'], default_params['nM'] = 5000, 1000



# Specify flags & usage
parser = argparse.ArgumentParser(description='Simulate the evolution of mutants '
                                             'after a shift in optimum phenotype. Vital to specify one of: '
                                             'lande_Folder, simulation_Folder or parent_simulation_Folder.')
parser.add_argument('-lF', '--lande_Folder', type=str, default = default_params['lF'],
                    help='The folder in which the simulation folder should be saved if only Lande D(t) is being used'
                         ' Default is '+str(default_params['lF']))
parser.add_argument('-sF', '--simulation_Folder', type=str, default = default_params['sF'],
                    help='The simulation folder from which the distance function is being used-if a distance function other than Lande'
                         'is being used.Default is '+str(default_params['sF']))
parser.add_argument('-psF', '--parent_simulation_Folder', type=str, default = default_params['psF'],
                    help='If this is not an empty string, this overrides simulation_Folder. If there is a parent simulation folder,'
                         'the program will search in the parent simulation folder, for a simulation folder with results from simulations'
                         ' using the population genetic parameters specified. i.e. specify the values of N, U, Vs, E2Ns, V2Ns, D_s0 and'
                         'if a simualtion folder exists in parent_simulation_Folder where sims were run with those parameters, then '
                         ' the program will run trajectories using the distance function from that folder-if a distance function other '
                         'than Lande is being used.Default is '+str(default_params['psF']))
parser.add_argument('-nM','--number_mutants', type=int, default = default_params['nM'],
                    help='The number of mutants (per tup) that we simulate in this run. Default '
                         +str(default_params['nM']))
parser.add_argument('-l', '--lande', help='If we are using params from a simulation folder, this leads us to use a lande'
                          'instead of the actual distance. Default is to use already existing D(t)', action="store_true")
parser.add_argument('-oL', '--only_Lande', help='Leads us to use run only trajectories with Lande distance and without'
                          'a distance function obtained from population simulations. Also without a Lande D(t) corresponding '
                                                'to parameters already used in some population simulations.'
                                                ' Default is not to do this', action="store_true")

parser.add_argument('-start_time', type=int, default = default_params['start_time'],
                    help='The time from which we are recording mutations. If the start time is zero,'
                         ' we consider all muts segregating at time of shift. If the start time is greater than zero, '
                         'we consider new mutations that appear at that time.'
                         '. Default is ' +str(default_params['start_time']))


# We need the following ONLY if we are running trajectories with ONLY Lande D(t) (PURE Lande)
# If  we are reading D(t) from simulations in a directory that already exists,
# then we will read the paramters used to generate sigma_0_del, sso no need to specify the below.
# And if you do specify it in this case, it will have no effect.
parser.add_argument('-sigma_0_del', type=float, default = default_params['sigma_0_del'],
                    help='Steady-state standard deviation of the phenotype distribution, sigma_0,'
                         ' in units of delta= root(Vs/(2N)). The default is ' + str(default_params['sigma_0_del']))

# We need the following for PURE LANDE, or for if we are reading D(t) from simulations in a directory that already exists,
# but need to read the simulation folder from the parent simulation folder
# If you are not using Pure Lande, have not specified a parent simulation folder, and have specified the simulation_Folder
# then you don't need to specify the following
# And if you do specify it in this case, it will have no effect.
parser.add_argument('-N', type=int, default = default_params['N'],
                    help='The population size. Default is ' +str(default_params['N']))
parser.add_argument('-Vs', type=float, default = default_params['Vs'],
                    help='Free parameter in fitness function.  If you choose Vs< 0, the simulator will use Vs = 2N. '
                         'We use as default ' +str(default_params['Vs']))
parser.add_argument('-D_s0','--shift_s0', type=float, default = default_params['shift_s0'],
                    help='The lande dist in terms of sigma_0. The default is ' + str(default_params['shift_s0']))

# We DO NOT need the following for PURE LANDE, but we do need it if we are reading D(t) from a parent_Simulation_Folder
parser.add_argument('-E2Ns', type=float, default = default_params['E2Ns'],
                    help='The expected steady-state scaled selection co-efficient of a new mutation (or E[a^2] in units'
                         ' of delta^2=Vs/(2N)).The scaled selection coefficients of new muations are gamma distributed.'
                         ' Default is ' + str(default_params['E2Ns']))
parser.add_argument('-V2Ns', type=float, default = default_params['V2Ns'],
                    help='The variance in steady-state scaled selection co-efficient of a new mutation. When this '
                         'negative then V_2Ns = E_2Ns**2, implying the distribution of scaled selection coefficients '
                         'of new mutations is exponentially distributed. Default is ' + str(default_params['V2Ns']))
parser.add_argument('-U', type=float, default = default_params['U'],
                    help='The mutation rate. The default is ' +str(default_params['U']))

# parser.add_argument('-Un', '--Uniform', help='Leads us a uniform prior on the initial MAFs'
#                           'Most useful for late times', action="store_true")

### Don't recommend changing the defaults for any of the parameters below, unless you really understand
# the details of how the simulator works
parser.add_argument('-s_min', type=float, default = default_params['s_min'],
                    help='The min scaled selection coefficient (2Ns). Default is'+ str(default_params['s_min']))
parser.add_argument('-s_max', type=float, default = default_params['s_max'],
                    help='The max scaled selection coefficient (2Ns). Default'
                         'is' + str(default_params['s_max']))

parser.add_argument('-nXi_s','--nXi_short', type=int, default = default_params['nXi_s'],
                    help='The number of specific initial frequencies included. '
                         'Chosen to have evenly spaced contribution to change in mean over the short term.'
                         'So chosen according to contribution to phenotypic variance'
                         'If this and -nXi_l are zero, then we integrate over all frequencies.'
                         'Default is ' +str(default_params['nXi_s']))

parser.add_argument('-nXi_l','--nXi_long', type=int, default = default_params['nXi_l'],
                    help='The number of specific initial frequencies included. '
                         'Chosen to have evenly spaced contribution to change in mean over the long term.'
                         'So they are evenly spaced'
                         'If nonzero this overides -nXi_s'
                         'If this and -nXi_s are zero, then we integrate over all frequencies.'
                         'Default is ' +str(default_params['nXi_l']))

parser.add_argument('-nT','--nTheta', type=int, default = default_params['nT'],
                    help='The number of specific acute angles from trait of interest. '
                         'If zero, then we integrate over all angles.'
                         'Default is ' +str(default_params['nT']))

MY_MAX_TIME = 10**6

mid_time_N = 5

# save the arguments entered in the terminal
args = parser.parse_args()

# if args.Vs<0, set it to be 2N
if args.Vs < 0:
    args.Vs = float(2*args.N)

# In this case, there exists no D(t) from prior sims. We NEED to use Lande D(t)
if args.only_Lande:
    args.lande = True
    # Set the Lande trajectory simulator to have specific parameters
    popSimulator = SimulateTrajectories(parent_simulation_folder=args.lande_Folder,shift_s0=args.shift_s0, N=args.N, Vs=args.Vs,
                                       mid_time_N=mid_time_N)
# In this case, there IS a D(t) from population sims. So we can either use that D(t), or use the Lande D(t) corresponding to the
# pop size, shift size and steady-state phenotypic variance for the parameters used in those simulations. In either case, we
# save the results in the same directory as the previous simulations in a folder called 'trajectories'
else:
    # if there is a parent simulation folder
    # we locate a simulation folder inside it,
    # with simulations run with the correct paramaters
    if args.parent_simulation_Folder:
        # if args.V2Ns<0, make effect dist exponential
        if args.V2Ns < 0:
            args.V2Ns = args.E2Ns ** 2
        param_dict = dict()
        param_dict['N'] = args.N
        param_dict['Vs'] = args.Vs
        param_dict['shift_s0'] = args.shift_s0
        param_dict['U'] = args.U
        param_dict['E2Ns'] = args.E2Ns
        param_dict['V2Ns'] = args.V2Ns
        sim_folder = find_name_of_sim_pD_folder(args.parent_simulation_Folder,param_dict)
        if sim_folder:
            args.simulation_Folder = os.path.join(args.parent_simulation_Folder,sim_folder)
        else:
            raise Exception('No population sim results with the specified parameters in '+ args.parent_simulation_Folder)

    # Set the trajectory simulator to read parameters andr D(t) from a full simulation folder
    if not args.lande:
        popSimulator = SimulateTrajectories(simulation_folder=args.simulation_Folder, nonlande=True,mid_time_N=mid_time_N)
    # Set the trajectory simulator to read parameters from a population simulation folder, and use those to produce
    # a lande D(t)
    else:
        # use the sigma_0_del for the parameters used to generate simulations in the args.simulation_Folder,
        # and run Lande simulations with those parameters
        popSimulator = SimulateTrajectories(simulation_folder=args.simulation_Folder, nonlande=False,
                                            mid_time_N=mid_time_N)

# produce the lists of effect sizes, MAFs and acute angles to create the tuples
popSimulator.make_lists_for_tuples(s_min=args.s_min, s_max =args.s_max, nXi_short=args.nXi_short, nXi_long=args.nXi_long, nTheta=args.nTheta,
                                   start_time=args.start_time)
# for each tupple, add args.number_mutants alleles to the trajectory simulator
popSimulator.add_tuples_to_the_trajectory_class(num_muts=args.number_mutants)

# Shift the optimum and run for many generations
popSimulator.shift_opt_and_run()
# Save the statistics collected at various sample times by
# this simulator
popSimulator.save_current_trajectory_stats()
# Save the final statistics (final means after all alleles are fixed or extinct)
# collected in this simulator
popSimulator.save_current_trajectory_final_stats()