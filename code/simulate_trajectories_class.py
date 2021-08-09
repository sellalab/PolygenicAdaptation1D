import copy
import os
from collections import defaultdict
from record_stats import statWriter
import numpy as np
from scipy.interpolate import interp1d
from read_data import dataClassFullSims
from combined_theory import VarDistrMinor
from trajectory_class import TrajectoriesBasic

class SimulateTrajectories(object):
    """
    Class to store one set of simulation parameters;
    evolve many trajectories using the D(t) corresponding to
    those simulation parameters (either Lande or nonLande);
    and save the resulting statistics to a directory specific
    to those sim parameters.
    If 'simulation_folder' is not None
    we are using parameters that we already used
    to run full population simulations, and the results
    will be saved in that directory in a folder called
    'trajectories'. Otherwise the trajectories will
    be run with a Lande D(t) and saved in a folder inside
    the 'parent_simulation_folder' directory

    Parameters:
        N: (int) Population size
        Vs: (float) Fitness parameter (if Vs<0, Vs becomes 2N)
        shift_s0: (float) shift size in units
                steady-state phenotypic SD
        max_sample_time_N: (int) The last time (after shift) in units
                    population size (N) that we sample statistics
        mid_time_N: (float) The time after which we start
                    checking if there are still segregating
                    mutations. If there are none, we stop simulating.
        simulation_folder: (str) The full path of the folder in which
                        the population sims with corresponding parameters are saved. If
                        this is not 'None' then the parameter values (i.e. N, sigma_0_del, etc)
                        that we input will be irrelevant because we read them from the simulation
                        folder
        parent_simulation_folder: (str) The full path of the folder in which we want to save
                        the results folder. We use this if we are running allele trajectories with
                        only a Lande D(t)
        nonlande: (bool) True if we use a nonLande D(t) obtained from population
                    simulations (that is using
    """
    def __init__(self, simulation_folder=None, parent_simulation_folder=None,nonlande=True, shift_s0=2, N=5000, sigma_0_del=20, Vs=-1, max_sample_time_N=11,mid_time_N=5):

        # dictionary of population genetics parameters corresponding to these simulations
        self._param_dict_basic = dict()
        # dictionary of population genetics parameters corresponding to these simulations, which
        # are only relevant to trajectory simulations
        self._param_dict_traj = dict()

        if simulation_folder is not None:
            if nonlande:
                self._LANDE = False
            else:
                self._LANDE = True

            self._USE_SIMULATION_FOLDER = True
            self._SIMULATION_FOLDER = simulation_folder
            self._PARENT_SIMULATION_FOLDER = os.path.split(os.path.abspath(self._SIMULATION_FOLDER))[0]
            # Class to read parameters and D(t) from population sims
            self.data_class = dataClassFullSims(base_dir=self._SIMULATION_FOLDER)
        else:
            # if there is no D from sims, we MUST use a lande D(t)
            self._LANDE = True
            self._USE_SIMULATION_FOLDER = False
            self._PARENT_SIMULATION_FOLDER = parent_simulation_folder
            self.N = N
            # if Vs<0, set it to be 2N
            if Vs < 0:
                self.Vs = float(2 * self.N)
            else:
                self.Vs = Vs
            self.shift_s0 = shift_s0
            self._DELTA_UNIT = self.get_delta_unit()
            var_0 = (sigma_0_del * self._DELTA_UNIT) ** 2
            # if you choose Vs = 2N, then _VAR_0 = _VAR_0_UNITS_DEL_SQUARE
            self._VAR_0, self._shift = var_0, float(self.shift_s0) * np.sqrt(var_0)
            self._VAR_0_UNITS_DEL_SQUARE = sigma_0_del**2

        # input parameter values in the parameter dictionaries
        # Create the D(t) we will use to simulate trajectories
        self._initiate_param_dicts()
        self._make_distance_function()

        # true if I want to record the
        # change per gen of stats
        self._RECORD_DELTA_STATS = False

        # After the mid_time we start checking if there are still segregating mutations
        # If there are none left, we stop simulating
        self._MID_TIME = int(mid_time_N * self.N)
        self._MAX_SAMPLE_TIME = int(max_sample_time_N * self.N)
        self._MID_TIME_UNITS_N = float(self._MID_TIME) / float(self.N)

        # No need to sample statistics before we've added mutations
        # Therefore record the time (after the shift) at which mutations are added
        # I always use time 0
        self._CURRENT_START_TIME = 0

        # Generate the times (after the shift)
        # at which you will want to sample statistics
        self._make_sampling_times()

        # Create the writer which will record the statistics in a folder
        self._make_the_stat_writer()

        # create the special scaling factor to multiply the contribution to mean phenotype stat by
        the_quantity = float(self._shift)*self._DELTA_UNIT/float(self._VAR_0) # the_quantity is the integral over time
        # from 0 to infinity of (1/(2N)*D_{L} (t)/delta_unit. It is unitless
        self._SPECIAL_SCALING_FACTOR = 1.0/the_quantity

        # create the trajectory class, which will simulate the trajectories
        self.refresh_trajectory_class()

    def add_tuples_to_the_trajectory_class(self,num_muts):
        # Add all the mutations to the trajectory class

        for si in self.s_list:
            for xi in self.x_minor_list[si]:
                # xi = -Delta_s0 if integrating over all MAFs
                # (S,x,t)
                self._trajectory_class.add_muts(si,  xi, nmuts=num_muts, time=self._CURRENT_START_TIME)

        # print('Trajectory class number of seg: ')
        # print(self._trajectory_class.num_segregating)
        self._TUPLES = self._trajectory_class.get_current_tuples()
        if len(self._TUPLES) > 0:
            self._THERE_ARE_MUTANTS = True

    def get_delta_unit(self):
        """Returns little delta = sqrt(Vs/2N)"""
        return np.sqrt(self.Vs)/np.sqrt(2.0*self.N)

    def shift_opt_and_run(self,runtime=None):
        """
        Shift the optimum and evolve the mutant pairs for runtime generations
        """
        run_time_over = False
        if runtime is None:
            end_time = 0
        else:
            end_time = runtime + self._CURRENT_START_TIME

        tM = self._CURRENT_START_TIME

        while self._THERE_ARE_MUTANTS and not run_time_over:
            # If we are recording delta stats, we update moments the generation before recording
            if self._RECORD_DELTA_STATS:
                if tM + 1 in self._SET_OF_SAMPLE_TIMES_AFTER_SHIFT:
                    self._trajectory_class.update_moms_to_record_delta_moms()

            # Collect statistics at sample_times_after_shift
            sample_time = False
            if tM in self._SET_OF_SAMPLE_TIMES_AFTER_SHIFT or tM == self._CURRENT_START_TIME:
                sample_time = True
                self._record_current_stats(nowtime=tM)

            self._trajectory_class.next_gen(distance = self.my_distance_function(tM))

            tM +=1

            # If we have a set runtime, stop when it is over
            if end_time > 0:
                if tM > end_time:
                    run_time_over = True

            # Every now and then, check if there are still mutations
            # Halt simulations if all mutations are fixed/extinct
            if sample_time or tM % self.N == 0:
                if tM > self._MID_TIME:
                    if not self._trajectory_class.are_there_segregating_mutations():
                        print('ceasing at generation ', tM)
                        self._THERE_ARE_MUTANTS = False

            # Can remove this later. Just to watch how many muts still segregating
            if tM % 10000 == 0:
                print('Time:')
                print(tM)
                print('Num seg:')
                for tup in self._trajectory_class.num_segregating:
                    if self._trajectory_class.num_segregating[tup]['pos'] > 0:
                        print(tup, ': ', self._trajectory_class.num_segregating[tup])

    def get_param_dict_stat_writer(self):
        """Returns the parameter dictionary that is used by the
        statistic writer to create a hash to save the directory"""
        if self._USE_SIMULATION_FOLDER:
            return self.data_class.stat_writer_param_dict()
        else:
            param_dict = copy.deepcopy(self._param_dict_basic)
            return param_dict

    def refresh_trajectory_class(self):
        """
        Replace the current trajectory class with a new one, with the same statwriter
        and stat dictionary.
        """
        # Set up the trajectory class
        # Always using variance sampling for initial frequencies now, so 'uniform' is False

        self._trajectory_class = TrajectoriesBasic(N=self.N, Vs=self.Vs, uniform=False)
        self._THERE_ARE_MUTANTS = False # no mutations yet

        # set up the special scaling factor
        self._trajectory_class.set_up_special_scaling_factor(special_scaling_factor=self._SPECIAL_SCALING_FACTOR)

        # [tup][moment][my_time]['pos'/'neg'/'both']['sum'/'sum_squares'/'num'] = statvalue
        self._current_stat_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))


    def save_current_trajectory_stats(self):
        """
        Save the trajectory statistics recorded for the current run.
        """
        # save a read_me.text type explanation of how the results are saved in the 'trajectories' folder
        stat_info_dict = self._trajectory_class._moment_explanation_dict
        self._the_stat_writer.write_explanation(stat_info_dict, 'traj')

        # Write pickle files with _current_stat_dict[my_time] = statvalue
        for tup in self._TUPLES:
            for stat in self._current_stat_dict[tup]:
                    # 'Save a dictionary of _current_stat_dict[my_time] =' + stat)
                    # Turn to a normal dictionary for pickling and picke it
                    try:
                        key = self._the_stat_writer.get_traj_key(stat, tup)
                    except:
                        print("the stat: ")
                        print(stat)
                        print("the tuple: ")
                        print(tup)
                    self._the_stat_writer.write(key, 'traj', self._LANDE, self._default_to_regular(self._current_stat_dict[tup][stat]))

        self._the_stat_writer.close_writers()

    def save_current_trajectory_final_stats(self):
        """
        Save the final trajectory statistics recorded for the current run.
        """
        # get the final stats
        # only record them all muts are fixed or extinct, else, they are
        # not final
        if not self._THERE_ARE_MUTANTS:

            # save a read_me.text type explanation of how the final stat results are saved in the 'trajectories' folder
            stat_info_dict = self._trajectory_class.final_stats_explanations()
            self._the_stat_writer.write_explanation(stat_info_dict, 'traj_tfD')

            final_stats = self._trajectory_class.final_stats()
            # Write pickle files with dict[my_time] = statvalue
            for tup in self._TUPLES:
                # record the final stats
                key = self._the_stat_writer.get_traj_key('final_stats', tup)
                self._the_stat_writer.write(key, 'traj_tfD', self._LANDE, self._default_to_regular(final_stats[tup]))

            self._the_stat_writer.close_writers()


    def make_lists_for_tuples(self, s_min=None, s_max =None, nXi_short=0, nXi_long=0, nTheta=0, start_time=0):

        self._CURRENT_START_TIME = start_time

        # creating the list of S's
        s_list_exponents = np.linspace(-1, 2, 13)
        s_list = [10.0 ** exponenti for exponenti in s_list_exponents]
        if s_min is not None:
            self.s_list = [ss for ss in s_list if ss > s_min]
        if s_max is not None:
            self.s_list = [ss for ss in s_list if ss < s_max]

        self._make_minor_allele_frequency_list(nXi_short=nXi_short, nXi_long=nXi_long)


    def _make_minor_allele_frequency_list(self,nXi_short=0, nXi_long=0):
        # create the list of minor allele frequencies
        self.x_minor_list = dict()
        if nXi_short > 1 and nXi_long > 1:
            nXi_short = 0  # spacing x's for long-term overides spacing for short-term

        # if we are not looking at standing variation then every mutation has initial freq 1
        if self._CURRENT_START_TIME != 0:
            nXi_short = 0
            nXi_long = 0

        # In this case we use specific initial frequencies
        if nXi_short > 1 or nXi_long > 1:
            self.SPECIFIC_FREQS = True
            denom = 1.0 / float(2.0 * self.N)

            x_perc_bins_minor = np.linspace(0, 0.5, num=nXi_short + 1)
            f_index_half = len(x_perc_bins_minor) - 1
            x_perc_bins_minor = np.delete(x_perc_bins_minor, [0, f_index_half])

            for ss in self.s_list:
                if nXi_long > 1:
                    x_minor_list_prelim = [xx for xx in x_perc_bins_minor]
                else:
                    x_minor_dist_rel_cont = VarDistrMinor(N=self.N, S=ss)
                    x_minor_list_prelim = [x_minor_dist_rel_cont.ppf(q) for q in x_perc_bins_minor]

                # Make sure the minor x's correspond with actual discrete frequencies
                freq_set = set([round(2.0 * self.N * xx) for xx in x_minor_list_prelim])  # remove repeated freqeuncies
                freq_list = [ff for ff in freq_set if 1 < ff < self.N]
                freq_list = sorted(freq_list + [1.0])  # add frequency 1 (x=1/(2N))
                x_minor_list_final = sorted([denom * ff for ff in freq_list])
                self.x_minor_list[ss] = x_minor_list_final
        # In this case, we integrate over all possible initial frequencies
        else:
            self.SPECIFIC_FREQS = False
            for ss in self.s_list:
                self.x_minor_list[ss] = [-self.shift_s0]

    def _make_distance_function(self):

        # If we are using results from sims, we use mu_3(t)/(2V_A(t)), as a substitute for D(t) after this time,
        # because it varies less between simulations
        self._TIME_FROM_WHICH_WE_USE_DIST_GUESS = self._get_time_from_which_we_use_dist_guess()

        if not self._LANDE:
            self._TIME_FROM_WHICH_WE_USE_DIST_GUESS = self._get_time_from_which_we_use_dist_guess()
            self.my_distance_function = self._get_distance_function_simulations(
                time=self._TIME_FROM_WHICH_WE_USE_DIST_GUESS)
            print('time_from_which_use_dist_guess: ', self._TIME_FROM_WHICH_WE_USE_DIST_GUESS)
        else:
            self._TIME_FROM_WHICH_WE_USE_DIST_GUESS = 0
            self.my_distance_function = self._my_lande_function_D

        # We need to record the time that we started using the guess for the distance function
        self._param_dict_traj['time_from_which_use_dist_guess'] = self._TIME_FROM_WHICH_WE_USE_DIST_GUESS



    def _get_time_from_which_we_use_dist_guess(self):
        # JUST FOR NOW - Find a better way to choose the time
        # The time is the time at which we start using dist guess instead of dist
        # We start at 2*(the time from which Lande D(t) would be 0.1 little delta from the optimum
        return 2*int(-np.log(0.1*self._DELTA_UNIT / self.shift_s0)*self.Vs / self._VAR_0)

    def _get_distance_function_simulations(self, time=None):
        """Returns the function which yields distance from
        the optimum as a function of time in units of w.
        If the distance function is based on the results of
        simulations then the function switches to using
        the guess for the distance (mu3/(2*var)) at
        the_time"""
        if self._USE_SIMULATION_FOLDER and not self._LANDE:
            # Make the distance function
            if time is None:
                dist_dict = self.data_class.get_distance_dict()
            else:
                dist_dict = self.data_class.get_distance_dict(time)
            # Set distance before shift to zero
            for t in dist_dict:
                if t < 0:
                    dist_dict[t] = 0
            times = sorted(dist_dict.keys())
            self.max_time = max(times)
            distances = [dist_dict[ti] for ti in times]
            # set distances after the maximum time to zero
            extra_times_after = list(range(self.max_time + 1, 2 * self.max_time, 10)) + [10000 * self.max_time]
            extra_dist_after = [0 for _ in extra_times_after]
            times = [-self.max_time,-1] + times + extra_times_after
            distances = [0,0] + distances + extra_dist_after
            dist_function = interp1d(times, distances)
            return dist_function
        else:
            return self._my_lande_function_D

    def _my_lande_function_D(self,time):
        if time >= 0:
            exp_decay = self._VAR_0 / float(self.Vs)
            ans = self._shift*np.exp(-exp_decay*time) #*self.units
        else:
            ans = 0.0
        return ans

    def _default_to_regular(self,d):
        """Changes a defaultdict to a regular dictionary (use it for pickling
        because I can't pickkle a defaultdict)
        """
        if isinstance(d, defaultdict):
            d = {k: self._default_to_regular(v) for k, v in d.items()}
        return d

    def _record_current_stats(self,nowtime):
        """
        Records the current time and current statistics values for
        the current population the statistics dictionary (_current_stat_dict)
        """

        # Basic_stats returns a dictionary with current population stats
        basic = self._trajectory_class.basic_stats()

        # Store stats in in the stat dictionary
        for tup in basic:
            for stat in basic[tup]:
                self._current_stat_dict[tup][stat][nowtime] = copy.deepcopy(basic[tup][stat])

        # Sample the delta stats too.
        if self._RECORD_DELTA_STATS:
            basic = self._trajectory_class.delta_stats()
            # Store the delta stats in the stat dict
            for tup in basic:
                for stat in basic[tup]:
                    self._current_stat_dict[tup][stat][nowtime] = copy.deepcopy(basic[tup][stat])

    def _make_the_stat_writer(self):
        """
        Create the unique statwriter for this set of traj parameters, and save some params
        If the trajectories are using D(t) from sims, then the statwriter will record stats
        in the same folder: self._PARENT_SIMULATION_FOLDER
        """
        # Create the unique statwriter for this set of trajectories
        stat_writer_param_dict = self.get_param_dict_stat_writer()
        print("stat paramdict: ")
        print(stat_writer_param_dict)
        self._the_stat_writer = statWriter(self._PARENT_SIMULATION_FOLDER, stat_writer_param_dict)
        # Pickle the extra parameters for traj sims
        self._the_stat_writer.write_info(info=self._param_dict_basic, info_name='param_dict_basic')
        # Record parameters
        self._the_stat_writer.write_info(info=self._param_dict_traj, info_name='param_dict_traj')

        self._the_stat_writer.close_writers()


    def _initiate_param_dicts(self):
        """
        Create the parameter dictionaries for this set of simulation parameters
        """
        if not self._USE_SIMULATION_FOLDER:
            self._param_dict_basic = dict()
            self._param_dict_basic['N'], self._param_dict_basic['Vs'] = self.N, self.Vs
            self._param_dict_basic['shift_s0'] = self.shift_s0
            self._param_dict_basic['var_0'] = self._VAR_0
            self._param_dict_basic['sigma_0_del'] = self._VAR_0_UNITS_DEL_SQUARE ** (0.5)
            self._param_dict_basic['shift'] = self._shift
            self._param_dict = copy.deepcopy(self._param_dict_basic)
        else:
            self.data_class = dataClassFullSims(base_dir=self._SIMULATION_FOLDER)
            self._param_dict_basic = self.data_class.params_basic()
            self._param_dict = self.data_class.param_dict
            self.N = self._param_dict['N']
            self.Vs = self._param_dict['Vs']
            self.shift_s0 = self._param_dict['shift_s0']
            self._shift = self._param_dict['shift']
            self._VAR_0 = self._param_dict['var_0']

            self._DELTA_UNIT = self.get_delta_unit()
            self._VAR_0_UNITS_DEL_SQUARE = self._VAR_0/self._DELTA_UNIT**2



    def _make_sampling_times(self):
        """
        Create the sets of times at which we sample population stats
        """
        sample_times_after_shift_1 = self._get_sample_times_after_shift()
        for elem in list(sample_times_after_shift_1):
            if elem > self._MAX_SAMPLE_TIME:
                sample_times_after_shift_1.discard(elem)
        self._SET_OF_SAMPLE_TIMES_AFTER_SHIFT = sample_times_after_shift_1
        self._MAX_SAMPLE_TIME = max(self._SET_OF_SAMPLE_TIMES_AFTER_SHIFT) +1
        self._MAX_SAMPLE_TIME_UNITS_N = float(self._MAX_SAMPLE_TIME)/float(self.N)
        self._param_dict['T_max'] = self._MAX_SAMPLE_TIME

    def _get_sample_times_after_shift(self):
        # The Lande time determines the rapid phase sampling timescale, unless it is too short
        self._SHORT_SAMPLE_TIME = self._short_sample_time()
        hundreth_lande_int = int(float(self._SHORT_SAMPLE_TIME) / 100.0)
        if hundreth_lande_int <= 1:
            hundreth_lande_int = 1
        # The population size determines the equilibration sampling timescale
        hundreth_N_int, tenth_N_int, half_N_int = int(float(self.N) / 100.0), int(float(self.N) / 10.0), int(float(self.N) / 2.0)
        if hundreth_N_int <= 1:
            hundreth_N_int = 1
        if tenth_N_int <= 1:
            tenth_N_int = 1

        # Rapid phase
        T_current, T_next, sampling_time_lag = 0, 100, 1  # initially sample every generation
        sample_times_after_shift = list(range(T_current, T_next + 1, sampling_time_lag))
        T_current, T_next, sampling_time_lag = T_next, T_next + self._SHORT_SAMPLE_TIME, hundreth_lande_int  # Sample 100 times during rapid phase
        sample_times_after_shift += list(range(T_current, T_next + 1, sampling_time_lag))
        T_current, T_next, sampling_time_lag = T_next, T_next + 2 * self._SHORT_SAMPLE_TIME, hundreth_lande_int  # Sample 200 times during intermediate phase
        sample_times_after_shift += list(range(T_current, T_next + 1, sampling_time_lag))

        # "Equilibration"
        T_current, T_next, sampling_time_lag = T_next, T_next + self.N, hundreth_N_int  # Sample 100 for the next N gen
        sample_times_after_shift += list(range(T_current, T_next + 1, sampling_time_lag))
        T_current, T_next, sampling_time_lag = T_next, T_next + 4 * self.N, tenth_N_int  # Sample 40 for the next 4*N gen
        sample_times_after_shift += list(range(T_current, T_next + 1, sampling_time_lag))
        T_current, T_next, sampling_time_lag = T_next, T_next + 6 * self.N, half_N_int  # Sample 12 for the next 6*N gen
        sample_times_after_shift += list(range(T_current, T_next + 1, sampling_time_lag))

        # SHORT TERM HACK FOR SMALL POPULATION
        if self.N <= 2000:
            T_current, T_next, sampling_time_lag = T_next, T_next + 9 * self.N, half_N_int  # Sample 18 for the next 9*N gen
            sample_times_after_shift += list(range(T_current, T_next + 1, sampling_time_lag))

        sample_times_after_shift = set(sample_times_after_shift)
        return sample_times_after_shift

    def _get_t_lande(self):
        """
        Returns the Lande time
        """
        return int(-np.log(self._DELTA_UNIT / self.shift_s0)*self.Vs / self._VAR_0)

    def _short_sample_time(self, min_time=100):
        """
        Returns the number of gens after the shift that we should sample stats every generation
        """
        return max(min_time,self._get_t_lande())