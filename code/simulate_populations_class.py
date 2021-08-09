import copy
from combined_theory_n_dim import normalized_density_of_contribution_from_vectors_with_acute_angle_theta_n_dim
from collections import defaultdict
from record_stats import statWriter
from population_class import MutationalProcess, PopulationExact, PopulationWF
import numpy as np
from scipy.integrate import quad
from scipy.stats import gamma
from scipy.special import dawsn

class SimulatePopulations(object):
    """
    Class to store one set of simulation parameters;
    evolve different runs of populations with those
    simulation parameters; and save the resulting
    statistics to a directory specific to those sim
    parameters.

    Parameters:
        N: (int) Population size
        Vs: (float) Fitness parameter
        E2Ns: (float) the expected steady-state
                scaled selection coeffecient of incoming mutations.
                We draw them from a gamma distribution with this expected value
        V2Ns: (float) the variance in the steady-state
                scaled selection coeffecient of incoming mutations.
                We draw them from a gamma distribution with this variance
        shift_s0: (float) shift size in units
                steady-state phenotypic SD
        U: (float) Mutational rate (per gamete per generation)
        histograms: (bool) True if we record histo data
        max_sample_time_N: (float) The last time (after shift)
                    that we sample statistics in units
                    population size (N)
        lag_time_N: (float) Burn time between runs in units of pop size (N)
    """
    def __init__(self, shift_s0=2, E2Ns=16, V2Ns=256, N=5000, U=0.01, Vs=1, histograms=False, lag_time_N=5, max_sample_time_N=11):
        self.N = N
        # if Vs<0, set it to be 2N
        if Vs < 0:
            self.Vs = float(2 * self.N)
        else:
            self.Vs = Vs
        self.U = U
        self.shift_s0 = shift_s0
        self.E2Ns= E2Ns
        self.V2Ns = V2Ns
        self._scale_s = float(self.V2Ns) / float(self.E2Ns)
        self._shape_s = float(self.E2Ns) / float(self._scale_s)
        self._DELTA_UNIT = float(self.Vs) ** (0.5) / float(2 * self.N) ** (0.5)
        self._MUT_INPUT = float(2.0 * self.N * self.U)  # Mutational input per generation
        self._VAR_0_UNITS_DEL_SQUARE = self._get_variance_units_little_del_square()
        self._VAR_0 = self._VAR_0_UNITS_DEL_SQUARE* self._DELTA_UNIT ** 2
        self._shift = self.shift_s0 * self._VAR_0_UNITS_DEL_SQUARE ** (0.5) * self._DELTA_UNIT
        self._initiate_param_dicts()
        self._HISTO = histograms # True if we record histrogram data
        self._FREEZE_NEW_MUTS = False
        self._FREEZE_MUTS = False

        # true if I want to record the
        # change per gen of stats
        self._RECORD_DELTA_STATS = False

        # True if there is a basic population
        self._BASIC_POPULATION = False
        # True if there is a current population
        self._CURRENT_POPULATION = False
        # True if this is the first current population
        # based on the basic population
        self._FIRST_CURRENT_POPULATION = True
        # True if the basic population has already had a
        # burn in to reach steady-state
        self._BURNED = False

        self._LAG_TIME = int(lag_time_N * self.N)
        self._MAX_SAMPLE_TIME = int(max_sample_time_N * self.N)
        self._LAG_TIME_UNITS_N = float(self._LAG_TIME) / float(self.N)
        self._param_dict["lag_time"] = self._LAG_TIME

        # Specifify the mutational process
        self._mu = MutationalProcess(mu=self.U, shape=self._shape_s, scale=self._scale_s)

        self._make_sampling_times()

    def burn_basic_population_to_reach_steady_state(self,burn_time_N=2):
        """
        Evolves the basic pop burn_time_N*N gens to reach steady-state
        """
        # if there was already a burn in period, no need to burn in
        if not self._BURNED:
            self._BURN_TIME = int(burn_time_N*self.N)
            self._BURN_TIME_UNITS_N = float(self._BURN_TIME) / float(self.N)
            self._param_dict['burn_time'] = self._BURN_TIME
            for time in range(self._BURN_TIME):
                # advance one generation
                self._basic_poulation_class.next_gen()
            self._BURNED = True

    def burn_current_pop(self):
        """
        Evolves the current population self._LAG_TIME generations to
        make it different from the basic population
        (unless it is the first current populion).
        We do this so our different runs are independent.
        To be safe do this a full standard burn in for the
        final simulations.
        However, when you are experimenting just to get an
        idea of how things work, I find a shorter lag time is fine
        """
        # if this is the first current pop we don't need additional burn in time
        if self._FIRST_CURRENT_POPULATION:
            start_current_burn = self._LAG_TIME - self._TIME_BEFORE_SHIFT_START_SAMPLING - 2
        else:
            start_current_burn = 0
        # we advance self._LAG_TIME- start_current_burn generations
        for time in range(start_current_burn, self._LAG_TIME):

            # If we are recording delta stats, we update moments the generation before recording
            if self._RECORD_DELTA_STATS:
                if time + 1 in self._SET_OF_SAMPLE_TIMES_BEFORE_SHIFT or time + 1 == self._LAG_TIME:
                    self._current_poulation_class.update_moms_to_record_delta_moms()

            # Collect statistics at _SET_OF_SAMPLE_TIMES_BEFORE_SHIFT
            if time in self._SET_OF_SAMPLE_TIMES_BEFORE_SHIFT:
                self._record_current_stats(nowtime=time-self._LAG_TIME)

            # Advance one generation
            self._current_poulation_class.next_gen()

    def shift_opt_and_run(self,runtime=None):
        """
        Shift the optimum and evolve for runtime generations
        """
        if runtime is None:
            T_max = self._MAX_SAMPLE_TIME
        else:
            T_max = runtime
        # Collect an empirical steady-state variance at the time of the shift
        self._CURRENT_VAR0_EMP = self._current_poulation_class.var()

        # Shift the optimum
        self._current_poulation_class.shift_optimum(self._shift)

        for tM in range(T_max):
            if self._HISTO:
                if self._FREEZE_MUTS:
                    if tM == self._t_freeze:
                        # Record which mutants are segregating at the time of freezing
                        self._current_poulation_class.freeze()
                        self._CURRENT_POP_FROZEN = True
                if self._FREEZE_NEW_MUTS:
                    if tM == self._t_freeze_new_muts_start:
                        self._current_poulation_class.freeze_new_mutations(time_length=self._t_length_freeze_new_muts)

            # If we are recording delta stats, we update moments the generation before recording
            if self._RECORD_DELTA_STATS:
                if tM + 1 in self._SET_OF_SAMPLE_TIMES_AFTER_SHIFT:
                    self._current_poulation_class.update_moms_to_record_delta_moms()

            # Collect statistics at sample_times_after_shift
            if tM in self._SET_OF_SAMPLE_TIMES_AFTER_SHIFT:

                self._record_current_stats(nowtime=tM)

            self._current_poulation_class.next_gen()

    def save_histos(self):
        if self._BASIC_POPULATION:
            self._HIST_BINS = self._basic_poulation_class.histogram_bins()

    def initiate_basic_population_classWF(self):
        self._param_dict['algorithm'] = 'approx'
        self._basic_poulation_class = PopulationWF(N=self.N, Vs=self.Vs, mu=self._mu)
        # record that we have a basic population
        self._BASIC_POPULATION = True
        # if we will need them, record the histogram bins
        if self._HISTO:
            self.save_histos()
        # record that we have a basic population

    def initiate_basic_population_classExact(self,selection_mode_offspring=False):
        """Selection mode can be by 'offspring' fitness or 'parents' fitness"""
        self._param_dict['algorithm'] = 'exact'
        if not selection_mode_offspring:
            self._param_dict['fitness'] = 'parents'
            # Sample by offspring fitness, if specified
        else:
            self._param_dict['fitness'] = 'offspring'
        self._basic_poulation_class = PopulationExact(N=self.N, Vs=self.Vs, mu=self._mu,selection_mode= self._param_dict['fitness'])
        # record that we have a basic population
        self._BASIC_POPULATION = True
        # if we will need them, record the histogram bins
        if self._HISTO:
            self.save_histos()

    def refresh_current_population(self, save_folder=None):
        """
        Replace the current population with a new one, with its own unique statwriter
        and stat dictionary. IF this is the first current population you need to specify
        a save_folder, where the results will be saved
        """

        #if there is already a current population, then this is not the first
        if self._CURRENT_POPULATION:
            self._FIRST_CURRENT_POPULATION = False

        # if this is the first current pop we need to specify a new save file
        if self._FIRST_CURRENT_POPULATION:
            if save_folder is None:
                print("Need file to save results in")
                return
        if save_folder is not None:
            # The file in which we are currently saving stats
            self._CURRENT_SAVE_FILE = save_folder

        self._CURRENT_POP_FROZEN = False

        self._current_poulation_class = self._copy_of_basic_population()
        self._make_the_current_stat_writer()

        # Make dictionary with dict[stat][my_time] =[statvalue]
        self._current_stat_dict = defaultdict(lambda: defaultdict(dict))

        self._CURRENT_POPULATION = True

    def save_stats_for_current_population(self):
        """
        Save the statistics recorded for the current run of evolution
        """
        # save a read_me.text type explanation of how the final stat results are saved in the 'trajectories' folder
        if self._HISTO:
            stat_info_dict = self._current_poulation_class.moment_explations_with_histos()
        else:
            stat_info_dict = self._current_poulation_class.moment_explations_no_histos()
        self._current_stat_writer.write_explanation(stat_info_dict, '')

        # Write pickle files with dict[my_time] = statvalue
        for stat in self._current_stat_dict:
            # 'Save a dictionary of dict[my_time] =' + stat)
            # Turn to a normal dictionary for pickling and picke
            self._current_stat_writer.write(stat, '_sD/', False, False, self._default_to_regular(self._current_stat_dict[stat]))

        self._current_stat_writer.write('var_0_emp', '_vD/', False, False, self._CURRENT_VAR0_EMP)
        self._current_stat_writer.close_writers()

    def initiate_histograms_new_muts(self,t_freeze_new_muts_start=1,t_length_freeze_new_muts=None):
        """
        Initiate the info for the histograms of statistics about new mutations that arise in
        the t_length_freeze_new_muts generations after t_freeze_new_muts_start.
        I only ever start in generation 1 (after shift)
        """
        if t_length_freeze_new_muts is None:
            self._t_length_freeze_new_muts = 5*self.N
        else:
            self._t_length_freeze_new_muts = t_length_freeze_new_muts
        self._t_freeze_new_muts_start = t_freeze_new_muts_start
        self._param_dict['t_freeze_new_muts_start'] = self._t_freeze_new_muts_start
        self._param_dict['t_length_freeze_new_muts'] = self._t_length_freeze_new_muts
        self._HISTO = True
        self._FREEZE_NEW_MUTS = True
        # if we haven't already record the histogram bins
        if self._BASIC_POPULATION and not self._FREEZE_MUTS:
            self.save_histos()

    def initiate_histograms_frozen_muts(self,t_freeze=0):
        """
        Initiate the info for the histograms of statistics about mutations that
        are segregating in generation t_freeze (after the shift).
        I only ever freeze in generation 0 (=at time of shift)
        """
        self._t_freeze = t_freeze
        self._param_dict['t_freeze'] = self._t_freeze
        self._FREEZE_MUTS = True
        self._HISTO = True
        # if we haven't already, record the histogram bins
        if self._BASIC_POPULATION and not self._FREEZE_NEW_MUTS:
            self.save_histos()

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
        basic = self._current_poulation_class.basic_stats()

        # Add dictionaries with histgrams
        if self._HISTO:
            hist = self._current_poulation_class.histo_stats(frozen=self._CURRENT_POP_FROZEN)
            basic.update(hist)

        # Store stats in in the stat dictionary
        for stat in basic:
            self._current_stat_dict[stat][nowtime] = basic[stat]

        if self._RECORD_DELTA_STATS:
            basic = self._current_poulation_class.delta_stats()
            # Store the delta stats in the stat dict
            for stat in basic:
                self._current_stat_dict[stat][nowtime] = basic[stat]

    def _copy_of_basic_population(self):
        """
        Returns a copy of the basic population
        """
        current_poulation_class = copy.deepcopy(self._basic_poulation_class)
        return current_poulation_class

    def _make_the_current_stat_writer(self):
        """
        Create the unique statwriter for this current population, and save some params
        """
        # Create the unique statwriter for this current population
        self._current_stat_writer = statWriter(self._CURRENT_SAVE_FILE, self._param_dict)
        # Pickle the extra parameters for full sims
        self._current_stat_writer.write_info(info=self._param_dict_basic, info_name='param_dict_basic')
        self._current_stat_writer.write_info(info=self._param_dict, info_name='extra_param_dict_full_sims')

        # Pickle the histogram bins in a dictionary
        if self._HISTO:
            self._current_stat_writer.write_info(info=self._HIST_BINS, info_name='hist_bins')
        #     # TEMP FOR DEBUGGING
        #     for key in hist_bins:
        #         print key
        #         print hist_bins[key]
        #         print len(hist_bins[key])
        #     print 'eff number: ',pop.eff_bin_number
        #     print 'f bin number: ', pop.f_bin_number

        self._current_stat_writer.close_writers()


    def _initiate_param_dicts(self):
        """
        Create the parameter dictionaries for this set of simulation parameters
        """
        self._param_dict_basic = dict()
        self._param_dict_basic['N'], self._param_dict_basic['Vs'] = self.N, self.Vs
        self._param_dict_basic['shift_s0'] = self.shift_s0
        self._param_dict_basic['var_0'] = self._VAR_0
        self._param_dict_basic['sigma_0_del'] = self._VAR_0_UNITS_DEL_SQUARE ** (0.5)
        self._param_dict_basic['shift'] = self._shift

        self._param_dict = copy.deepcopy(self._param_dict_basic)
        self._param_dict['U'] = self.U
        self._param_dict['E2Ns'], self._param_dict['V2Ns'] = self.E2Ns, self.V2Ns
        self._param_dict['scale_s'] = self._scale_s
        self._param_dict['shape_s'] = self._shape_s

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

        self._SET_OF_SAMPLE_TIMES_BEFORE_SHIFT = self._get_sample_times_before_shift()

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

    def _get_sample_times_before_shift(self):
        # Sample statistics for the 100 generations right before the shift
        self._TIME_BEFORE_SHIFT_START_SAMPLING = 100
        if self._LAG_TIME < 100:
            before_lag_time = 0
        else:
            before_lag_time = self._LAG_TIME - self._TIME_BEFORE_SHIFT_START_SAMPLING
        sample_times_before_shift = set(range(before_lag_time, self._LAG_TIME, 10))  #
        return sample_times_before_shift

    def _get_variance_units_little_del_square(self):
        """
        Returns steady-state phenotypic variance, in units little delta squared
        """
        SHAPE_S, SCALE_S = float(self.E2Ns) ** 2 / float(self.V2Ns), float(self.V2Ns) / float(self.E2Ns)
        S_dist = gamma(SHAPE_S, loc=0., scale=SCALE_S)
        to_integrate = lambda ss: 4.0 * np.sqrt(np.abs(ss)) * dawsn(np.sqrt(np.abs(ss)) / 2.0) * S_dist.pdf(ss)
        b = S_dist.ppf(0.99999999999999)
        myintegral = quad(to_integrate, 0, b)[0]
        var_0_del_square = myintegral * self._MUT_INPUT
        return var_0_del_square

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