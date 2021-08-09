"""This module has the container classes for the information about trajectories
In here we have the class TrajectoriesBasic
"""
from collections import defaultdict, namedtuple
import math
from mutation_class import MutationPosNeg
import numpy as np
from scipy.special import dawsn
from combined_theory import VarDistrMinor
from scipy.integrate import quad


def find_index_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def combined_mean_and_variance(mean_1,mean_2,var_1,var_2,n_1,n_2):
    mean_c = float(n_1*mean_1+n_2*mean_2)/float(n_1+n_2)
    var_c = (n_1*(var_1+(mean_1-mean_c)**2)+n_2*(var_2+(mean_2-mean_c)**2))/float(n_1+n_2)
    return mean_c, var_c

def un_normalised_weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    newvals = [vv * ww for vv, ww in zip(values, weights)]
    average = np.average(newvals)
    # Fast and numerically precise:
    variance = np.average([(nv - average)**2 for nv in newvals])
    return average, math.sqrt(variance)

def un_normalised_weighted_avg_and_std_and_third_moment(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    newvals = [vv*ww for vv,ww in zip(values,weights)]
    average = np.average(newvals)
    # Fast and numerically precise:
    variance = np.average([(nv - average)**2 for nv in newvals])
    third_moment = np.average([(nv - average)**3 for nv in newvals])
    return average, math.sqrt(variance), third_moment

def weighted_avg_and_std_and_third_moment(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average([(vv - average)**2 for vv in values], weights=weights)
    third_moment = np.average([(vv - average)**3 for vv in values], weights=weights)
    return average, math.sqrt(variance), third_moment

#Change a defaultdict to a regular one for pickling
def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


# parameters for the mutational process
# mu: haploid mutation rate per-generation
# shape, scale: gamma-distribution parameters, describing the effect size distribution
MutationalProcess = namedtuple('MutationalProcess', 'mu shape scale')


class TrajectoriesBasic(object):
    """
    Class to store and evolve mutations

    Parameters:
        N: (int) Population size
        Vs: (float) Fitness parameter
        mu: (namedtuple) MutationalProcess (mu, shape, scale)
        uniform: (bool) True if we use uniform importance sampling.

    Attributes:
        N: (int) Population size
        Vs: (float) Fitness parameter
        _moments: (list) The basic quantities - mean, var, dist, mean_seg, mean_fixed
                    - of the phenotype distribution
        _delta_moments: (list) Stats that we record the change in.

        _moments_new_d: (dict) Storing the current value of all the stats in _stat_set
        _old_moments_new_d: (dict) Storing the value in the previous generation of all the stats in _delta_statistics

        _segregating: (set) A set containing all the mutations currently segregating in the population
        _frozen: (set) Set of mutations segregating at freeze my_time

        _DENOMINATOR: (float) 1/(2*N)
        _FITNESSCOEF: (float) 1/(2*Vs)
        _MU_SCALING_COEFF: (float) Mutation scaling parameter (2w**2)/N
"""
    def __init__(self, N, Vs, uniform =None):

        self.N = N # population size
        # Vs is the squared width of fitness function (dummy variable)
        # if Vs<0, set it to be 2N
        if Vs < 0:
            self.Vs = float(2 * self.N)
        else:
            self.Vs = Vs

        # Will be a list of frequencies chosen with uniform prior. Used so so we don't resample for every selection coefficient
        self.my_random_frequencies = []

        if uniform is None:
            # We use importance sampling to choose initial frequencies. If uniform is false with we choose them with
            # probability proportional to their equilibrium contribution to phenotypic variance
            # If uniform is true, we use importance sampling with a uniform prior on alleles initial frequencies
            # One could this scheme to learn about fixations/long term behavior
            self.uniform_prior_frequencies = False
        else:
            self.uniform_prior_frequencies = uniform

        # The number of generations since we last removed the fixed and extinct mutations from the the segregating list
        self.last_removal_of_extinct_fixed_from_seg = 1
        # The default number of generations after which we remove fixed and extinct mutations from the segregating list
        self.how_often_to_remove = 100

        # the number of digits that we round frequencies, selection coefficients when we save them
        self.round_XI = 5
        self.round_S = 3

        # A dictionary storing the normalising constants for the sojourn time,
        # given a particular scaled selection coefficient
        self.half_tau_norm = dict()

        # The beginning 'pos' and 'neg' denote the contribution from alleles with aligned (pos)
        # and opposing (neg) phenotypic effects. 'both' denotes the contribution from both aligned
        # and opposing
        self._starts = ['pos','neg','both']

        self.set_up_the_stats_to_collect()

        # for each stat we collect the sum and sum squares of the stat across alleles, and the num of alleles
        self._values = ['sum','sum_squares','num']

        #dictionary with keys tuples #(S,x,t) and values sets of mutants
        self._segregating = defaultdict(set)

        self.tuples_pos = set()

        # [freq] = percentile
        self.freqency_to_percentile_dict = dict()
        # [effect_size] = percentile
        self.effect_size_to_percentile_dict = dict()

        # dictionaries with keys #(S,x,t) #(S,x,t):'pos'/'neg'/'both'
        self.mean_initial_frequency = defaultdict(dict)

        self.num_fixed = defaultdict(dict)
        self.num_extinct = defaultdict(dict)
        self.num_segregating = defaultdict(dict)
        self.NUM_MUTANTS = defaultdict(dict)

        self.sum_weighted_initial_frequency = defaultdict(dict)
        self.sum_fixed_weights = defaultdict(dict)
        self.sum_extinct_weights = defaultdict(dict)
        self.sum_all_weights = defaultdict(dict)
        self.sum_squares_fixed_weights = defaultdict(dict)
        self.sum_squares_extinct_weights = defaultdict(dict)
        self.sum_squares_fixed_weights_dx = defaultdict(dict)
        self.sum_squares_extinct_weights_dx = defaultdict(dict)


        #[tuple_pos][moment]['pos'/'neg'/'both']['sum'/'sum_squares'/'num']
        self._moments_new_d = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self._old_moments_new_d = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        #useful factors to multiply stuff with
        self._SCAL =  1.0 /  float(self.N)
        self._DENOMINATOR = 1.0 / (2.0 * float(self.N))

        # mutation scaling parameter. Scaled selection coefficients are squared HETEROZYOGOTE effect sizes of alleles
        # in units of delta^2 =Vs/(2N) = 1. We use _MU_SCALING_COEF to do two things. 1) To convert these squared
        # HETEROZYOGOTE effect sizes to squared HOMOZYGOTE effect sizes, by multiplying by 4. And 2) to change the
        # squared effect sizes into the units in which Vs = self.Vs. For example, if  self.Vs =1, then _MU_SCALING_COEF
        # is used to convert to squared homozygote effect sizes in units of Vs = 1. But if Vs = 2N, then it is used to
        # convert to squared homozygote effect sizes to units of Vs = 2N  (i.e. delta^2 =Vs/(2N) = 1). So in the second
        # example, when Vs =2N, _MU_SCALING_COEFF =4, and it is just used to change squared HETEROZYOGOTE
        # effect sizes to squared homozygote effect sizes, without changing the units
        self._MU_SCALING_COEFF = 2.0 * float(self.Vs) / float(N)

         # selection model paramter
        self._FITNESSCOEF = 0.5 / float(self.Vs)

        # if you would like to also scale the contribution to change in mean, per unit mutational input
        # by multiplying by an amount special_scaling_factor, use set_up_special_sclaing_factor
        self._SCALE_CONTRIBUTION_PER_UNIT_MUT_INPUT = False
        self._SPECIAL_SCALING_FACTOR = 1.0

    def set_up_special_scaling_factor(self, special_scaling_factor):
        """if you would like to also scale the contribution to change in mean, per unit mutational input
        by an amount special_scaling_factor"""
        if special_scaling_factor != 0:
            if 'U1_d2ax_per_mut_input' in self._moments:
                self._SCALE_CONTRIBUTION_PER_UNIT_MUT_INPUT = True
                self._SPECIAL_SCALING_FACTOR = float(special_scaling_factor)
                self._moments.append('U1_d2ax_scaled_per_mut_input')
                explantion = "The contribution to the change in mean phenotype, divided by the unitless quantity" \
                             " (shift*delta_unit)/V_{A} since the start time, per unit mutational input.\n " \
                             "Only interesting as a 'both' stat.\n This scaling should makes results from " \
                             "simulations different shift size, and mutation rate,\n" \
                             " but with the same effect size distribution of new muts, approximately coincide."
                self._moment_explanation_dict['U1_d2ax_scaled_per_mut_input'] = explantion


    def set_up_the_stats_to_collect(self):
        """Define the list of stats (self._moments) that we sample.
        And also the list of stats for which we collect the per gen change"""

        self._moment_explanation_dict = dict()
        self._moment_explanation_dict["dx_per_seg_var"] = "The average change in allele frequency since the start time"
        self._moment_explanation_dict["x_per_seg_var"] = "The average allele frequency "
        self._moment_explanation_dict["d2ax_per_mut_input"] = "The contribution to the change in mean phenotype since" \
                                                              " the start time, per unit mutational input"

        # The "U_" at the beggining of the stat name represents the units of that stat. E.g. a stat that starts with "U2_"
        # is in units of trait value squared, but it it starts with "U0_" it is unitless.
        moments = ['U0_dx_per_seg_var', 'U0_x_per_seg_var','U1_d2ax_per_mut_input']

        self._moments = []
        for mom in moments:
            self._moments.append(mom)


    def starts(self):
        return self._starts

    def values(self):
        return self._values

    def _add_muts(self, nmuts, tup=None):

        # If XI is less than 1 we integrate over all possible values
        # If the time is less than 1, an integration over times after the shift is occuring
        S = self.scaled_size(tup)
        XI = self.x_i(tup)
        time = self.time(tup)

        # Every mutation has the same freq, scaled s weight if we aren't integrating
        freq_and_S_weight = 1.0
        # if mutations arise before the shift sample freqs from stationary MAF dist for that S=2Ns
        if time == 0:
            if XI > 0:
                frequency = int(round(2 * self.N * XI))
                freq_list = [frequency for _ in range(nmuts)]
                freq_and_S_weight = self._half_folded_tau(x=XI, S=S)
            else:
                if self.uniform_prior_frequencies:
                    freq_list = self._get_random_freqs_uniform_prior(nfreqs=nmuts)
                else:
                    freq_list = self._generate_random_frequencies_var_distr_prior(num=nmuts,S=S)
                    # We are conditioning on the allele segregating
                    numzeros = freq_list.count(0)
                    for _ in range(numzeros):
                        freq_list.remove(0)
                        freq_list.append(1)
        # if mutations arise after shift they are new muts and start at frequency 1
        else:
            freq_list = [1 for _ in range(nmuts)]


        total_weights = 0.0
        sum_weighted_freqs = 0.0
        sum_weighted_freqs_squares = 0.0

        for freq in freq_list:

            # the weight of the mutation in the initial frequency_pos distributionw
            if time == 0:
                xii = float(freq) * self._DENOMINATOR
                if XI <= 0: #if we are integrating (w/ importance sampling) over initial frequencies
                    if self.uniform_prior_frequencies:
                        weight_freq_and_S = self._get_muation_weight_xS_uniform_prior(x=xii, S=S)
                    else:
                        weight_freq_and_S = self._half_folded_tau_over_folded_var_freq_dist(x=xii, S=S)
                else: # if we aren't integrating weight all initial frequencies by initial density
                    weight_freq_and_S = freq_and_S_weight
            else:
                # the new mutation starts at frequency 1 and has freq weight 1
                weight_freq_and_S = 1

            weight = weight_freq_and_S

            mut = self._get_mutation(S, freq, weight)

            weighted_freq = mut.x_initial()*mut.weight

            sum_weighted_freqs += weighted_freq
            sum_weighted_freqs_squares += weighted_freq**2
            total_weights += mut.weight

            self._segregating[tup].add(mut)

        mean_init_freq = sum_weighted_freqs/float(nmuts)

        # Initialize the tup if it doesn't exist yet
        if not tup in self.tuples_pos:
            # Add the tup
            self.tuples_pos.add(tup)
            #starts = ['pos', 'neg','both']
            for st in self._starts:
                if st != self._starts[-1]:
                    if time == 0:
                        self.mean_initial_frequency[tup][st] = mean_init_freq
                        self.sum_weighted_initial_frequency[tup][st] = sum_weighted_freqs
                    else:
                        # for the purpose of recording statistics
                        # pretend mutations after the shift started at frequency 0
                        self.mean_initial_frequency[tup][st] = 0
                        self.sum_weighted_initial_frequency[tup][st] = 0
                else:
                    self.mean_initial_frequency[tup][st] = 0
                    self.sum_weighted_initial_frequency[tup][st] = 0

                self.NUM_MUTANTS[tup][st] = nmuts
                self.num_segregating[tup][st] = nmuts
                self.num_extinct[tup][st] = 0
                self.num_fixed[tup][st] = 0

                self.sum_fixed_weights[tup][st] = 0
                self.sum_extinct_weights[tup][st] = 0
                self.sum_all_weights[tup][st] = total_weights

                self.sum_squares_fixed_weights[tup][st] = 0
                self.sum_squares_extinct_weights[tup][st] = 0
                self.sum_squares_extinct_weights_dx[tup][st] =0
                self.sum_squares_fixed_weights_dx[tup][st] = 0

                # ['pos'/'neg'/'both']['sum','num','sum_squares']
                # Initialise the moment dictionaries
                for m in self._moments:
                    self._moments_new_d[tup][m][st]['sum'] = 0.
                    self._moments_new_d[tup][m][st]['sum_squares'] = 0.0
                    self._moments_new_d[tup][m][st]['num'] = float(nmuts)

        else: # This is if we are looking at new mutations after the shift
            # starts = ['pos', 'neg','both']
            for st in self._starts:
                self.NUM_MUTANTS[tup][st] += nmuts
                self.num_segregating[tup][st] += nmuts
                self.sum_all_weights[tup][st] += total_weights

                # Add the new mutants to the number of mutants
                for m in self._moments:
                    self._moments_new_d[tup][m][st]['num'] += float(nmuts)

        self._remove_extinct_fixed_from_seg()

    def get_current_tuples(self):
        return self.tuples_pos

    def add_muts(self,S, XI,nmuts,time = None):
        """
        Add nmuts new mutants with frequency XI and scaled selection coeff S
        If XI <= 0, then average over the initial MAF
        """
        if S < 0:
            S = -S

        if XI <=0:
            integrate_xi = True # this means we integrate over all initial MAFs
        else:
            integrate_xi = False

        if time is None:
            time = 0
        # If time < 0, then integration over time is occuring

        if not integrate_xi:
            frequency = 2*self.N*XI
            frequency = int(round(frequency))
            XI = round(frequency*self._DENOMINATOR,self.round_XI)
            #S = round(S, self.round_S)
            if XI <= 0.0 or XI >= 1:
                print('x_i is '+ str(XI) + ' for S= '+ str(S))
                return

        tuple_pos = (S,XI,time)

        if tuple_pos in self.tuples_pos:
            # We can only add mutations multiple times if time < 0.
            # In this case we are averaging over many times after the shift
            if time >= 0:
                print(str(tuple_pos)+' already in self.tuples')
                return

        self._add_muts(nmuts, tup=tuple_pos)


    @staticmethod
    def number_gens_of_mut_input(tup):
        """
        Returns the number of generations of time we are integrating over.
        """
        if tup[2] >= 0:
            return 1.0
        else:
            return np.abs(tup[2])

    @staticmethod
    def scaled_size(tup):
        """
        Returns the scaled squared size of mutants for this tup.
        """
        return np.abs(tup[0])

    @staticmethod
    def x_i(tup):
        """
        Returns the initial frequency_pos (in (0,1/2)) in this tup.
        """
        return tup[1]

    @staticmethod
    def time(tup):
        """
        Returns the the my_time at which mutants in this tup were added. Or the index if we don't care about the my_time
        """
        return tup[2]

    def pheno_size(self, tup):
        """
        Returns the phenotypic effect size of mutants for this tup.
        """
        S = self.scaled_size(tup)
        pheno_size = np.sqrt(self._MU_SCALING_COEFF * S)
        return pheno_size

    def frequency(self, tup):
        """
        Returns the initial fraction of the population with mutations in this tup.
        """
        XI = self.x_i(tup)
        frequency = int(2 * self.N * XI)
        return frequency


    def next_gen(self,distance = None):
        """
        Progresses to the next generation.
        """
        self.last_removal_of_extinct_fixed_from_seg += 1

        if distance is None:
            distance = 0

        #Update mutation frequencies via Wright-Fisher
        self._wright_fisher(distance)

        # Remove fixed and extinct mutations from list and updates fixed effect (meanFixed):
        if self.last_removal_of_extinct_fixed_from_seg > self.how_often_to_remove:
            self._remove_extinct_fixed_from_seg()

    def basic_stats(self):
        """
        Returns a dictionary of dictionaries of the basic stats. Keys are tuples (S, XI,my_time)
        """
        self._update_all_moments()
        return self._moments_new_d


    def final_stats_explanations(self):
        # set up a dictionary to explain what th
        final_moment_explanation_dict = dict()

        final_moment_explanation_dict['NUM_MUTANTS'] = "This is just the total number of alleles that were simulated." \
                                                       " Not really a final stat, as it was the same " \
                                                       "throughout the simulation.\n It's just useful to be " \
                                                       "able to read it from this dictionary"
        final_moment_explanation_dict['NUM_SEGREGATING'] = "This is just the total number of alleles still segregating " \
                                                       "when final stats are recorded. It should ALWAYS be zero."
        final_moment_explanation_dict['frac_fixed_per_seg_var'] = "This is the fraction of alleles in the tuple that" \
                                                                  "fixed. It can also be thought of as the long-term" \
                                                                  "frequency."
        explanation = "The final contribution to the change in mean phenotype since the start time, " \
                      "per unit mutational input"
        final_moment_explanation_dict['2a_frac_fixed_per_mut_input'] = explanation
        if self._SCALE_CONTRIBUTION_PER_UNIT_MUT_INPUT:
            explanation = "The final contribution to the change in mean phenotype, divided by the unitless quantity" \
                             " (shift*delta_unit)/V_{A}(0) since the start time, per unit mutational input.\n " \
                             "Only interesting as a 'both' stat.\n This scaling should makes results from " \
                             "simulations different shift size, and mutation rate,\n" \
                             " but with the same effect size distribution of new muts, approximately coincide."
            final_moment_explanation_dict['2a_frac_fixed_scaled_per_mut_input'] = explanation

        return final_moment_explanation_dict

    def final_stats(self):
        """
        Returns a dictionary of dictionaries of the final stats. The final stats are statistics collected after
        every segregating variant has been fixed or lost.
        Keys are tuples (S, XI,my_time)
        """

        final_dict = defaultdict(lambda : defaultdict(lambda :defaultdict(dict)))

        for tup in self.tuples_pos:

            # This is just 1 unless time <0
            # If time less than zero, then avering over |time| gens after the shift
            num_gens_of_muts = self.number_gens_of_mut_input(tup)

            S = self.scaled_size(tup)
            XI = self.x_i(tup)
            time = self.time(tup)

            homozyg_size = self.pheno_size(tup)

            if time == 0:
                if XI > 0: # we had a specific initial frequency
                    half_integral_of_tau = self._half_folded_tau(x=XI, S=S)
                else: # we integrated over initial frequencies
                    half_integral_of_tau = float(self._half_tau_normaliser(S))
            else: # Initial frequency was 1 (xi=1/(2N))
                half_integral_of_tau = 1.0

            normalizer_multiply = 1.0/float(half_integral_of_tau)

            for st in self._starts:

                nmuts = float(self.NUM_MUTANTS[tup][st])
                nfixed = float(self.num_fixed[tup][st])
                nextinct = float(self.num_extinct[tup][st])

                nfixed_weights = float(self.sum_fixed_weights[tup][st])

                final_dict[tup]['U0_NUM_MUTANTS'][st]['sum'] = nmuts
                final_dict[tup]['U0_NUM_MUTANTS'][st]['sum_squares']= nmuts**2
                final_dict[tup]['U0_NUM_MUTANTS'][st]['num'] =nmuts

                # should be zero
                final_dict[tup]['U0_NUM_SEGREGATING'][st]['sum'] = self.num_segregating[tup][st]
                final_dict[tup]['U0_NUM_SEGREGATING'][st]['sum_squares'] = self.num_segregating[tup][st]**2
                final_dict[tup]['U0_NUM_SEGREGATING'][st]['num'] = self.num_segregating[tup][st]

                final_dict[tup]['U0_frac_fixed_per_seg_var'][st]['sum'] = float(nfixed_weights) * normalizer_multiply
                final_dict[tup]['U0_frac_fixed_per_seg_var'][st]['sum_squares'] = \
                    float(self.sum_squares_fixed_weights[tup][st]) * normalizer_multiply ** 2
                final_dict[tup]['U0_frac_fixed_per_seg_var'][st]['num'] = float(nmuts)

                final_dict[tup]['U1_2a_frac_fixed_per_mut_input'][st]['sum'] = \
                    float(nfixed_weights)*homozyg_size*num_gens_of_muts
                final_dict[tup]['U1_2a_frac_fixed_per_mut_input'][st]['sum_squares'] =\
                    float(self.sum_squares_fixed_weights[tup][st]) *homozyg_size**2*num_gens_of_muts**2
                final_dict[tup]['U1_2a_frac_fixed_per_mut_input'][st]['num'] = float(nmuts)

                if self._SCALE_CONTRIBUTION_PER_UNIT_MUT_INPUT:
                    final_dict[tup]['U1_2a_frac_fixed_scaled_per_mut_input'][st]['sum'] = \
                        self._SPECIAL_SCALING_FACTOR*final_dict[tup]['U1_2a_frac_fixed_per_mut_input'][st]['sum']
                    final_dict[tup]['U1_2a_frac_fixed_scaled_per_mut_input'][st]['sum_squares'] = \
                        self._SPECIAL_SCALING_FACTOR**2*final_dict[tup]['U1_2a_frac_fixed_per_mut_input'][st]['sum_squares']
                    final_dict[tup]['U1_2a_frac_fixed_scaled_per_mut_input'][st]['num'] = float(nmuts)

        return default_to_regular(final_dict)


    def segregating(self):
        """
        This method spews out the set of segregating mutations.

        Returns:
            (dict) dictionary with values sets of segregating mutations
        """
        return self._segregating

    def are_there_segregating_mutations(self):
        """
        Returns True if there are any mutations still segregating. Otherwise returns false.
        """
        for tup in self.tuples_pos:
            if self.num_segregating[tup]['pos'] > 0:
                return True

        return False


    def number_segregating_mutations(self):
        """
        Returns the number of mutations still segregating.
        """
        num = 0
        for tup in self.tuples_pos:
            num+= self.num_segregating[tup]
        return num


    def update_moms_to_record_delta_moms(self):
        """
        Updates moments in _old_stats_d so that the delta of those moments can be recorded
        """
        self._update_all_moments()

        # [tuple_pos][moment]['pos'/'neg'/'both']['sum'/'sum_squares'/'num']

        for mom in self._moments:
            if mom in self._delta_moments:
                for vali in self._values:
                    for st in self._starts:
                        for tup in self.tuples_pos:
                            self._old_moments_new_d[tup][mom][st][vali] = self._moments_new_d[tup][mom][st][vali]

    ######## DELETE FOR ONLINE VERSION########
    def delta_stats(self):
        """
        Returns a diction of the delta_stats since the previous generation.

        Notes:
            Make sure you update for delta stats FIRST. Then next_gen, then this
        """
        delta_stats = defaultdict(lambda:defaultdict(lambda: defaultdict(dict)))

        for tup in self.tuples_pos:
            for stat in self._delta_moments:
                unit, body = stat[:3], stat[3:]
                dstat = unit+'delta_' + body
                for st in self._starts:
                    delta_stats[tup][dstat][st]['sum'] = 0.0
                    one = 0
                    two = 0
                    try:
                        one=   self._moments_new_d[tup][stat][st]['sum']
                    except KeyError:
                        print('tup ', tup)
                        print('start ', st)
                        print('stat ', stat)
                        print('one prob', self._moments_new_d)
                    try:
                        two = self._old_moments_new_d[tup][stat][st]['sum']
                    except KeyError:
                        print('tup ', tup)
                        print('start ', st)
                        print('stat ', stat)
                        print('two prob', self._moments_new_d)
                    try:
                        delta_stats[tup][dstat][st]['sum'] = one -two
                    except KeyError:
                        print('tup ', tup)
                        print('start ', st)
                        print('stat ', stat)
                        print('delta prob',delta_stats)

                    delta_stats[tup][dstat][st]['sum_squares'] = self._moments_new_d[tup][stat][st]['sum_squares'] +  self._old_moments_new_d[tup][stat][st]['sum_squares']
                    delta_stats[tup][dstat][st]['num'] = self._moments_new_d[tup][stat][st]['num']

        return default_to_regular(delta_stats)

    def _half_folded_tau(self, x, S):
        """Returns half the folded sojourn time for a given
        scaled selection coefficient (folded so that it is a
        function of the minor allele frequency).
        This is the density of pairs of steady-state"""
        if x < self._DENOMINATOR:
            facti = 2 * self.N
        else:
            facti = 1.0/x
        thetau =2 * facti * np.exp(-np.abs(S)* x * (1 - x))/(1 - x)
        return 0.5*thetau

    def _half_tau_normaliser(self, S):
        """
        Returns the normalising constant that transforms the sojourn time to the MAF distribution.
        """
        if S < 0:
            S = -S

        if S not in self.half_tau_norm:
            half_tau_norm = quad(self._half_folded_tau, 0, 0.5, args=(S), points=[self._DENOMINATOR], limit=100)[0]
            self.half_tau_norm[S] = half_tau_norm

        return self.half_tau_norm[S]

    def _folded_tau_normalised(self,x,S):
        """Returns the frequency_pos distribution (of minor alleles) for a given S"""
        return self._half_folded_tau(x, S) / self._half_tau_normaliser(S)

    def _get_muation_weight_xS_uniform_prior(self,x,S):
        """Returns the weight of a mutation under an importance sampling regime
        when a uniform prior on allele frequencies is used"""
        weight = self._half_folded_tau(x=x, S=S)
        return weight

    def _half_folded_tau_over_folded_var_freq_dist(self, x, S):
        if S < 0:
            S = -S
        rs = np.sqrt(S)
        var_normaliser =  4.0*rs*dawsn(rs/2.0)-S*self._SCAL
        if x <= self._DENOMINATOR:
            numerati = float(self.N)/(S*(1-x))
        else:
            numerati = 1.0 / (2.0 * S * x * (1 - x))
        foldtau_over_fold_var_freq_dist =numerati*float(var_normaliser)
        return 0.5*foldtau_over_fold_var_freq_dist

    def _get_random_freqs_uniform_prior(self, nfreqs):
        nrand_freqs = len(self.my_random_frequencies)
        if nfreqs <= nrand_freqs:
            return self.my_random_frequencies[:nfreqs]
        else:
            diff = nfreqs - nrand_freqs
            morefreqs  = self._generate_random_frequencies_uniform_prior_1(num=diff)
            morefreqs = [xii for xii in morefreqs]
            self.my_random_frequencies +=morefreqs
            return self.my_random_frequencies

    def _generate_random_frequencies_uniform_prior_1(self, num):
        high = self.N
        morefreqs = np.random.randint(high, size=num)
        return morefreqs


    def _generate_random_frequencies_var_distr_prior(self,num,S):
        mydist = VarDistrMinor(S=S,N=self.N)
        the_frss = mydist.rvs(size=num)
        the_frs = [int(round(2*self.N*ff)) for ff in the_frss]
        return the_frs

    def _get_mutation(self,S,frequency,weight):
        """
        Returns a new mutation with scaled effective size S
        """

        if S <= 0:
            S =-S
        # the scaled effect size has gamma distribution
        scaled_size = S
        pheno_size_hom = math.sqrt(self._MU_SCALING_COEFF * S)

        # we add the mutation to the segregating list
        mu = MutationPosNeg(scaled_size=scaled_size,pheno_size_homozyg=pheno_size_hom,N=self.N)
        mu.update_freq(frequency_pos=frequency,frequency_neg=frequency,update=False)
        mu._frequency_class_pos.prev_freq = mu._frequency_class_pos.frequency
        mu._frequency_class_neg.prev_freq = mu._frequency_class_neg.frequency
        mu.initial_freq = frequency

        #set the mutations weight
        mu.weight = weight
        return mu

    def _remove_extinct_fixed_from_seg(self):
        """
        Calculatess which mutations are extinct or fixed and removes them
        from the segregating list. Adds effect to _stats_d['mean_fixed']
        """
        # iterate over segregating mutation lists and search for extinct or fixed mutations

        self.last_removal_of_extinct_fixed_from_seg = 0

        for tup in self.tuples_pos:
            #S,  XI, time = tup

            #if both the pos mut and the neg mut are gone we remove it from the list
            extinct_or_fixed_aligned_and_nonaligned = [mu for mu in self._segregating[tup] if not mu.either_segregating()]

            both_extinct = [mu for mu in  extinct_or_fixed_aligned_and_nonaligned if mu.is_extinct(pos=True) and mu.is_extinct(pos=False)]
            both_fixed = [mu for mu in  extinct_or_fixed_aligned_and_nonaligned if mu.is_fixed(pos=True) and mu.is_fixed(pos=False)]
            pos_fixed_neg_extinct = [mu for mu in  extinct_or_fixed_aligned_and_nonaligned if mu.is_fixed(pos=True) and mu.is_extinct(pos=False)]
            neg_fixed_pos_extinct = [mu for mu in extinct_or_fixed_aligned_and_nonaligned if mu.is_extinct(pos=True) and mu.is_fixed(pos=False)]

            # for fixed mutations
            for mu in both_fixed:
                self._fix_mut_in_tuple(tup, mu, pos=True)
                self._fix_mut_in_tuple(tup, mu, pos=False)
                self._segregating[tup].remove(mu)

            # for extinct mutations
            for mu in both_extinct:
                self._extinct_mut_in_tuple(tup, mu, pos=True)
                self._extinct_mut_in_tuple(tup, mu, pos=False)
                self._segregating[tup].remove(mu)

            for mu in pos_fixed_neg_extinct:
                self._fix_mut_in_tuple(tup, mu, pos=True)
                self._extinct_mut_in_tuple(tup, mu, pos=False)
                self._fix_and_extinct_mut_in_tuple(tup, mu, pos=True)
                self._segregating[tup].remove(mu)

            for mu in neg_fixed_pos_extinct:
                self._fix_mut_in_tuple(tup, mu, pos=False)
                self._extinct_mut_in_tuple(tup, mu, pos=True)
                self._fix_and_extinct_mut_in_tuple(tup, mu, pos=False)

                self._segregating[tup].remove(mu)

    def _fix_and_extinct_mut_in_tuple(self, tup, mut, pos=None):
        square_weight = mut.weight ** 2
        weight = mut.weight

        if pos is None:
            pos = True  # fixed in the pos, extinct in the neg
        st = self._starts[-1] # st = 'both'
        self.num_fixed[tup][st] += 1 # This is the number we are averaging over
        self.num_extinct[tup][st] += 1  # This is the number we are averaging over
        if pos:
            self.sum_fixed_weights[tup][st] += weight
            self.sum_extinct_weights[tup][st] -= weight
        else:
            self.sum_fixed_weights[tup][st] -= weight
            self.sum_extinct_weights[tup][st] += weight

        self.sum_squares_fixed_weights[tup][st] += square_weight
        self.sum_squares_fixed_weights_dx[tup][st] += square_weight

        self.sum_squares_extinct_weights[tup][st] += square_weight
        self.sum_squares_extinct_weights_dx[tup][st] += square_weight


    def _fix_mut_in_tuple(self, tup, mut, pos=None):

        tup_time = self.time(tup)

        square_weight = mut.weight ** 2
        weight = mut.weight
        if pos is None:
            pos = True
        if pos:
            st = 'pos'
            if tup_time == 0: # segregating mutations
                dx = mut.x_change(pos=True)
            else: # new mutations
                dx = 1.0
        else:
            st = 'neg'
            if tup_time == 0: # segregating mutations
                dx = mut.x_change(pos=False)
            else: # new mutations
                dx = 1.0

        self.num_fixed[tup][st] += 1
        self.sum_fixed_weights[tup][st] += weight
        self.sum_squares_fixed_weights[tup][st] += square_weight
        self.sum_squares_fixed_weights_dx[tup][st] += square_weight * dx ** 2
        self.num_segregating[tup][st] -= 1


    def _extinct_mut_in_tuple(self, tup, mut, pos=None):
        square_weight = mut.weight ** 2
        weight = mut.weight

        tup_time = self.time(tup)

        if pos is None:
            pos = True
        if pos:
            st = 'pos'
            if tup_time == 0: # segregating mutations
                dx = mut.x_change(pos=True)
            else: # new mutations
                dx = 0.0
        else:
            st = 'neg'
            if tup_time == 0: # segregating mutations
                dx = mut.x_change(pos=False)
            else: # new mutations
                dx = 0.0

        self.num_extinct[tup][st] += 1
        self.sum_extinct_weights[tup][st] += weight
        self.sum_squares_extinct_weights[tup][st] += square_weight
        self.sum_squares_extinct_weights_dx[tup][st] += square_weight * dx ** 2
        self.num_segregating[tup][st] -= 1

    def _update_all_moments(self):
        """
        Updates all the moments.
        """
        #Remove extinct and fixed mutations from the segregating list
        #if we have not done so this generation
        if self.last_removal_of_extinct_fixed_from_seg > 0:
            self._remove_extinct_fixed_from_seg()

        for tup in self.tuples_pos:

            S = self.scaled_size(tup)
            xi = self.x_i(tup)
            time = self.time(tup)

            number_gens_of_mutations = self.number_gens_of_mut_input(tup)

            if time == 0:
                if xi > 0:
                    half_integral_of_tau = float(self._half_folded_tau(x=xi, S=S)) #over 2?
                else:
                    half_integral_of_tau = float(self._half_tau_normaliser(S))
            else:
                half_integral_of_tau = 1.0

            normalizer_multiply = 1.0 / float(half_integral_of_tau)

            homozyg_size_pos = self.pheno_size(tup)

            last_start = self._starts[-1]

            for st in self._starts:

                nmuts = float(self.NUM_MUTANTS[tup][st])

                if st == last_start:
                    last = True
                else:
                    last = False

                if st == 'neg':
                    homozyg_size = -homozyg_size_pos
                    pos = False
                else:
                    pos = True
                    homozyg_size = homozyg_size_pos

                nmuts_seg = 0.0
                sum_init_x = self.sum_weighted_initial_frequency[tup][st]

                sum_seg_weighted_freq = 0.0
                sum_square_seg_weighted_freq = 0.0
                sum_square_seg_weighted_freq_change = 0.0
                this_var_sum = 0.0
                this_var_sum_squares = 0.0
                this_third_mom_sum = 0.0
                this_third_mom_sum_squares = 0.0
                weighted_nmuts = 0.0

                fixed_state_total_contributions = 0.0
                fixed_state_total_contributions_square = 0.0

                for mut in self._segregating[tup]:

                    dx = mut.x_change(pos=pos,both=last)
                    if last:
                        xm = dx
                    else:
                        xm = mut.x(pos=pos)

                    nmuts_seg += 1
                    weight = mut.weight
                    weighted_nmuts +=weight

                    sum_seg_weighted_freq += xm*weight
                    sum_square_seg_weighted_freq += xm**2*weight**2
                    sum_square_seg_weighted_freq_change += dx**2*weight**2

                sum_fixed_weights = self.sum_fixed_weights[tup][st]
                sum_squares_fixed_weights = self.sum_squares_fixed_weights[tup][st]
                sum_squares_fixed_weights_dx = self.sum_squares_fixed_weights_dx[tup][st]
                sum_squares_extinct_weights_dx = self.sum_squares_extinct_weights_dx[tup][st]

                sum_all_weighted_freqs = sum_seg_weighted_freq + sum_fixed_weights
                sum_all_squared_weighted_freqs = sum_square_seg_weighted_freq + sum_squares_fixed_weights
                sum_all_squared_weighted_freqs_dx = sum_square_seg_weighted_freq_change + sum_squares_fixed_weights_dx+sum_squares_extinct_weights_dx

                #I *think* I can delete weighted_nmuts
                #weighted_nmuts += (sum_fixed_weights+sum_extinct_weights)

                self._moments_new_d[tup]['U0_x_per_seg_var'][st]['sum'] = \
                    sum_all_weighted_freqs * normalizer_multiply
                self._moments_new_d[tup]['U0_x_per_seg_var'][st]['sum_squares'] = \
                    sum_all_squared_weighted_freqs * normalizer_multiply ** 2
                self._moments_new_d[tup]['U0_x_per_seg_var'][st]['num'] = nmuts

                self._moments_new_d[tup]['U0_dx_per_seg_var'][st]['sum'] = \
                    (sum_all_weighted_freqs - sum_init_x) * normalizer_multiply
                self._moments_new_d[tup]['U0_dx_per_seg_var'][st]['num'] = nmuts
                self._moments_new_d[tup]['U0_dx_per_seg_var'][st]['sum_squares'] = \
                    sum_all_squared_weighted_freqs_dx * normalizer_multiply ** 2



                self._moments_new_d[tup]['U1_d2ax_per_mut_input'][st]['sum'] = \
                    (sum_all_weighted_freqs - sum_init_x) * homozyg_size * number_gens_of_mutations
                self._moments_new_d[tup]['U1_d2ax_per_mut_input'][st]['num'] = nmuts
                self._moments_new_d[tup]['U1_d2ax_per_mut_input'][st]['sum_squares'] = \
                    sum_all_squared_weighted_freqs_dx * homozyg_size ** 2 * number_gens_of_mutations ** 2

                if self._SCALE_CONTRIBUTION_PER_UNIT_MUT_INPUT:
                    self._moments_new_d[tup]['U1_d2ax_scaled_per_mut_input'][st]['sum'] = self._SPECIAL_SCALING_FACTOR*\
                        self._moments_new_d[tup]['U1_d2ax_per_mut_input'][st]['sum']
                    self._moments_new_d[tup]['U1_d2ax_scaled_per_mut_input'][st]['num'] = nmuts
                    self._moments_new_d[tup]['U1_d2ax_scaled_per_mut_input'][st]['sum_squares'] = \
                        self._SPECIAL_SCALING_FACTOR**2*\
                        self._moments_new_d[tup]['U1_d2ax_per_mut_input'][st]['sum_squares']


    def _power(self, r_diff, varies, p, c, a): 
        """
        a is the phenotypic effect of the mutation. c is 0,0.5 or 1. r_diff is the
        signed difference ( optimum phenotype -mean phenotype)
        """
        return -(r_diff - a*(c-p))**2*float(varies)

    def _wright_fisher(self,distance = None):
        """
        Updates the list of segregating mutations according to a Wright-Fisher process
        """
        if distance is None:
            distance = 0

        varies = 1.0/float(2 *  self.Vs)

        for tup in self.tuples_pos:
            # update the frequencies stochastically
            for mut in self._segregating[tup]:
                for pos in [True, False]:
                    if mut.is_segregating(pos=pos):
                        p = mut.x(pos=pos)
                        q = 1.0-p
                        a_magnitude = mut.pheno_size_homozyg()

                        a_trait_interest = mut.sign(pos) * a_magnitude

                        c_00, c_01, c_11 = 0, 0.5, 1
                        power_00 = self._power(distance, varies, p, c_00, a_trait_interest)
                        power_01 = self._power(distance, varies, p, c_01, a_trait_interest)
                        power_11 = self._power(distance, varies, p, c_11, a_trait_interest)

                        w_00 = math.exp(power_00)
                        w_01 = math.exp(power_01)
                        w_11= math.exp(power_11)
                        meanfit = p**2*w_11 +q**2*w_00 +2*p*q*w_01
                        weight = (p**2*w_11 + p*q*w_01)/meanfit
                        if weight >1:
                            print('weight:', weight, ' p:', p,' distance:' , distance,' phenoSize:', a)
                            raise Exception("The weight is bigger than one.")

                        newfreq = np.random.binomial(2*self.N, weight)

                        if pos:
                            mut.update_freq(frequency_pos=newfreq)
                        else:
                            mut.update_freq(frequency_neg=newfreq)