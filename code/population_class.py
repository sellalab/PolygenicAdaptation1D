"""This module has the container classes for the information about the population at any time
In here we have: Mutation, MutationalProcess, PopulationBasic, PopulationExact, PopulationWF
PopulationWF is the approximation via a Wright-Fisher process
PopulationExact simulates what happens to members of the population every generation
"""
from collections import defaultdict, namedtuple
import math
from mutation_class import Mutation
import random
import numpy as np
import abc
from scipy.stats import gamma
from scipy.special import dawsn, erf
from scipy.integrate import quad
from itertools import tee

def cubic_root(x):
    return math.copysign(math.pow(abs(x), 1.0/3.0), x)

def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ...
allows you to iterarte over all possible pairs in a list or set
EG. for v, w in pairwise([5,4,6])
    print v, w"""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

# parameters for the mutational process
# mu: haploid mutation rate per-generation
# shape, scale: gamma-distribution parameters, describing the effect size distribution
MutationalProcess = namedtuple('MutationalProcess', 'mu shape scale')

class _PopulationBasic(object):
    """
    Class to store and evolve the population

    Parameters:
        N: (int) Population size
        Vs: (float) Fitness parameter
        mu: (namedtuple) MutationalProcess (mu, shape, scale)

    Attributes:
        N: (int) Population size
        Vs: (float) Fitness parameter
        _mu: (namedtuple) MutationalProcess (U, shape, scale)

        _squared_effect_size_dist: (distribution) gamma dist of incoming squared effects (units little del squared)

        _dist_ess: (float) The distance of the mean phenotype from the
                    optimum phenotype . _ess = Essential to record
        _mean_fixed_ess: (flaot) The mean fixed phenotype.
                    The mean fixed phenotype is the phenotype from fixed mutations. _ess = Essential to record
        _mean_seg_ess: (float) The mean phenotype from segregating derived
                    mutations . _ess = Essential to record
        _fixed_state_minus_mean_fixed_ess: (float) The fixed state minus mean fixed
                    phenotype. The fixed state is what the mean phenotype would be if every
                    minor allele were to go extinct. The mean fixed phenotype is the phenotype from fixed mutations
        _var_ess: (flaat) The phenotypic variance . _ess = Essential to record

        _statistics: (list) Some basic quantities - e.g. mean, var, dist, mean_seg, mean_fixed
                    - of the phenotype distribution, that we may want to record
        _stats_essential_stats: (list) The statistics that can be updated using only the essential statistics
        _delta_statistics: (list) Stats that we record the per generation change in.
        _stat_set: (set) Set of all the statistics recorded

        _stats_d: (dict) Storing the current value of all the stats in _statistics
        _old_stats_d: (dict) Storing the value in the previous generation of all the stats in _delta_statistics

        _segregating: (set) A set containing all the mutations currently segregating in the population
        _frozen: (set) Set of mutations segregating at freeze time

        _DENOMINATOR: (float) 1/(2*N)
        _FITNESSCOEF: (float) 1/(2*Vs)
        _MU_SCALING_COEFF: (float) Mutation scaling parameter (2Vs)/N
        _FITNESS_OPTIMUM: (float) Value of the fitness optimum

        freq_percentiles: (list) A list of the frequency_pos bin percentiles (according to equilibrium distrib)
        freq_bins: (list) A list of the frequencies corresponding to the frequency_pos bin percentiles
        effect_percentiles: (list) A list of the effect size bin percentiles (according to dist of incoming muts)
        effect_bins: (list) A list of the effectsizes corresponding to the effectsize bin percentiles
        f_bin_number: (int) Number of frequency_pos bins
        eff_bin_number: (int) Number of effectsize bins
"""
    metaclass = abc.ABCMeta

    def __init__(self, N, Vs, mu):
        self.N = N
        # if Vs <0, set it to be 2N
        if Vs < 0:
            self.Vs = float(2 * self.N)
        else:
            self.Vs = Vs
        # mutational process parameters
        self._mu = mu
        self._squared_effect_size_dist = gamma(self._mu.shape, loc=0., scale=self._mu.scale)

        # We will always store the moments
        self._statistics = ['U3_mu3', 'U1_dist_guess']
        self._statistics += ['U0_num_seg', 'U0_skewness']
        # Relavent after shift
        self._statistics += ['U0_d2ax_frozen_over_shift','U1_dist_sum_over_2N_scaled']


        self._stats_essential_stats = ['U2_var', 'U1_dist']
        self._stats_essential_stats += ['U0_fixed_state_minus_mean_fixed_over_shift', 'U0_mean_minus_fixed_state_over_shift']
        self._stats_essential_stats += ['U0_mean_minus_mean_fixed_over_shift', 'U2_dist_square']
        self._stats_essential_stats += ['U0_opt_minus_fixed_state_over_shift', 'U0_opt_minus_mean_fixed_over_shift']

        # Moments that are required to update the segregating variants
        self._dist_ess = 0
        self._mean_fixed_ess = 0
        self._mean_seg_ess = 0
        self._fixed_state_minus_mean_fixed_ess = 0
        self._var_ess = 0

        stat_list = self._statistics + self._stats_essential_stats


        #We record the per generation change in these stats
        self._delta_statistics = ['U1_dist', 'U2_var', 'U3_mu3']
        for mom in self._delta_statistics:
            if mom not in stat_list:
                self._delta_statistics.remove(mom)

        self._stat_set = set(stat_list)

        # Dictionaries to store the statistics
        self._stats_d = {}
        self._old_stats_d = {}

        for m in self._stat_set:
            self._stats_d[m] = 0.
        for om in self._delta_statistics:
            self._old_stats_d[om] = 0.

        # set of segregating mutations
        self._segregating = set()

        self._set_up_useful_constants()

        self._set_up_histograms(nF = 11, nE =13,nA=6)

        #when did we last update the moments - ensures we don't do it twice in a generation
        self._last_update_moments = 0

        self._frozen = set() #set of mutants segregating at freeze_time
        self._number_generations_after_freeze = -1 # how many generations have passed since mutants were frozen
        self._have_we_frozen = False #Tells us if we are after the freeze or not
        self._frozen_nm = set()  # set of currently segregating mutants that arose after freeze_time
        self._have_we_frozen_new_mutations = False #Tells us if we are after the freeze of new mutations or not
        self._currently_freezing_new_mutations = False #Tells us if we are currently freezing of new mutations or not
        self._sums_update_time = -1#Lets us know if we should update the sums
        self._var_sum = 0.0 #Tells us the integral of variance after the freeze
        self._dist_sum = 0.0 #Tells us the integral of the distance after the freeze
        self._dist_guess_sum =0.0 #Tells us the integral of the distance guess after the freeze

    def _moment_explations_extra_histos(self):
        moment_explation_dict = dict()

        # This bstat is only recorded when there are histo stats
        explanation = "The fraction contribution to the change in mean phenotype after the shift, coming"
        moment_explation_dict['d2ax_frozen_over_shift'] = explanation

        # These are histo stats
        explanation = "The number of alleles that were segregating at the time of the shift (freeze_time), that" \
                      "are still segregating, per unit mutational input"
        moment_explation_dict["frozen_numseg_per_unit_mut_input"] = explanation

        explanation = "The contribution to the change in mean phenotype from (frozen) alleles segregating at the \n" \
                      "time of the shift in optimum,  divided by the shift size, per unit mutational input"
        moment_explation_dict["frozen_d2ax_over_shift"] = explanation
        explanation = "The contribution to the change in mean phenotype from (frozen) mutations after the \n" \
                      " shift in optimum, divided by the shift size, per unit mutational input"
        moment_explation_dict["frozen_nm_d2ax_over_shift"] = explanation
        explanation = "The contribution to the change in mean phenotype from (frozen) alleles segregating at the \n" \
                      "time of the shift in optimum, divided by the unitless quantity" \
                     " (shift*delta_unit)/V_{A}, per unit mutational input.\n " \
                     " This scaling should makes results from " \
                     "simulations different shift size, and mutation rate,\n" \
                     " but with the same effect size distribution of new muts, approximately coincide."
        moment_explation_dict["frozen_d2ax_scaled_per_mut_input"] = explanation
        explanation = "The contribution to the change in mean phenotype from (frozen) new mutations mutations after\n" \
                      "the shift in optimum, divided by the unitless quantity (shift*delta_unit)/V_{A}, " \
                      "per unit mutational input.\n " \
                     " This scaling should makes results from simulations different shift size, and mutation rate,\n" \
                     " but with the same effect size distribution of new muts, approximately coincide."
        moment_explation_dict["frozen_nm_d2ax_scaled_per_mut_input"] = explanation
        explanation = "The combined contribution to the change in mean phenotype from (frozen) alleles segregating \n" \
                      "at the time of the shift and (frozen) new mutations arising after the shift in optimum  \n" \
                      ", divided by the unitless quantity (shift*delta_unit)/V_{A},per unit mutational input.\n " \
                     " This scaling should makes results from simulations different shift size, and mutation rate,\n" \
                     " but with the same effect size distribution of new muts, approximately coincide."
        moment_explation_dict["frozen_nm_and_standing_d2ax_scaled_per_mut_input"] = explanation

        return moment_explation_dict

    def moment_explations_no_histos(self):
        moment_explation_dict = dict()
        moment_explation_dict['dist'] = "Distance between the trait mean and the fitness optimum (=optimum - mean)"
        explanation = "The integral (from the time of the shift)of the distance of the mean phenotype from the optimum\n " \
                      "divided by the unitless quantity (shift*delta_unit)/V_{A}(0) and also divided by 2N\n " \
                      "(For very Lande type squared effect distributions, it should approach 1 over long times)"
        moment_explation_dict['dist_sum_over_2N_scaled'] = explanation

        moment_explation_dict['mu3'] = "Third central moment of the phenotype distribution"
        moment_explation_dict['var'] = "Variance in the phenotype distribution"
        moment_explation_dict['disst_guess'] = "The quasi-static approximation for the dist: = mu3(t)/(2*var(t))"
        moment_explation_dict['skewness'] = "Skewness in the phenotype distribution"
        moment_explation_dict['numseg'] = "Number of segregating variants"

        explanation = "The fitness optimum minus the fixed background, all divided by the size of the shift"
        moment_explation_dict['opt_minus_mean_fixed_over_shift'] = explanation
        explanation = "The fitness optimum minus the consensus genotype, all divided by the size of the shift"
        moment_explation_dict['opt_minus_fixed_state_over_shift'] = explanation
        explanation = "The mean phenotype minus the fixed background, all divided by the size of the shift"
        moment_explation_dict['mean_minus_mean_fixed_over_shift'] = explanation
        explanation = "The mean phenotype minus the consensus genotype, all divided by the size of the shift"
        moment_explation_dict['mean_minus_fixed_state_over_shift'] = explanation
        explanation = "The consensus genotype minus the fixed background, all divided by the size of the shift"
        moment_explation_dict['fixed_state_minus_mean_fixed_over_shift'] = explanation

        moment_explation_dict['dist_square'] = "Distance between the trait mean and the fitness optimum, squared " \
                                        "(=[optimum - mean]^2)"

        return moment_explation_dict

    def moment_explations_with_histos(self):
        moment_explation_dict = self.moment_explations_no_histos()
        moment_explation_dict.update(self._moment_explations_extra_histos())
        return moment_explation_dict

    def _set_up_useful_constants(self):

        self._DELTA_UNIT = np.sqrt(float(self.Vs)) / np.sqrt(2 * self.N)
        self._DENOMINATOR = 1.0 / (2.0 * float(self.N)) # useful factor to multiply stuff with
        self._MUT_INPUT = float(2.0 * self.N * self._mu.mu) # Mutational input per generation
        self._ONE_OVER_MUT_INPUT = 1.0/self._MUT_INPUT
        self._FITNESS_OPTIMUM = 0.0 # initialize fitness optimum as zero
        self._FITNESSCOEF = 0.5 / float(self.Vs) # selection model paramter

        # mutation scaling parameter. Scaled selection coefficients are squared HETEROZYOGOTE effect sizes of alleles
        # in units of delta^2 =Vs/(2N) = 1. We use _MU_SCALING_COEF to do two things. 1) To convert these squared
        # HETEROZYOGOTE effect sizes to squared HOMOZYGOTE effect sizes, by multiplying by 4. And 2) to change the
        # squared effect sizes into the units in which Vs = self.Vs. For example, if  self.Vs =1, then _MU_SCALING_COEF
        # is used to convert to squared homozygote effect sizes in units of Vs = 1. But if Vs = 2N, then it is used to
        # convert to squared homozygote effect sizes to units of Vs = 2N  (i.e. delta^2 =Vs/(2N) = 1). So in the second
        # example, when Vs =2N, _MU_SCALING_COEFF =4, and it is just used to change squared HETEROZYOGOTE
        # effect sizes to squared homozygote effect sizes, without changing the units
        self._MU_SCALING_COEFF = 2.0 * float(self.Vs) / float(self.N) # mutation scaling parameter

        self._SPECIAL_SCALING_FACTOR = 1.0 #when the opt shifts this changes to 1/(shift/sigma_0)*(DELTA_UNIT/sigma_0)
        self._RATIO_V_TO_F_MINUS_ONE_SCALING_FACTOR = self._one_over_ratio_V_to_F_minus_one()
        self._SHIFT_SIZE_SCALING_FACTOR = self._one_over_sigma_0_units_rootVs()
        self._RATIO_V_TO_F_SCALING_FACTOR = self._one_over_ratio_V_to_F()
        self._MU3_NORMALIZER = 1.0

    def var(self):
        return self._var_ess

    @abc.abstractmethod
    def next_gen(self):
        """
        Progresses to the next generation.
        """
        return

    @abc.abstractmethod
    def basic_stats(self):
        """
        Returns a dictiionary of the basic stats
        """
        return self._basic_stats()

    @abc.abstractmethod
    def histo_stats(self, frozen = None):
        """
        Returns a dictionary of the current histogram data.

        If frozen = True also has histogram data for the frozen mutants, and frozen new mutatnts.
        """
        if frozen is None:
            frozen = False

        self._update_segregating_list()

        hist_fbins, hist_efs_bins  = dict(), dict()
            # collect statistics on segregating mutations

        if frozen:
            if self._have_we_frozen:
                hist_fbins_frozen, hist_efs_bins_frozen = self._histo_frozen()
                hist_fbins.update(hist_fbins_frozen)
                hist_efs_bins.update(hist_efs_bins_frozen)
            if self._have_we_frozen_new_mutations:
                hist_efs_bins_frozen  = self._histo_frozen_nm()
                hist_efs_bins.update(hist_efs_bins_frozen)

            # reacord the combined contribution from new mutations after the shift and standing variation
            if self._have_we_frozen and self._have_we_frozen_new_mutations:
                stat1 = 'U1_frozen_d2ax_scaled_per_mut_input'
                stat2 = 'U1_frozen_nm_2ax_scaled_per_mut_input'
                combined_stat_name = 'U1_frozen_nm_and_standing_d2ax_scaled_per_mut_input'
                hist_efs_bins = self._add_combination_two_hist_stats(stat1=stat1,stat2=stat2,newstatname=combined_stat_name,
                                                             histo_stat_dict=hist_efs_bins)

        for key in list(hist_fbins.keys()):
            hist_fbins[key+'_fbins_H'] = hist_fbins.pop(key)
        for key in list(hist_efs_bins.keys()):
            hist_efs_bins[key+'_efs_bins_H'] = hist_efs_bins.pop(key)

        hist = dict()
        hist.update(hist_efs_bins)
        hist.update(hist_fbins)

        return hist

    def _add_combination_two_hist_stats(self,stat1,stat2,newstatname,histo_stat_dict):
        if stat1 in histo_stat_dict and stat2 in histo_stat_dict:
            num_bins1 = len(histo_stat_dict[stat1])
            num_bins2 = len(histo_stat_dict[stat2])
            if num_bins1 == num_bins2:
                histo_stat_dict[newstatname] = [0 for _ in range(num_bins1)]
                for i in range(num_bins1):
                    val1 = histo_stat_dict[stat1][i]
                    val2 = histo_stat_dict[stat2][i]
                    histo_stat_dict[newstatname][i] = val1 + val2
        return histo_stat_dict

    def segregating(self):
        """
        This method updates(if needed) and spews out the set of segregating mutations.

        Returns:
            (set) Set of segregating mutations
        """
        self._update_segregating_list()
        return self._segregating

    def freeze(self):
        """
        Records -- in a set called _frozen --- which mutations are segregating at the
        instance of use so that the future of those particular mutations can be tracked

        Can use more than once. But each my_time the previous list is replaced
        """

        self._update_segregating_list()
        self._freeze()
        self._have_we_frozen = True
        self._number_generations_after_freeze = 0

    def freeze_new_mutations(self,time_length=None):
        """
        Records -- in a set called _frozen --- which mutations are segregating at the
        instance of use so that the future of those particular mutations can be tracked

        Can use more than once. But each time the previous list is replaced
        """

        if not self._number_generations_after_freeze >= 0:
            return

        self._have_we_frozen_new_mutations = True
        self._currently_freezing_new_mutations = True

        self._number_generations_after_freeze_new_mutations = self._number_generations_after_freeze
        if time_length is None:
            self._time_length_freeze_new_muations = 10*self.N
        else:
            self._time_length_freeze_new_muations = time_length
        self._number_generations_after_freeze_end_new_mutations = self._number_generations_after_freeze_new_mutations+self._time_length_freeze_new_muations

        self._freeze_nm()

    def frozen(self):
        """
        Updates the segregating mutations (if necessary) and returns the current list of frozen mutations.

        Notes:
            To be used only after freeze has been used at least once.
        """
        self._update_segregating_list()

        return self._frozen

    def shift_optimum(self, delta):
        """Shifts the fitness optimum by delta"""
        self._number_generations_after_freeze = 0
        self._have_we_frozen = True
        self._FITNESS_OPTIMUM += delta
        the_quantity = 1.0/float(2*self.N)*float(delta)/self._DELTA_UNIT*1.0/float(self._var_0_units_Vs())
        self._SPECIAL_SCALING_FACTOR = 1.0/the_quantity
        self._SHIFT_SIZE_SCALING_FACTOR = 1.0/float(delta)
        self._update_essential_moments()
        self._update_moments_from_essential_moments()

    def histogram_bins(self):
        hist_bins = dict()
        hist_bins['fbins'] = self.freq_bins_minor_pos
        hist_bins['frac_muts_in_fbins'] = self.freq_fraction_muts_in_bin_pos
        hist_bins['one_over_mutinput_in_fbins'] = self.freq_one_over_mutational_input_in_bin_pos
        hist_bins['efs_bins'] = self.effect_bins_pos
        hist_bins['frac_muts_in_efs_bins']= self.effect_fraction_muts_in_bin_pos
        hist_bins['one_over_mutinput_in_efs_bins'] = self.effect_one_over_mutational_input_in_bin_pos
        return hist_bins

    def update_moms_to_record_delta_moms(self):
        """
        Updates moments in _old_stats_d so that the delta of those moments can be recorded
        """

        if self._delta_statistics:
            self._update_all_moments()
            for mom in self._delta_statistics:
                self._old_stats_d[mom] = self._stats_d[mom]


    def delta_stats(self):
        """
        Returns a diction of the delta_stats since the previous generation.

        Notes:
            Make sure you update for delta stats FIRST. Then next_gen, then this
        """
        delta_stats = {}
        for stat in self._delta_statistics:
            units, bod = stat[:3], stat[3:]
            delta_stats[units+'delta_' + bod] = self._stats_d[stat] - self._old_stats_d[stat]

        return delta_stats


    @abc.abstractmethod
    def _update_segregating_list(self):
        """
        This method updates the segregating mutations list. Done every generation.
        Removes extinct or fixed mutations. Adds effect to meanFixed
        """
        return

    @abc.abstractmethod
    def _remove_mut(self, mut):
        """
        Takes a mutation and removes it from the population
        """
        return

    def _get_mutation(self):
        """
        Returns a new mutation with effective size chosen according to a gamma distribution.

        The sign is negative with probability half.
        """

        # the magnitude of the scaled squared effect  size has gamma distribution
        scaled_size = self._get_random_selection_coefficient()
        # turn the scaled selection coefficient into the effect of a HOMOZYGOTE
        pheno_size_hom  = math.sqrt(self._MU_SCALING_COEFF * scaled_size)

        # and its sign is a uniform Bernouli variable
        derivedsign = 1
        if random.getrandbits(1):
            derivedsign = -1

        # we add the mutation to the segregating list
        mu = Mutation(scaled_size=scaled_size,pheno_size_homozyg=pheno_size_hom,derived_sign=derivedsign,N=self.N)

        if self._currently_freezing_new_mutations:
            self._freeze_new_mutation(mu)

        return mu


    def _get_random_selection_coefficient(self):
        if type(self._mu).__name__ == 'MutationalProcessU':
            return self._squared_effect_size_dist.get_random()
        elif type(self._mu).__name__ == 'MutationalProcess':
            return np.random.gamma(self._mu.shape, self._mu.scale)


    def _remove_extinct_fixed_from_seg(self):
        """
        Calculatess which mutations are extinct or fixed and removes them
        from the segregating list. Adds effect to _stats_d['mean_fixed']
        """
        # iterate over segregating mutation list and search for extinct or fixed mutations
        extinct = [mu for mu in self._segregating if mu.is_extinct()]
        fixed   = [mu for mu in self._segregating if mu.is_fixed()]

        # remove extinct and fixed mutations from the segregating list:
        for mu in (extinct + fixed):
            self._segregating.remove(mu)

        # for fixed mutations
        for mu in fixed:
            self._mean_fixed_ess += mu.fixing_effect() * mu.derived_sign()

            # remove mutation from all offspring:
            self._remove_mut(mu)

        if self._have_we_frozen:
            self._frozen_remove_fixed_extinct()

        if self._have_we_frozen_new_mutations:
            self._frozen_nm_remove_fixed_extinct()

    def _update_generations_after_freeze_related(self):
        if self._have_we_frozen:
            self._number_generations_after_freeze += 1

            if self._have_we_frozen_new_mutations:
                if self._number_generations_after_freeze > self._number_generations_after_freeze_end_new_mutations:
                    self._currently_freezing_new_mutations = False #stop freezing new mutation

    def _set_up_histograms(self, nF = 11,nE = 13,nA=6):# nE =6):
        """
        set up bins for histograms. Called on initialisation.
        """
        self._set_up_histograms_effects(nE=nE)
        self._set_up_histograms_freqs(nF=nF)

    def _set_up_histograms_freqs(self, nF = 11):# nE =6):
        """
        set up bins for histograms. Called on initialisation.
        """
        freq_bins_minor = np.linspace(0,0.5,num =nF)
        f_index_half = len(freq_bins_minor)-1
        freq_bins_minor = np.delete(freq_bins_minor,[0,f_index_half])

        self.freq_bins_minor_pos = [ff for ff in freq_bins_minor]
        self.freq_bins_minor = sorted([-x for x in self.freq_bins_minor_pos] + [0] + self.freq_bins_minor_pos)
        self.f_bin_number = len(self.freq_bins_minor) + 1

        self.freq_one_over_mutational_input_in_bin_pos = []
        self.freq_fraction_muts_in_bin_pos = []
        for _ in range(len(self.freq_bins_minor_pos)+1):
            frac_in_bin = 1.0/float(self.f_bin_number)
            # The frac in bin is just to make it equiv to effect bins
            self.freq_fraction_muts_in_bin_pos.append(frac_in_bin)
            self.freq_one_over_mutational_input_in_bin_pos.append(1.0/float(self._MUT_INPUT*frac_in_bin))

        mycopy = [one_over for one_over in self.freq_one_over_mutational_input_in_bin_pos]
        mycopy.reverse()

        self.freq_one_over_mutational_input_in_bin = mycopy + self.freq_one_over_mutational_input_in_bin_pos


    def _set_up_histograms_effects(self, nE = 13):# nE =6):
        """
        Set up bins for histograms. Called on initialisation.
        """
        s_list_exponents = np.linspace(-1, 2, nE)
        s_list = [10.0 ** exponenti for exponenti in s_list_exponents]

        # some bins may have too few mutations in them.
        THRESHOLD_FOR_NUM_MUTS = 10**(-3)
        remove_the_s = []
        percentile_prev = 0
        s_percentiles = dict()
        for i in range(len(s_list)):
            s = s_list[i]
            percentile = self._squared_effect_size_dist.cdf(s)
            s_percentiles[s] = percentile
            if s < 1:
                num_muts_per_generation_in_bin = (percentile-percentile_prev)*self._MUT_INPUT
                if num_muts_per_generation_in_bin < THRESHOLD_FOR_NUM_MUTS:
                    remove_the_s.append(s)
                else:
                    percentile_prev = percentile
        for ss in remove_the_s:
            s_list.remove(ss)

        self.effect_percentiles_pos = [s_percentiles[ss] for ss in s_list]
        self.effect_bins_pos = [ss for ss in s_list]

        self.effect_fraction_muts_in_bin_pos = []
        self.effect_one_over_mutational_input_in_bin_pos = []


        for perc_low, perc_high in zip([0]+self.effect_percentiles_pos,self.effect_percentiles_pos+[1]):
            frac_input_in_bin = perc_high - perc_low
            self.effect_fraction_muts_in_bin_pos.append(frac_input_in_bin)
            mut_input_in_bin = frac_input_in_bin*self._MUT_INPUT # expected # muts in bin per generation

            if mut_input_in_bin <=0: # avoid division by zero
                one_over_frac_input_in_bin = 1
            else:
                one_over_frac_input_in_bin = 1.0/mut_input_in_bin

            self.effect_one_over_mutational_input_in_bin_pos.append(one_over_frac_input_in_bin)

        mycopy = [one_over for one_over in self.effect_one_over_mutational_input_in_bin_pos]
        mycopy.reverse()
        self.effect_one_over_mutational_input_in_bin = mycopy + self.effect_one_over_mutational_input_in_bin_pos

        self.effect_bins = sorted([-x for x in self.effect_bins_pos] + [0.0] + self.effect_bins_pos)
        self.eff_bin_number = len(self.effect_bins) + 1


    def _freeze(self):
        """
        Records -- in a set called _frozen --- which mutations are segregating at the
        instance of use so that the future of those particular mutations can be tracked

        Can use more than once. But each my_time the previous list is replaced
        """
        self._frozen = set() #DO NOT REMOVE. Allows us to replace the mutants if they do not cont
        #the mutants that we want.

        #extra
        MINOR = True
        FROZEN = True
        SCALED = True
        self._num_frozen_fbins = [0.0 for _ in range(self.f_bin_number)]
        self._num_frozen_efs_bins = [0.0 for _ in range(self.eff_bin_number)]

        i = 0
        for mu in self._segregating:
            mu.freeze()
            self._frozen.add(mu)
            i+=1

            #extra
            signed_freq_frozen_minor = mu.signed_freq_prev(minor=MINOR,frozen =FROZEN)
            s_frozen = mu.signed_pheno_size_het_prev(minor =MINOR, frozen=FROZEN, scaled=SCALED)

            fbin_frozen = np.searchsorted(self.freq_bins_minor, signed_freq_frozen_minor)
            ebin_frozen = np.searchsorted(self.effect_bins, s_frozen)

            self._num_frozen_fbins[fbin_frozen] +=1
            self._num_frozen_efs_bins[ebin_frozen] += 1

        self._num_frozen = i

        # collect statistics on extinct fixed mutations that were segregating at the time of the shift
        self._hist_fbins_fixed_extinct = dict()
        self._hist_efs_bins_fixed_extinct = dict()

        fixed_extinct = ['U1_d2ax', 'U0_fixed', 'U0_extinct']

        for stat in fixed_extinct:
            self._hist_fbins_fixed_extinct[stat] = [0.0 for _ in range(self.f_bin_number)]
            self._hist_efs_bins_fixed_extinct[stat] = [0.0 for _ in range(self.eff_bin_number)]

    def _freeze_nm(self):

        self._num_frozen_nm_efs_bins = [0.0 for _ in range(self.eff_bin_number)]
        self._frozen_nm = set()
        self._num_frozen_nm = 0

        self._hist_efs_bins_fixed_extinct_nm = dict()

        fixed_extinct = ['U1_2ax', 'U0_fixed', 'U0_extinct']

        for stat in fixed_extinct:
            self._hist_efs_bins_fixed_extinct_nm[stat] = [0.0 for _ in range(self.eff_bin_number)]


    def _freeze_new_mutation(self,mut):
        """
        Records -- in a set called _frozen --- which mutations are segregating at the
        instance of use so that the future of those particular mutations can be tracked

        Can use more than once. But each my_time the previous list is replaced
        """
        SCALED = True

        mut.freeze()
        self._frozen_nm.add(mut)

        s_frozen = mut.signed_pheno_size_het(minor =False, scaled=SCALED)
        ebin_frozen = np.searchsorted(self.effect_bins, s_frozen)

        self._num_frozen_nm_efs_bins[ebin_frozen] += 1
        self._num_frozen_nm+=1




    @abc.abstractmethod
    def _update_segregating_list(self):
        """
        Updates the segregating mutations list.
        """
        return

    def _update_all_moments(self):
        """
        Updates the moments in _statistics (mean, var, dist, mean_seg) and _even_more_statistics (mu3).
        """
        if self._last_update_moments == 0:
            return
        else:
            self._update_standard_stats()
            self._update_moments_from_essential_moments()
            self._update_after_shift_moments()

            self._last_update_moments = 0

    @abc.abstractmethod
    def _update_essential_moments(self):
        """
        Updates the moments in _statistics (mean, var, dist, mean_seg,mu3,dist_guess).
        """
        prevous_dist = self._dist_ess

        mu3_dist_guess = 0.0

        mean_seg = 0.0
        var = 0.0
        fixed_state_minus_mean_fixed = 0.0
        for mut in self._segregating:
            thesign = mut.derived_sign()
            mean_seg += mut.segregating_effect()*thesign
            var += mut.central_moment(nth=2)*thesign**2
            fixed_state_minus_mean_fixed+= mut.fixed_state_effect() * thesign
            mu3_dist_guess += mut.central_moment(nth=3)*thesign

        mean = mean_seg+self._mean_fixed_ess


        dist = self._FITNESS_OPTIMUM - mean

        self._mean_seg_ess = mean_seg
        self._dist_ess = dist
        self._var_ess = var

        self._fixed_state_minus_mean_fixed_ess = fixed_state_minus_mean_fixed

        if self._have_we_frozen:
            dist_guess = mu3_dist_guess / (2.0 * self._var_ess)

            if self._sums_update_time != self._number_generations_after_freeze:
                dist = self._dist_ess
                var = self._var_ess
                add_dist = prevous_dist
                if self._number_generations_after_freeze ==0:
                    add_dist =0
                self._dist_sum += add_dist
                self._var_sum += var
                self._dist_guess_sum += dist_guess
                self._sums_update_time = self._number_generations_after_freeze



    def _update_standard_stats(self):
        """
        Updates the stats in _update_standard_stats (num_seg, skewness, fixed_state).
        """
        mu3_both, var_both, num = 0.0, 0.0, 0.0

        for mut in self._segregating:
            vari = mut.central_moment(nth=2)
            mu3 = mut.central_moment(nth=3)*mut.derived_sign()
            num += 1
            mu3_both += mu3
            var_both += vari

        self._stats_d['U3_mu3'], self._stats_d['U0_num_seg'] = mu3_both, num
        self._stats_d['U1_dist_guess'], self._stats_d['U0_skewness'] = mu3_both / (2.0 * var_both), mu3_both / var_both ** (3 / 2)


    def _update_after_shift_moments(self):
        if self._have_we_frozen:
            time = float(self._number_generations_after_freeze+1)
            var_mean = self._var_sum / float(time)
            self._stats_d['U1_dist_sum_over_2N_scaled'] = self._dist_sum / float(2.0 * self.N) * self._SPECIAL_SCALING_FACTOR


    def _update_moments_from_essential_moments(self):
        """
        Updates the stats that rely on essential stats
        """
        self._stats_d['U1_dist'], self._stats_d['U2_var'] = self._dist_ess, self._var_ess
        self._stats_d['U2_dist_square'] = self._dist_ess ** 2

        mean_fixed =self._mean_fixed_ess
        fixed_state = self._fixed_state_minus_mean_fixed_ess + mean_fixed
        mean = mean_fixed + self._mean_seg_ess

        self._stats_d['U0_fixed_state_minus_mean_fixed_over_shift'] = self._fixed_state_minus_mean_fixed_ess * self._SHIFT_SIZE_SCALING_FACTOR
        self._stats_d['U0_mean_minus_fixed_state_over_shift'] = (mean - fixed_state) * self._SHIFT_SIZE_SCALING_FACTOR
        self._stats_d['U0_mean_minus_mean_fixed_over_shift'] = (mean - mean_fixed) * self._SHIFT_SIZE_SCALING_FACTOR
        self._stats_d['U0_opt_minus_fixed_state_over_shift'] = (self._FITNESS_OPTIMUM - fixed_state) * self._SHIFT_SIZE_SCALING_FACTOR
        self._stats_d['U0_opt_minus_mean_fixed_over_shift'] = (self._FITNESS_OPTIMUM - mean_fixed) * self._SHIFT_SIZE_SCALING_FACTOR


    def _basic_stats(self):
        """ Returns a dictionary of all the non-hist stats and non-delta stats
        """
        self._update_all_moments()
        return self._stats_d

    def _histo_frozen(self):
        """ Returns 2 dictionaries of current histogram data, according to freq bins and effectsize bins.

        Returns a freq bin dictionary with keys: frozen_var_fbins_H, frozen_numseg_fbins_H,
        frozen_contr_d_2ax_fbins_H, frozen_mean_efs_fbins_H, frozen_mean_d2ax_fbins_H,
        frozen_extinct_fbins_H, frozen_fixed_fbins_H

        And an effectsize bin dictionary with keys: frozen_var_efs_bins_H, frozen_numseg_efs_bins_H,
        frozen_contr_d_2ax_efs_bins_H, frozen_mean_f_efs_bins_H, frozen_mean_d2ax_efs_bins_H,
        frozen_extinct_efs_bins_H, frozen_fixed_efs_bins_H
        """

        hist_fbins = dict()
        hist_efs_bins = dict()
        # collect statistics on segregating mutations

        # we want to record these stats per unit mutational input
        convert_to_per_mut_input = ['U0_numseg']


        normal_stats_all_bin_type = ['U0_numseg', 'U1_d2ax_scaled_per_mut_input']

        normal_stats_efs = []

        normal_stats_f = normal_stats_all_bin_type

        for stat in normal_stats_efs:
            hist_efs_bins[stat] = [0.0 for _ in range(self.eff_bin_number)]
        for stat in normal_stats_f:
            hist_fbins[stat] = [0.0 for _ in range(self.f_bin_number)]


        hist_fbins['U1_d2ax'] = [self._hist_fbins_fixed_extinct['U1_d2ax'][fbinni] for fbinni in range(self.f_bin_number)]
        hist_efs_bins['U1_d2ax']= [self._hist_efs_bins_fixed_extinct['U1_d2ax'][efsbinni] for efsbinni in range(self.eff_bin_number)]

        hist_efs_bins['U0_numseg'] = [0 for _ in range(self.eff_bin_number)]
        hist_fbins['U0_numseg'] = [0 for _ in range(self.f_bin_number)]

        var_total = 0
        d2ax_total_segregating = 0
        d2ax_total_fixed_extinct = np.sum(self._hist_efs_bins_fixed_extinct['U1_d2ax'])

        MINOR = True
        FROZEN = True
        SCALED = True
        for mut in self._frozen:

            signed_freq_frozen_minor = mut.signed_freq_prev(minor=MINOR,frozen =FROZEN)
            s_frozen = mut.signed_pheno_size_het_prev(minor =MINOR, frozen=FROZEN, scaled=SCALED)

            fbin_frozen = np.searchsorted(self.freq_bins_minor, signed_freq_frozen_minor)
            ebin_frozen = np.searchsorted(self.effect_bins, s_frozen)

            # need this for the tatal variance
            var = mut.central_moment(nth=2)

            d2ax_frozen = mut.delta_contrib_to_mean(frozen=FROZEN)
            # x_rel_frozen = mut.x_rel_frozen(minor=MINOR)

            # sum_f_efs_bins[ebin_frozen]+= x_rel_frozen

            hist_efs_bins['U1_d2ax'][ebin_frozen] += d2ax_frozen
            # sum_d2ax_efs_bins[ebin_frozen]+=d2ax_frozen
            hist_fbins['U1_d2ax'][fbin_frozen] += d2ax_frozen

            var_total += var
            d2ax_total_segregating += d2ax_frozen

            if not mut.is_extinct(frozen=FROZEN,minor=MINOR) and not mut.is_fixed(frozen=FROZEN,minor=MINOR):
                hist_fbins['U0_numseg'][fbin_frozen]+= 1
                hist_efs_bins['U0_numseg'][ebin_frozen]+= 1

        if self._var_ess >0:
            self._stats_d['U0_frac_var_seg'] = var_total / self._var_ess
        else:
            self._stats_d['U0_frac_var_seg'] = 0

        if self._FITNESS_OPTIMUM > 0:
            self._stats_d['U0_d2ax_frozen_over_shift'] = (d2ax_total_segregating+d2ax_total_fixed_extinct)/ self._FITNESS_OPTIMUM


        hist_efs_bins['U1_d2ax_scaled_per_mut_input'] = self._convert_total_effect_bins_to_per_unit_mut_input(hist_efs_bins['U1_d2ax'],
                                                                                                                 scaled=True)
        hist_fbins['U1_d2ax_scaled_per_mut_input'] = self._convert_total_freq_bins_to_per_unit_mut_input(
            hist_fbins['U1_d2ax'], scaled=True)


        if self._FITNESS_OPTIMUM > 0:
            mulit = 1.0/float(self._FITNESS_OPTIMUM)
        else:
            mulit = 1.0

        hist_efs_bins['U0_d2ax_over_shift'] = [d2ax*mulit for d2ax in hist_efs_bins.pop('U1_d2ax')]
        hist_fbins['U0_d2ax_over_shift'] = [d2ax * mulit for d2ax in hist_fbins.pop('U1_d2ax')]

        # convert the stats that need to be converted to per unit mutational input
        for stat in hist_efs_bins:
            if stat in convert_to_per_mut_input:
                hist_efs_bins[stat+'_per_mut_input']= self._convert_total_effect_bins_to_per_unit_mut_input(hist_efs_bins.pop(stat))

        for stat in hist_fbins:
            if stat in convert_to_per_mut_input:
                hist_fbins[stat+'_per_mut_input']= self._convert_total_freq_bins_to_per_unit_mut_input(hist_fbins.pop(stat))

        for statkey in list(hist_fbins.keys()):
            units, statbody = statkey[:3], statkey[3:]
            hist_fbins[units+'frozen_'+ statbody] = hist_fbins.pop(statkey)
        for statkey in list(hist_efs_bins.keys()):
            units, statbody = statkey[:3], statkey[3:]
            hist_efs_bins[units+'frozen_'+ statbody] = hist_efs_bins.pop(statkey)

        return hist_fbins, hist_efs_bins

    def _histo_frozen_nm(self):
        """ Returns a dictionaries of current histogram data, according to effectsize bins.
        hist_efs_bins with hist_efs_bins[stat][bin] = value
        """
        hist_efs_bins = dict()
        # collect statistics on segregating mutations

        convert_to_per_mut_input =['U1_2ax_scaled']

        stats_both = ['U1_2ax_scaled']


        stats_efs = stats_both

        for stat in stats_efs:
            hist_efs_bins[stat] = [0.0 for _ in range(self.eff_bin_number)]

        # these are the stats that we take an average value in the bin for

        hist_efs_bins['U1_2ax']= [self._hist_efs_bins_fixed_extinct_nm['U1_2ax'][efsbinni] for efsbinni in range(self.eff_bin_number)]

        MINOR = False
        FROZEN = True
        SCALED = True
        for mut in self._frozen_nm:

            #s_frozen = mut.signed_pheno_size_het(minor=MINOR, scaled=SCALED)
            s_frozen = mut.signed_pheno_size_het_prev(minor=MINOR, frozen=FROZEN, scaled=SCALED)

            ebin_frozen = np.searchsorted(self.effect_bins, s_frozen)

            ax_frozen = mut.contrib_to_mean_derived()

            hist_efs_bins['U1_2ax'][ebin_frozen] += ax_frozen


        hist_efs_bins['U1_2ax_scaled'] = [hist_efs_bins['U1_2ax'][efsbinni]*self._SPECIAL_SCALING_FACTOR for efsbinni in range(self.eff_bin_number)]

        if self._FITNESS_OPTIMUM > 0:
            mulit = 1.0 / float(self._FITNESS_OPTIMUM)
        else:
            mulit = 1.0
        hist_efs_bins['U0_2ax_over_shift'] = [d2ax * mulit for d2ax in hist_efs_bins.pop('U1_2ax')]

        # convert the stats that need to be converted to per unit mutational input
        for stat in hist_efs_bins:
            if stat in convert_to_per_mut_input:
                hist_efs_bins[stat+'_per_mut_input']= self._convert_total_effect_bins_to_per_unit_mut_input(hist_efs_bins.pop(stat), scaled=False)

        for statkey in list(hist_efs_bins.keys()):
            units, statbody = statkey[:3], statkey[3:]
            hist_efs_bins[units + 'frozen_nm_' + statbody] = hist_efs_bins.pop(statkey)

        return hist_efs_bins


    def _frozen_nm_remove_fixed_extinct(self):
        """ removes extinct and fixed mutations from the frozen set
        """
        fixed_extinct_mutants = set()
        SCALED = True
        for mut in self._frozen_nm:

            s_frozen = mut.signed_pheno_size_het(minor=False, scaled=SCALED)
            ebin_frozen = np.searchsorted(self.effect_bins, s_frozen)


            if mut.is_extinct(frozen=False,minor=False):
                fixed_extinct_mutants.add(mut)
                self._hist_efs_bins_fixed_extinct_nm['U0_extinct'][ebin_frozen] += 1
            elif mut.is_fixed(frozen=False,minor=False):
                fixed_extinct_mutants.add(mut)
                self._hist_efs_bins_fixed_extinct_nm['U0_fixed'][ebin_frozen] += 1

            if mut in fixed_extinct_mutants:
                twoax_frozen = mut.contrib_to_mean_derived()
                self._hist_efs_bins_fixed_extinct_nm['U1_2ax'][ebin_frozen] += twoax_frozen


        for mut in fixed_extinct_mutants:
            self._frozen_nm.remove(mut)


    def _frozen_remove_fixed_extinct(self):
        """ removes extinct and fixed mutations from the frozen set
        """
        fixed_extinct_mutants = set()
        MINOR = True
        FROZEN = True
        SCALED = True
        for mut in self._frozen:

            signed_freq_frozen_minor = mut.signed_freq_prev(minor=MINOR,frozen =FROZEN)
            s_frozen = mut.signed_pheno_size_het_prev(minor =MINOR, frozen=FROZEN, scaled=SCALED)

            fbin_frozen = np.searchsorted(self.freq_bins_minor, signed_freq_frozen_minor)
            ebin_frozen = np.searchsorted(self.effect_bins, s_frozen)


            if mut.is_extinct(frozen=FROZEN,minor=MINOR):
                fixed_extinct_mutants.add(mut)
                self._hist_fbins_fixed_extinct['U0_extinct'][fbin_frozen] +=1
                self._hist_efs_bins_fixed_extinct['U0_extinct'][ebin_frozen] += 1

            elif mut.is_fixed(frozen=FROZEN,minor=MINOR):
                fixed_extinct_mutants.add(mut)
                self._hist_fbins_fixed_extinct['U0_fixed'][fbin_frozen] += 1
                self._hist_efs_bins_fixed_extinct['U0_fixed'][ebin_frozen] += 1

            if mut in fixed_extinct_mutants:
                d2ax_frozen = mut.delta_contrib_to_mean(frozen=FROZEN)
                self._hist_efs_bins_fixed_extinct['U1_d2ax'][ebin_frozen] += d2ax_frozen
                self._hist_fbins_fixed_extinct['U1_d2ax'][fbin_frozen] += d2ax_frozen

        for mut in fixed_extinct_mutants:
            self._frozen.remove(mut)

    def _convert_total_freq_bins_to_per_unit_mut_input(self, histo_freq_bins, scaled=False):
        return self._convert_total_bins_to_per_unit_mut_input(self.freq_one_over_mutational_input_in_bin, histo_freq_bins, scaled)

    def _convert_total_effect_bins_to_per_unit_mut_input(self, histo_efs_bins, scaled=False):
        return self._convert_total_bins_to_per_unit_mut_input(self.effect_one_over_mutational_input_in_bin, histo_efs_bins, scaled)

    def _convert_total_bins_to_per_unit_mut_input(self, one_over_mutational_input_in_bin_bins, histo_bins, scaled=False):

        if scaled:
            scaling = self._SPECIAL_SCALING_FACTOR
        else:
            scaling = 1.0

        histo_bins_per_unit_mut_input = []
        for one_over_mut_input, totali in zip(one_over_mutational_input_in_bin_bins, histo_bins):
            multiplic = one_over_mut_input * scaling
            histo_bins_per_unit_mut_input.append(multiplic * totali)

        return histo_bins_per_unit_mut_input

    def _combine_pos_neg_list(self,list,add=True):
        if (len(list) %2) != 0:
            return []
        else:
            num_nonzero_entries = int(len(list)/2)
            new_list = [0 for _ in range(num_nonzero_entries)]
            for i in range(num_nonzero_entries):
                if add:
                    new_list[i] = list[num_nonzero_entries+i]+list[num_nonzero_entries-1-i]
                else:
                    new_list[i] = list[num_nonzero_entries+i]-list[num_nonzero_entries-1-i]
            return new_list


    def _var_0_units_Vs(self):
        to_integrate = lambda sss: 4.0 * np.sqrt(np.abs(sss)) * dawsn(np.sqrt(np.abs(sss)) / 2.0)*self._squared_effect_size_dist.pdf(sss)
        b = 2.0*self._squared_effect_size_dist.ppf(0.99999999999999)
        integral = quad(to_integrate, 0,b)
        return integral[0]*self._mu.mu

    def _one_over_ratio_V_to_F_minus_one(self):
        to_integrate = lambda sss: 4.0 * np.sqrt(np.abs(sss)) * dawsn(
            np.sqrt(np.abs(sss)) / 2.0) * self._squared_effect_size_dist.pdf(sss)
        to_integrate_second = lambda sss: 2.0 * np.sqrt(np.abs(sss)) ** 3*np.exp(-np.abs(sss)/4.0) / (np.sqrt(np.pi) *
                                                                             erf(np.sqrt(np.abs(sss)) / 2.0)) * self._squared_effect_size_dist.pdf(sss)

        b = 2.0 * self._squared_effect_size_dist.ppf(0.99999999999999)
        integral = quad(to_integrate, 0,b)
        integral_second = quad(to_integrate_second, 0,b)

        my_factor = (integral[0]-integral_second[0])/integral_second[0]
        if my_factor > 10**(-6):
            return 1.0/float(my_factor)
        else:
            return 1.0

    def _one_over_ratio_V_to_F(self):
        to_integrate = lambda sss: 4.0 * np.sqrt(np.abs(sss)) * dawsn(
            np.sqrt(np.abs(sss)) / 2.0) * self._squared_effect_size_dist.pdf(sss)
        to_integrate_second = lambda sss: 2.0 * np.sqrt(np.abs(sss)) ** 3*np.exp(-np.abs(sss)/4.0) / (np.sqrt(np.pi) *
                                                                             erf(np.sqrt(np.abs(sss)) / 2.0)) * self._squared_effect_size_dist.pdf(sss)

        b = 2.0 * self._squared_effect_size_dist.ppf(0.99999999999999)
        integral = quad(to_integrate, 0,b)
        integral_second = quad(to_integrate_second, 0,b)

        my_factor = integral_second[0]/integral[0]
        return my_factor

    def _one_over_sigma_0_units_rootVs(self):
        to_integrate = lambda sss: 4.0 * np.sqrt(np.abs(sss)) * dawsn(
            np.sqrt(np.abs(sss)) / 2.0) * self._squared_effect_size_dist.pdf(sss)

        b = 2.0 * self._squared_effect_size_dist.ppf(0.99999999999999)
        integral = quad(to_integrate, 0,b)

        my_factor = self._mu.mu*integral[0]
        return 1/np.sqrt(my_factor)


class PopulationExact(_PopulationBasic):
    """Container class for the population keeping track of members

    With the exact version of this class we actually keep track of ever population member each
    generation.

    Parameters:
        N: (int) Population size
        Vs: (float) Fitness parameter
        mu: (namedtuple) MutationalProcess (U, shape, scale)
        selectionMode:  {'parents', 'offspring'}, optional

    Attributes:
        _individuals: (list) List of N defaultdict[int] <- Member of the population
        _offspring: (list) List of N defaultdict[int] <- Member of the population offspring
        _last_update_seg_list: (int) Number of generations ago that _segregating mutations were updated
        _last_update_ess_mom: (int) Number of generations ago that essential moments were updated.
                    These moments are not essential in evolving the full population, only recording stats
"""

    def __init__(self, N, Vs, mu,  selection_mode ='parents'):
        super(self.__class__, self).__init__(N, Vs, mu)

        # list of mutations carried by each individual of the population:
        # self._individuals[i] is a dictionary representing the genotype of individual i in the population:
        #   - The keys are mutations present in the individual
        #   - Values are the ploidity of the mutation (either 1 or 2)
        self._individuals = [defaultdict(int) for _ in range(self.N)]

        # same data structure as self._individuals is used as a temporary data-structure when constructing next generation
        self._offspring = [defaultdict(int) for _ in range(self.N)]

        # segregation update counter
        self._last_update_seg_list = 0

        # essential moment update counter
        self._last_update_ess_mom = 0

        self._COLLECT_MEAN_FITNESS = True

        if self._COLLECT_MEAN_FITNESS:
            self._exact_population_moments = ['U0_mean_fit']
        # self._exact_population_moments += ['U0_r_2', 'U0_r_2_std', 'U0_LD', 'U0_LD_std']

            for momi in self._exact_population_moments:
                self._stat_set.add(momi)

            for m in self._exact_population_moments:
                self._stats_d[m] = 0.
            for om in self._exact_population_moments:
                self._old_stats_d[om] = 0.

        # read selection mode
        if selection_mode == 'parents':
            self._sample_offspring = self._sample_offspring_by_parents_fitness
        elif selection_mode == 'offspring':
            self._sample_offspring = self._sample_offspring_by_children_fitness

    def next_gen(self):
        """
        Generates the next generation of offspring.

        Segregating mutations are updated every auto_update = 64 generations unless
        manually updated sooner with method _update_segregating_list
        """

        # Not sure what is optimal
        auto_update = 64

        # increment generation counter
        self._last_update_seg_list += 1
        self._last_update_moments += 1
        self._last_update_ess_mom += 1

        # create all offspring:
        self._sample_offspring()
        # advance generation:

        # swap individuals and offspring
        self._individuals, self._offspring = self._offspring, self._individuals

        # once every few iterations, update the segregating mutations list:
        if self._last_update_seg_list >= auto_update:
            self._update_segregating_list()

        self._update_generations_after_freeze_related()

    def basic_stats(self):
        """ Returns a dictiionary of the current non-delta or histogram stats.

        Also adds the value of r_2 (coefficient of correlation squared) averaged over all pairs of muts,
        and the LD
        """
        if not self._last_update_ess_mom == 0:
            self._update_essential_moments()
        if not self._last_update_moments == 0:
            self._update_all_moments()

        if self._COLLECT_MEAN_FITNESS:
            self._update_exact_population_mean_fitness()

        return super(self.__class__, self).basic_stats()


    def histo_stats(self, frozen = False):
        """
        Returns a dictionary of the current histogram data
        """
        hist = super(self.__class__, self).histo_stats(frozen=frozen)

        # #Not so efficient --- going to calculate LD lists twice
        # _, _, _, _, r_2_quantiles, LD_quantiles = self._calculate_av_LD()
        # hist['r_2_quantiles_H'] = r_2_quantiles
        # hist['LD_quantiles_H'] = LD_quantiles
        return hist


    def _update_essential_moments(self):
        """
        """
        if self._last_update_ess_mom == 0:
            return

        self._update_segregating_list()
        super(self.__class__, self)._update_essential_moments()

        self._last_update_ess_mom = 0

    def _update_exact_population_mean_fitness(self):

        self._calculate_mean_fitness()


    def _calculate_mean_fitness(self):
        """
        Calculate mean fitness of population. Only possible with exact sims
        """
        sum_fit = 0.0

        for indv in self._individuals:
            pheno = self._phenotype(indv)
            sum_fit += self._fitness_of_pheno(pheno)

        mean_fit = sum_fit/float(self.N)
        self._stats_d['U0_mean_fit'] = mean_fit

    @staticmethod
    def _mean_var_mu3(sum, sum_squares,sum_cubes, num = None):
        if num is None:
            mean, var, mu3 = 0, 0, 0
        else:
            num =float(num)
            if num > 0:
                mean = float(sum) / num
                mean_sum_squares = sum_squares/num
                mean_sum_cubes = sum_cubes/num
                var = mean_sum_squares - mean  ** 2
                mu3 = mean_sum_cubes-3.0*mean*mean_sum_squares+2*mean**3

            else:
                mean, var, mu3 = 0,0, 0

        return mean, var, mu3

    def _remove_mut(self, mut):
        """
        Removes the mutation mut from all offspring.
        """
        for indv in self._individuals:
            del indv[mut]

    def _reset_frequencies(self):
        """
        Recalcualtes the frequencies of all the mutations in the population
        """

        # reset counts in storage frequency
        for mu in self._segregating:
            mu._frequency_class.store_freq = 0

        # update counts of "current freq"
        for indv in self._individuals:
            for mu,ploidity in indv.items():
                mu._frequency_class.store_freq += ploidity

        # update frequency of muts
        for mut in self._segregating:
            new_freq = mut._frequency_class.store_freq
            mut.update_freq(new_freq)


    def _update_segregating_list(self):
        """
        Updates the segregating mutations list.

        - Computes the frequency_pos of each mutation
        - Removes extinct or fixed mutations
        - Set _last_update to zero.
        - Note that this occurs automatically every 64 generations
        """

        if self._last_update_seg_list == 0:
            return


        #Recalculate frequencies of muations
        self._reset_frequencies()

        self._remove_extinct_fixed_from_seg()

        #self._numSeg = len(self._segregating)

        # reset last-update my_time
        self._last_update_seg_list = 0


    def _phenotype(self,indv):

        """Computes the phenotype of an individual

        Parameters:
            indv: (dict) The member of the population whose fitness is to be computed
            (segregating genotype dictionary)

        Returns: (float) The phenotype of the indvidual
        """

        a2_total = 0

        # sum the effect of segregating mutations
        for mu,ploidity in indv.items():
            a_total = mu.pheno_size_homozyg()
            derivedsign = mu.derived_sign()
            a2_total += ploidity*a_total*derivedsign

        pheno_total = a2_total*0.5+self._mean_fixed_ess

        return pheno_total

    def _fitness_of_pheno(self, phenotype):
        """Computes the fitness of a phenotype

        Parameters:
            phenotype: (float) The phenotype of the member of the population whose fitness is to be computed

        Returns: (float) The fitness of the indvidual
        """
        a2_total_square = 0.0
        a2_total_square += (phenotype - self._FITNESS_OPTIMUM) ** 2

        return math.exp(- a2_total_square * self._FITNESSCOEF)

    def _fitness_of_indv(self, indv):
        """Computes the fitness of an individual

        Parameters:
            indv: (dict) The member of the population whose fitness is to be computed
            (segregating genotype dictionary)

        Returns: (float) The fitness of the indvidual
        """
        pheno_total = self._phenotype(indv)
        return self._fitness_of_pheno(pheno_total)


    def _sample_offspring_by_parents_fitness(self):
        """
        sample self._offspring, based on parents fitness

        We will sample parents with probabilities proportional to their fitness.
        First, we compute a list of the fitnesss of each member of the population
        Then normalize to a probability vector and compute the CDF.
        For every empty child slot (N of them) in offspring pick random parents based on fitness
        Then use method _sample_child to create the child
        """

        # Compute the fitness
        fit_vals = [self._fitness_of_indv(self._individuals[i]) for i in range(self.N)]
        # Normalize to a probability vector and compute the CDF.
        tmp = 1.0 / np.sum(fit_vals)
        cdf = np.add.accumulate([x*tmp for x in fit_vals])

        # generate offspring one by one
        for child in self._offspring:

            # sample two non-identical random parents from the cdf
            p1, p2 = np.searchsorted(cdf,random.random()), np.searchsorted(cdf,random.random())
            while p1 == p2:
                p2 = np.searchsorted(cdf,random.random())

            # generate random child of these two parents
            self._sample_child(child, self._individuals[p1], self._individuals[p2])


    def _sample_offspring_by_children_fitness(self):
        """
        Sample self._offspring, based on offspring fitness.

        We use viability selection to accept or reject randomly generated offspring.
        Generate offspring one by one.
        Pick 2 parents randomly
        Then use method _sampleChild to create the child
        Reject or accept the resulting child with probability proportional to its fitness
        Keep going till we have N little brats
        """

        # generate offspring one by one
        index = 0
        while index < self.N:

            # sample distinct random parents uniformly
            p1, p2 = random.randint(0,self.N - 1), random.randint(0,self.N - 1)

            while p1 == p2:
                p2 = random.randint(0,self.N - 1)

            # generate random child of these two parents
            self._sample_child(self._offspring[index], self._individuals[p1], self._individuals[p2])

            # reject or accept the resulting child with probability proportional to its fitness
            if random.random() < self._fitness_of_indv(self._offspring[index]):
                index+= 1

   # samples one random child of parents p1 and p2
    def _sample_child(self, child, p1, p2):
        """
        Takes a child and 2 parents and generates a random child with new mutations

        The childs dictionary gets cleard
        The each mutation of the parents is given to it with probability 0.5
        if ploidity 1 or definitely if ploidity 2. Then de novo mutations are
        added according to a poisson distribution with mean 2*mu
        """

        # clear offspring dictionary
        child.clear()

        # for each parent
        for p in [p1, p2]:

            # for each mutation carried by the parent
            for mu,ploidity in p.items():

                # if the parent has two copies of the mutation he is bound to pass it on
                if ploidity == 2:
                    child[mu] += 1

                # if the parent is heterozygous, s/he passes it on with probabilty 0.5
                elif random.getrandbits(1):
                    child[mu] += 1

        # add random de-novo mutations:
        # number of de-novo mutations is a poisson variable
        for _ in range(np.random.poisson(2.0*self._mu.mu)):

            # we add the mutation to the segregating list and to the new offspring (in heterozygous state)
            mu = self._get_mutation()

            self._segregating.add(mu)
            child[mu] = 1


class PopulationWF(_PopulationBasic):
    """
    Container class for info about population every generation

    With the new version of this class we do not keep track of every population member each
    generation. Instead we merely keep track of a list of segregating mutations in the populations
    updating using a Wright-Fisher process

    Parameters:
        N: (int) Population size
        Vs: (float) Fitness parameter
        mu: (namedtuple) MutationalProcess (U, shape, scale)
    """

    def __init__(self, N, Vs, mu):
        super(self.__class__, self).__init__(N, Vs, mu)

    def next_gen(self):
        """
        Progresses to the next generation.
        """

        self._last_update_moments += 1

        #Update mutation frequencies via Wright-Fisher
        self._wright_fisher()

        # Add de novo mutations
        self._new_mutations()

        # Remove fixed and extinct mutations from list and updates fixed effect (meanFixed):
        self._remove_extinct_fixed_from_seg()

        #update mean phenotype and variance in phenotype

        self._update_essential_moments()

        #counts time after the shift
        self._update_generations_after_freeze_related()


    def basic_stats(self):
        """
        Returns a dictiionary of the basic stats
        """
        self._update_segregating_list()
        return super(self.__class__, self).basic_stats()


    def histo_stats(self, frozen = None):
        """
        Returns a dictionary of the current histogram data.

        If (frozen == True) also has histogram data for the frozen mutants.
        """
        if frozen is None:
            frozen = False
        return super(self.__class__, self).histo_stats(frozen=frozen)

    def _power(self, r_diff, varies, p, c, a):
        """
        a is the phenotypic effect of the mutation. c is 0,0.5 or 1. r_diff is the
        signed difference ( optimum phenotype -mean phenotype)
        """
        return -(r_diff - a*(c-p))**2*float(varies)


    def _wright_fisher(self):
        """
        Updates the list of segregating mutations according to a Wright-Fisher process
        """

        #varies = 1.0/(2 * (self._stats_d['U2_var'] + self.w ** 2))
        varies = 1.0 / (2.0 * float(self.Vs))
        #r_diff = self._stats_d['U1_dist']
        for mut in self._segregating:
            p = mut.x()
            q = 1-p
            a = mut.pheno_size_homozyg()
            c_00, c_01, c_11 = 0, 0.5, 1

            power_00 = 0.0
            power_01 = 0.0
            power_1 = 0.0

            r_diff = self._dist_ess
            derivedsign = mut.derived_sign()
            power_00 += self._power(r_diff, varies, p, c_00, a*derivedsign)
            power_01 += self._power(r_diff, varies, p, c_01, a*derivedsign)
            power_1 += self._power(r_diff, varies, p, c_11, a*derivedsign)

            w_00 = math.exp(power_00)
            w_01 = math.exp(power_01)
            w_11= math.exp(power_1)

            meanfit = p**2*w_11 +q**2*w_00 +2*p*q*w_01
            weight = (p**2*w_11 + p*q*w_01)/meanfit
            if weight >1:
                print('weight:', weight, ' p:', p,' r_diff:', self._dist_ess, ' phenoSize:', a)
                raise Exception("The weight is bigger than one.")
            mut.update_freq(np.random.binomial(2*self.N, weight))

    def _update_segregating_list(self):
        """
        Does nothing when there are no individual population members
        """
        return

    def _new_mutations(self):
        """
        Add random de-novo mutations to segregating list.
        The number of de-novo mutations is a poisson variable with mean 2Nmu
        """

        for _ in range(np.random.poisson(2.0*self.N*self._mu.mu)):
            mu = self._get_mutation()
            self._segregating.add(mu)

    def _remove_mut(self, mut):
        """
        Does nothing when there are no individual population members
        """
        return

    def _update_essential_moments(self):
        """
        """
        super(self.__class__, self)._update_essential_moments()




# def get_equal_percentile_bins(data,number_bins):
#     percs = [0, 50,100] # list of number_bins percentiles
#     t = scoreatpercentile(data,0.5)
