import numpy as np

class _MutationBasic(object):
    """
    Container class for mutations.

    Parameters:
        scaled_size: (float) The scaled size of the mutation
        pheno_size: (float) The phenotypic effect size of the mutation.
        N: (int) The size of the population in which the mutation appears

    Attributes:
        _N: (int) The size of the population in which the mutation appears.
        _DENOMINATOR: (float) 1/(2*_N)
        PHENO_SIZE_HOMOZYG: (float) The phenotypic effect size of the mutation in a homozygote.
        SCALED_SIZE: (float) The scaled selection coefficient of the mutation at steady-state
    """
    # metaclass = abc.ABCMeta

    def __init__(self, scaled_size, pheno_size_homozyg, N=10000):

        # Making sure scaled and pheno size are positive
        if scaled_size < 0:
            scaled_size = -scaled_size
        if pheno_size_homozyg < 0:
            pheno_size_homozyg = -pheno_size_homozyg

        self.SCALED_SIZE = scaled_size
        self.PHENO_SIZE_HOMOZYG = pheno_size_homozyg

        # population size
        self._N = N
        # useful factor to multiply stuff with
        self._DENOMINATOR = 1.0 / (2.0 * float(self._N))
        self._DETERMINISTIC_DEATH_THRESHOLD = self._DENOMINATOR/2.0


    def N(self):
        return self._N

    def pheno_size_homozyg(self):
        return self.PHENO_SIZE_HOMOZYG


    def _x(self,frequency, minor=False):
        """
        Returns the fraction haplotypes with the mutation.

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
        Returns:
            (float) If not minor, the fraction of the 2*_N haplotypes with the mutation.
            Otherwise, the fraction of the 2*_N haplotypes with the minor mutation.
        """
        if not minor:
            return frequency * self._DENOMINATOR
        else:
            return 1-frequency * self._DENOMINATOR


    def _is_fixed(self,frequency):
        if frequency == 2 * self._N:
            return True
        else:
            return False


    @staticmethod
    def _is_extinct(frequency):
        if frequency == 0:
            return True
        else:
            return False

    def _is_segregating(self,frequency):
        if frequency == 0 or frequency == 2 * self._N:
            return False
        else:
            return True


    def _is_minor(self,frequency):
        """
        Tells us if the variant is minor.
        """
        if frequency > self._N:
            return False
        else:
            return True

    def _central_moment(self,frequency,nth=3):
        """
        Returns the current contribution of the mutant to
        the nth central moment in the phenotype distribution.
        if nth <= 4
        """
        if nth == 1:
            return 0.0
        if nth == 2:
            return self._var(frequency)
        elif nth == 3:
            return self._mu3(frequency)
        elif nth ==4:
            self._mu4(frequency)
        else:
            return 0.0


    def _var(self,frequency):
        """
        Returns the current contribution of the mutant to phenotypic variance in all dimensions.
        """
        p = frequency * self._DENOMINATOR
        q = 1 - p
        a = self.PHENO_SIZE_HOMOZYG
        return 0.5 * (a) ** 2 * p * q

    def _mu3(self,frequency):
        """
        Returns the current contribution of the mutant to
        the third moment in the phenotype distribution.
        """
        p = frequency * self._DENOMINATOR
        a = self.PHENO_SIZE_HOMOZYG
        q = 1 - p
        return p * q * (0.5 - p) * a ** 3 / 2.0

    def _mu4(self,frequency):
        """
        Returns the current contribution of the mutant to
        the fourth moment in the phenotype distribution.
        """
        p = frequency * self._DENOMINATOR
        a = self.PHENO_SIZE_HOMOZYG
        q = 1 - p
        return p * q * a ** 4 / 8.0

    def _cumulant(self,frequency,nth=4):
        """
        Returns the current contribution of the mutant to
        the nth cumulant in the phenotype distribution.
        if nth <= 6

        """
        if nth == 2:
            return self._var(frequency)
        elif nth == 3:
            return self._mu3(frequency)
        elif nth ==4:
            self._c4(frequency)
        elif nth == 5:
            return self._c5(frequency)
        elif nth ==6:
            return self._c6(frequency)
        else:
            return 0

    def _c4(self,frequency):
        """
        Returns the current contribution of the mutant to
        the fourth cumulant in the phenotype distribution.

        """
        p = frequency * self._DENOMINATOR
        a = self.PHENO_SIZE_HOMOZYG
        q = 1 - p
        return (a ** 4 / 8.0) * p * q * (1 - 6 * p * q)

    def _c5(self,frequency):
        """
        Returns the current contribution of the mutant to
        the fitht cumulant in the phenotype distribution.

        """
        p = frequency * self._DENOMINATOR
        a = self.PHENO_SIZE_HOMOZYG
        q = 1 - p
        return (a ** 5 / 16.0) * p * q * (1 - 2 * p) * (1 - 12 * p * q)

    def _c6(self,frequency):
        """
        Returns the current contribution of the mutant to
        the sixth cumulant in the phenotype distribution.

        """
        p = frequency * self._DENOMINATOR
        a = self.PHENO_SIZE_HOMOZYG
        q = 1 - p
        return (a ** 6 / 32.0) * p * q * (1 - 30 * p * q * (1 - 2 * p) ** 2)

    def _fixing_effect(self):
        """
        Returns the phenotypic effect of the mutant fixing.
        """
        return self.PHENO_SIZE_HOMOZYG

    def _segregating_effect(self,frequency):
        """
        Returns the contribution of the mutant to the average phenotype.
        """
        return frequency * self._DENOMINATOR * self.PHENO_SIZE_HOMOZYG

    def _delta_x(self, prev_frequency,frequency):
        """
        Returns the change in frequency

         Parameters:
            prev_frequency: (int) The previous frequency
            mutant
            frequency: (int) The current frequency
        Returns:
            (float) The change in frequency/(2N).
        """
        return (frequency - prev_frequency) * self._DENOMINATOR


    def _contrib_to_mean(self, frequency):
        """
        Returns the current contribution to the mean pheno, which is =
        homozygous_phenotypic_effect*frequency/2N
        """
        return self._x(frequency) * self.PHENO_SIZE_HOMOZYG

    def _delta_central_moment(self,prev_frequency,frequency,nth=1):
        """
        Returns the change in contribution of the mutant to
        the nth central moment in the phenotype distribution.
        if nth <= 4
        """
        if nth == 1:
            return self._delta_x(prev_frequency, frequency) * self.PHENO_SIZE_HOMOZYG
        if nth == 2:
            return self._var(frequency) - self._var(prev_frequency)
        elif nth == 3:
            return self._mu3(frequency) - self._mu3(prev_frequency)
        elif nth ==4:
            return self._mu4(frequency) - self._mu4(prev_frequency)
        else:
            return 0.0


class _MutationFrequencies(object):
    """
    Container class for a mutation's frequencies. No phenotypic size information included.

    Parameters:
        N: (int) The size of the population in which the mutation appears

    Attributes:
        _N: (int) The size of the population in which the mutation appears.
        _denominator: (float) 1/(2*_N)
        frequency: (int) The number of times the mutation currently occurs in the population.
        prev_freq: (int) The frequency of the mutation in the previous generation.
        frozen_freq: (int) The frequency_pos of the mutation when the population was frozen.
        lifetime: (int) Counts the number of times the mutant has had its frequency updated
        index: (int) A positive int that keeps track of which frozen mutation this one is.
                Set to -1 if the mutation has not been frozen.
        minor_status: (bool) True if the derived mutant is minor.
        prev_minor_status: (bool) True if the derived mutant was minor in the previous generation.
    """

    def __init__(self, N=10000):

        # useful factor to multiply stuff with
        self._N = N
        self._DENOMINATOR = 1.0 / (2.0 * float(self._N))

        self.frequency = 1
        self.prev_freq = 0  # records the freq in the previous generation
        self.frozen_freq = -1  # start undefined
        self.store_freq = 0 # to store the frequency

        # Is the mutant minor or not
        self.minor_status = True
        self.prev_minor_status = True

        # How many times muts has had its frequency updated
        self.lifetime = 0

    def _minor_frequency(self,frequency):
        """
        Returns the frequency of the minor allele
        """
        if frequency <= self._N:
            return frequency
        else:
            return 2*self._N- frequency

    def _x(self,frequency, minor=False):
        """
        Returns the fraction haplotypes with the mutation - frequency/(2N)

        Parameters:
            minor: (boolean) True if we are interested in the fraction haplos with the minor mutant
        Returns:
            (float) If not minor, the fraction of the 2*_N haplotypes with the mutation.
            Otherwise, the fraction of the 2*_N haplotypes with the minor mutation.
        """
        if not minor:
            return frequency * self._DENOMINATOR
        else:
            return 1-frequency * self._DENOMINATOR

    def _previous_frequency_of_current_minor(self):
        """
        Returns the frequency in the previous generation of the allele that is currently minor
        """
        prev_minor_frequency = self.prev_freq

        if self.prev_minor_status and not self.minor_status:  # if it was minor, but now isn't
            prev_minor_frequency = self.prev_freq
            # but we've flipped so
            prev_minor_frequency = 2 * self._N - prev_minor_frequency

        if not self.prev_minor_status and self.minor_status:  # if it was not minor, but now is
            prev_minor_frequency = self.prev_freq

        if self.prev_minor_status and self.minor_status:  # if was minor, and still is
            prev_minor_frequency = self.prev_freq

        if not self.prev_minor_status and not self.minor_status:  # if it wasn't minor, and still isn't
            prev_minor_frequency = 2 * self._N - self.prev_freq

        return prev_minor_frequency

    def update_freq(self, frequency, update=None):
        """
        Updates the frequency of the mutation.

        Updates the mutations attributes: frequency, minor_status, prev_minor_status
         and (possibly) liftetime

        Parameters:
            frequency: The new frequency_pos of the mutation
            update: (boolean) If true, adds a generation to the mutants total life
        """
        if self.is_segregating():
            self.prev_minor_status = self.minor_status
            self.prev_freq = self.frequency

            self.frequency = frequency

            if self.frequency <= self._N:
                self.minor_status = True
            else:
                self.minor_status = False

            if update is None:
                update = True
            if update:
                self.lifetime += 1

    def freeze(self):
        """Record's the mutation's current freq as the frozen freq. Updates attribute frozen_freq """
        self.frozen_freq = self.frequency

    def x(self, minor=False):
        """
        Returns the fraction haplotypes with the mutation.(=frequency/2N)

        Parameters:
            minor: (boolean) True if we are interested in the fraction of haplos with the minor allele
        Returns:
            (float) If not minor, the fraction of the 2*_N haplotypes with the mutation.
            Otherwise, the fraction of the 2*_N haplotypes with the minor mutation.
        """
        if not minor:
            frequency = self.frequency
        else:
            frequency = self._minor_frequency(self.frequency)
        return self._x(frequency)

    def x_prev(self, minor=False, frozen=False):
        """
        Returns the fraction haplotypes with the mutation in the previous generation
        or at freeze time, if frozen is True.

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
            frozen: (boolean) True if we are interested in freq at freeze my_time.
        Returns:
            (float) If not minor, the fraction of the 2*_N haplotypes with the mutation
            in the previous generation.
            Otherwise, the fraction of the 2*_N haplotypes with the (currently) minor
            mutation in the previous generation.
        """
        if not frozen:
            if not minor:
                return self.prev_freq * self._DENOMINATOR
            else:
                return self._previous_frequency_of_current_minor() * self._DENOMINATOR
        else:
            if not minor:
                return self.frozen_freq * self._DENOMINATOR
            else:
                return self._previous_frequency_of_current_minor() * self._DENOMINATOR


    def x_rel_frozen(self, minor=False):
        """
        Returns the fraction haplotypes with the mutation, relative to the frozen frequency.

        Parameters:
            minor: (boolean) True if we are interested in the (frozen) minor mutant
        Returns:
            (float) If not minor, the fraction of the 2*_N haplotypes with the mutation.
            Otherwise, the fraction of the 2*_N haplotypes mutation that was minor at
            freeze my_time.
        """
        x = self._x(self.frequency)
        if not minor:
            return x
        else:
            if self.frozen_freq < self._N:
                return x
            else:
                return 1 - x

    def is_minor(self):
        """
        Tells us if the variant is minor.
        """
        if self.frequency > self._N:
            return False
        else:
            return True

    def is_segregating(self):
        """
        Tells us if the variant is not yet fixed or exctinct.
        """
        if self.frequency == 0 or self.frequency == 2 * self._N:
            return False
        else:
            return True

    def is_extinct(self, minor=False, frozen=False):
        """
        Tells us if the mutant is extinct.

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
            frozen: (boolean) True if we are interested the frozen mutant.
        Returns:
            (boolean) If not minor or frozen, returns True if the mutant is extinct.
            If minor and frozen, returns True if the mutant that was minor at freeze
            my_time is now extinct.
        """
        if minor and frozen:
            if self.frozen_freq > self._N:
                if self.frequency == 2 * self._N:
                    return True
                else:
                    return False
        if self.frequency == 0:
            return True
        else:
            return False

    def is_fixed(self, minor=False, frozen=False):
        """
        Tells us if the mutant is fixed.

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
            frozen: (boolean) True if we are interested in the frozen mutant.
        Returns:
            (boolean) If not minor or frozen, returns True if the mutant is fixed.
            If minor and frozen, returns True if the mutant that was minor at freeze
            my_time is now fixed.
        """
        if minor and frozen:
            if self.frozen_freq > self._N:
                if self.frequency == 0:
                    return True
                else:
                    return False
        if self.frequency == 2 * self._N:
            return True
        else:
            return False

    def is_over_half(self, minor=False, frozen=False):
        """
        Tells us if the mutant has crossed half.

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
            frozen: (boolean) True if we are interested in the frozen mutant.
        Returns:
            (boolean) If not minor or frozen, returns True if the mutant is over a half.
            If minor and frozen, returns True if the mutant that was minor at freeze
            my_time is now over half.
        """
        if minor and frozen:
            if self.frozen_freq > self._N:
                if self.frequency <= self._N:
                    return True
                else:
                    return False
        if self.frequency >= self._N:
            return True
        else:
            return False


    def signed_freq(self, minor=False):
        """
        Returns the signed fraction of haplotypes with the mutation.

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
        Returns:
            (float) If not minor, the signed (according to the sign of the
             phenotypic effect) fraction of the 2*_N haplotypes with the mutation.
            Otherwise, the signed fraction of the 2*_N haplotypes with the minor mutation.
        """
        if not minor:
            return self.frequency * self._DENOMINATOR
        else:
            minor_freq = self._minor_frequency(self.frequency)
            if self.is_minor():
                return minor_freq * self._DENOMINATOR
            else:
                return -minor_freq* self._DENOMINATOR

    def signed_freq_relative_frozen(self, minor=False):
        """
        Returns the signed fraction of haplotypes with the mutation,
        relative to the frozen frequency_pos.

        Parameters:
            minor: (boolean) True if we are interested in the (frozen) minor mutant
        Returns:
            (float) If not minor, the signed (according to the sign of the
             phenotypic effect) fraction of the 2*_N haplotypes with the mutation.
            Otherwise, the signed fraction of the 2*_N haplotypes with the mutation
            that was minor at freeze my_time.
        """
        if not minor:
            return self.signed_freq(minor=False)
        else:
            if self.frozen_freq < self._N:
                return self.signed_freq(minor=False)
            else:
                x = 1 - self.frequency * self._DENOMINATOR


    def signed_freq_prev(self, minor = False,frozen = False):
        """
        Returns the signed fraction of haplotypes with the mutation in the
        previous generation or at freeze time, if frozen is True.
        .

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
            frozen: (boolean) True if we are interested in freq at freeze time.
        Returns:
            (float) If not minor or frozen, the signed (according to the sign of the
             phenotypic effect) fraction of the 2*_N haplotypes with the mutation
            in the previous generation.
            Otherwise, if minor and frozen, the signed fraction of the 2*_N
            haplotypes with the mutation that was minor at freeze my_time.
        """
        if not frozen:
            if not minor:
                return self.prev_freq*self._DENOMINATOR
            else:
                if self.is_minor():
                    return self._previous_frequency_of_current_minor()*self._DENOMINATOR
                else:
                    return -self._previous_frequency_of_current_minor()*self._DENOMINATOR
        else:
            ffreq = self.frozen_freq
            sign = 1
            if minor:
                if ffreq > self._N:
                    ffreq = 2*self._N - ffreq
                    sign = -sign

            if sign > 0:
                return ffreq * self._DENOMINATOR
            else:
                return -ffreq * self._DENOMINATOR


    def delta_x(self, minor=False, frozen=False):
        """
        Returns the change in frequency_pos since the previous generation, or freeze
        time if frozen is True.

         Parameters:
            minor: (boolean) True if we are interested in the previous or frozen minor
            mutant
            frozen: (boolean) True if we are interested in change since freeze my_time.
        Returns:
            (float) If not minor, the change in fraction of the 2*_N haplotypes with the mutation.
            If frozen, the change in fraction of the 2*_N haplotypes with the mutation
            since freeze my_time. If not frozen, then since the previous generation.
            If minor and not frozen, returns the change in fraction of the 2*_N haplotypes
            with the minor (in previous generation) mutation.
            If minor and frozen, returns the change in fraction of the 2*_N haplotypes
            with the minor (at freeze my_time) mutation.
        """

        if not frozen:
            if not minor:
                return (self.frequency - self.prev_freq) * self._DENOMINATOR
            else:
                return (self._minor_frequency(self.frequency) - self._previous_frequency_of_current_minor()) * self._DENOMINATOR
        else:
            if not minor:
                return (self.frequency - self.frozen_freq) * self._DENOMINATOR
            else:
                if self.frozen_freq <= self._N:
                    return (self.frequency - self.frozen_freq) * self._DENOMINATOR
                else:
                    return -(self.frequency - self.frozen_freq) * self._DENOMINATOR

class Mutation(_MutationBasic):
    """
    Container class for mutations.

    Parameters:
        N: (int) The size of the population in which the mutation appears
        scaled_size: (float) The scaled size of the mutation
        pheno_size: (float) The phenotypic effect size of the mutation.
        derived_sign: (int) The sign of the derived allele's effect on the trait (= plus/minus 1)

    Attributes:
        _frequency_class: (class object) A class that records all the frequency information of the mutation
        DERIVED_SIGN: (float) The sign of the derived allele's effect on the trait (= plus/minus 1)
    """

    def __init__(self, scaled_size, pheno_size_homozyg, derived_sign, N=10000):

        _MutationBasic.__init__(self,scaled_size, pheno_size_homozyg, N)

        #super(self.__class__, self).__init__(scaled_size, pheno_size_homozyg, N)

        self.DERIVED_SIGN = np.sign(derived_sign)
        # the class that stores the mutation's frequency information
        self._frequency_class = _MutationFrequencies(N=N)

    def derived_sign(self):
        return self.DERIVED_SIGN

    def delta_x(self, minor=False, frozen=False):
        """
        Returns the change in frequency since the previous generation, or since the freeze
        if frozen is True.

         Parameters:
            minor: (boolean) True if we are interested in the previous or frozen minor
            mutant
            frozen: (boolean) True if we are interested in change since freeze my_time.
        Returns:
            (float) If not minor, the change in fraction of the 2*_N haplotypes with the mutation.
            If frozen, the change in fraction of the 2*_N haplotypes with the mutation
            since freeze my_time. If not frozen, then since the previous generation.
            If minor and not frozen, returns the change in fraction of the 2*_N haplotypes
            with the minor (in previous generation) mutation.
            If minor and frozen, returns the change in fraction of the 2*_N haplotypes
            with the minor (at freeze my_time) mutation.
        """
        return self._frequency_class.delta_x(minor=minor, frozen=frozen)

    def contrib_to_mean_derived(self):
        """
        Returns the signed contribution to the mean phenotype of the derived allele.
        Still needs to be multiplied with unit vector magnitude
        """
        return self._contrib_to_mean(self._frequency_class.frequency) * self.DERIVED_SIGN

    def delta_contrib_to_mean(self, frozen=False):
        """
        Returns the change in contribution to the mean phenotype since the previous generation.
        Still needs to be multiplied with unit vector magnitude
        or freeze time if frozen is True.
        """
        return self.delta_x(minor=False, frozen=frozen) * self.PHENO_SIZE_HOMOZYG * self.DERIVED_SIGN

    def delta_central_moment(self, nth=2, frozen=False):
        """
        Returns the change in contribution to the nth central moment since the previous generation,
        (or freeze my_time if frozen is True) if nth <= 4
        Still needs to be multiplied with nth power of unit vector magnitude.
        """
        frequency = self._frequency_class.frequency
        if not frozen:
            prev_frequency = self._frequency_class.prev_freq
        else:
            prev_frequency = self._frequency_class.frozen_freq
        return self._delta_central_moment(prev_frequency, frequency, nth=nth)

    def update_freq(self, frequency, update=None):
        """
        Updates the frequency of the mutation.

        Updates the mutation's _frequency_class attributes: frequency, minor_status, prev_minor_status
        and maybe lifetime if update is True

        Parameters:
            frequency: The new frequency of the mutation
            update: (boolean) If true, adds a generation to the mutants total life
        """
        if update is None:
            update = True
        self._frequency_class.update_freq(frequency, update)

    def freeze(self):
        self._frequency_class.freeze()

    def central_moment(self, nth=2):
        """
        Returns the current contribution of the mutant to
        the nth central moment in the phenotype distribution.
        if nth <= 4
        Still needs to be multiplied with nth power of unit vector magnitude.
        """
        frequency = self._frequency_class.frequency
        return self._central_moment(frequency, nth=nth)

    def cumulant(self, nth=4):
        """
        Returns the current contribution of the mutant to
        the nth cumulant in the phenotype distribution.
        if nth <= 6
        Still needs to be multiplied with nth power of unit vector magnitude.
        """
        frequency = self._frequency_class.frequency
        return self._cumulant(frequency, nth=nth)

    def x(self, minor=False):
        """
        Returns the fraction haplotypes with the mutation.

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
        Returns:
            (float) If not minor, the fraction of the 2*_N haplotypes with the mutation.
            Otherwise, the fraction of the 2*_N haplotypes with the minor mutation.
        """
        return self._frequency_class.x(minor)

    def x_prev(self, minor=False, frozen=False):
        """
        Returns the fraction haplotypes with the mutation in the previous generation
        or at freeze time, if frozen is True.

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
            frozen: (boolean) True if we are interested in freq at freeze my_time.
        Returns:
            (float) If not minor, the fraction of the 2*_N haplotypes with the mutation
            in the previous generation.
            Otherwise, the fraction of the 2*_N haplotypes with the (currently) minor
            mutation in the previous generation.
        """
        x_prev = self._frequency_class.x_prev(minor=minor, frozen=frozen)
        return x_prev

    def x_rel_frozen(self, minor=False):
        """
        Returns the fraction haplotypes with the mutation, relative to the frozen frequency.

        Parameters:
            minor: (boolean) True if we are interested in the (frozen) minor mutant
        Returns:
            (float) If not minor, the fraction of the 2*_N haplotypes with the mutation.
            Otherwise, the fraction of the 2*_N haplotypes mutation that was minor at
            freeze time.
        """
        x_rel_frozen = self._frequency_class.x_rel_frozen(minor=minor)
        return x_rel_frozen

    def is_minor(self):
        """
        Tells us if the variant is minor. Returns boolean
        """
        return self._frequency_class.is_minor()

    def is_segregating(self):
        """
        Tells us if the variant is not yet fixed or exctinct. Returns boolean
        """
        return self._frequency_class.is_segregating()

    def is_extinct(self, minor=False, frozen=False):
        """
        Tells us if the mutant is extinct. Returns boolean

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
            frozen: (boolean) True if we are interested the frozen mutant.
        Returns:
            (boolean) If not minor or frozen, returns True if the mutant is extinct.
            If minor and frozen, returns True if the mutant that was minor at freeze
            my_time is now extinct.
        """
        return self._frequency_class.is_extinct(minor=minor, frozen=frozen)

    def is_fixed(self, minor=False, frozen=False):
        """
        Tells us if the mutant is fixed. Returns boolean

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
            frozen: (boolean) True if we are interested in the frozen mutant.
        Returns:
            (boolean) If not minor or frozen, returns True if the mutant is fixed.
            If minor and frozen, returns True if the mutant that was minor at freeze
            my_time is now fixed.
        """
        return self._frequency_class.is_fixed(minor=minor, frozen=frozen)

    def is_over_half(self, minor=False, frozen=False):
        """
        Tells us if the mutant has crossed half. Returns boolean

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
            frozen: (boolean) True if we are interested in the frozen mutant.
        Returns:
            (boolean) If not minor or frozen, returns True if the mutant is over a half.
            If minor and frozen, returns True if the mutant that was minor at freeze
            my_time is now over half.
        """
        return self._frequency_class.is_over_half(minor=minor, frozen=frozen)

    def fixing_effect(self):
        """
        Returns the phenotypic effect of the derived mutant fixing.
        Still needs to be multiplied with unit vector.
        """
        return self.PHENO_SIZE_HOMOZYG

    def segregating_effect(self):
        """
        Returns the contribution of the mutant to the average phenotype.
        Still needs to be multiplied with unit vector
        """
        x = self._frequency_class.x(minor=False)
        return x * self.PHENO_SIZE_HOMOZYG

    def fixed_state_effect(self):
        """
        Returns the contribution of the mutant to fixed state. Still needs to be multiplied with unit vector

        (The fixed state is what the mean phenotype would be if all minor mutants were
        extinct.)
        Still needs to be multiplied with unit vector.
        """
        minor = self._frequency_class.is_minor()
        if not minor:
            return self.PHENO_SIZE_HOMOZYG
        else:
            return 0.0

    def signed_pheno_size_het(self, minor=False, scaled=True):
        """
        Returns the effect size on a het of the mutant

         Parameters:
            minor: (boolean) True if we are interested in the minor mutant
            scaled: (boolean) True if we want the scaled effect size.
        Returns:
            (float) If not minor, returns the (scaled, if scaled is True)
             effect size of the mutant.
            If minor, returns the effect size of the minor mutant.
        """
        mysign_freq = np.sign(self.signed_freq(minor=minor))

        if scaled:
            return self.SCALED_SIZE * mysign_freq
        else:
            return self.PHENO_SIZE_HOMOZYG * mysign_freq / 2.0

    def signed_pheno_size_het_prev(self, minor=False, frozen=False, scaled=True):
        """
        Returns the effect size of the mutant in the previous generation
        or at freeze time.

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
            scaled: (boolean) True if we want the scaled effect size.
            frozen: (boolean) True if we are interested in the frozen mutant.
        Returns:
            (float)  If not minor and not frozen, returns the (scaled, if scaled
            is True) effect size of the mutant.
            If minor and frozen, returns the effect size of the mutant that was minor
            at freeze time.
            If minor and not frozen, returns the effect size of the mutant that
            was minor in the previous generation.
        """
        if frozen:
            mysign_freq_effect = np.sign(self.signed_freq_prev(minor=minor, frozen=frozen))
        else:
            mysign_freq_effect = np.sign(self.signed_freq_prev(minor=minor))

        if scaled:
            return self.SCALED_SIZE * mysign_freq_effect
        else:
            return self.PHENO_SIZE_HOMOZYG * mysign_freq_effect / 2.0

    def signed_freq(self, minor=False):
        """
        Returns the signed fraction of haplotypes with the mutation.

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
        Returns:
            (float) If not minor, the signed (according to the sign of the
             phenotypic effect) fraction of the 2*_N haplotypes with the mutation.
            Otherwise, the signed fraction of the 2*_N haplotypes with the minor mutation.
        """
        return self._frequency_class.signed_freq(minor=minor) * self.DERIVED_SIGN

    def signed_freq_relative_frozen(self, minor=False):
        """
        Returns the signed fraction of haplotypes with the mutation,
        relative to the frozen frequency_pos.

        Parameters:
            minor: (boolean) True if we are interested in the (frozen) minor mutant
        Returns:
            (float) If not minor, the signed (according to the sign of the
             phenotypic effect) fraction of the 2*_N haplotypes with the mutation.
            Otherwise, the signed fraction of the 2*_N haplotypes with the mutation
            that was minor at freeze my_time.
        """
        return self._frequency_class.signed_freq_relative_frozen(minor=minor) * self.DERIVED_SIGN

    def signed_freq_prev(self, minor=False, frozen=False):
        """
        Returns the signed fraction of haplotypes with the mutation in the
        previous generation or at freeze my_time, if frozen is True.
        .

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
            frozen: (boolean) True if we are interested in freq at freeze my_time.
        Returns:
            (float) If not minor or frozen, the signed (according to the sign of the
             phenotypic effect) fraction of the 2*_N haplotypes with the mutation
            in the previous generation.
            Otherwise, if minor and frozen, the signed fraction of the 2*_N
            haplotypes with the mutation that was minor at freeze my_time.
        """
        return self._frequency_class.signed_freq_prev(minor=minor, frozen=frozen) * self.DERIVED_SIGN

class MutationPosNeg(_MutationBasic):
    """
    Container class for a pair of mutations with same trait effect, except that one is aligned to the trait of interest. The other opposing.

    Parameters:
        N: (int) The size of the population in which the mutation appears
        scaled_size: (float) The scaled size of the mutation
        pheno_size: (float) The phenotypic effect size of the mutation.

    Attributes:
        _frequency_class_pos: (class object) A class that records all the frequency information of the aligned mutation
        _frequency_class_neg: (class object) A class that records all the frequency information of the opposing mutation
        DERIVED_SIGN: (float) The sign of the derived allele's effect on trait 1 (= plus/minus 1)
    """

    def __init__(self, scaled_size, pheno_size_homozyg, N=10000):

        _MutationBasic.__init__(self,scaled_size, pheno_size_homozyg, N)

        # the class that stores the aligned mutation's frequency information
        self._frequency_class_pos = _MutationFrequencies(N=N)
        # the class that stores the opposing mutation's frequency information
        self._frequency_class_neg = _MutationFrequencies(N=N)

        # The frequency that the two mutations start at
        self.initial_freq = 1

        self.weight = 1.0

    def x_initial(self):
        return self._x(self.initial_freq,minor=False)

    def lifetime(self,pos=True):
        return self.frequency_class(pos=pos).lifetime

    def frequency_class(self,pos=True):
        if pos:
            return self._frequency_class_pos
        else:
            return self._frequency_class_neg

    @staticmethod
    def sign(pos=True):
        if pos:
            return 1
        else:
            return -1

    def delta_x(self, pos=True):
        """
        Returns the change in frequency of the aligned allele since the previous generation.

         Parameters:
            pos: (boolean) True if we are interested in the previous aligned mutant
        Returns:
            (float) The change in fraction of the 2*_N haplotypes with the mutation.
            If pos gives the change in haplos with aligned mutation.
            Otherwise gives change in haplos with opposing mutation
        """

        return self.frequency_class(pos).delta_x(minor=False, frozen=False)

    def contrib_to_mean_derived(self,pos=True):
        """
        Returns the signed contribution to the mean phenotype of the derived allele.
        Still needs to be multiplied with unit vector magnitude

        Parameters:
            pos: (boolean) True if we are interested in the previous aligned mutant
        """
        return self._contrib_to_mean(self.frequency_class(pos).frequency) * self.sign(pos)

    def delta_contrib_to_mean(self, pos=True):
        """
        Returns the change in contribution to the mean phenotype since the previous generation.
        Still needs to be multiplied with unit vector magnitude
        or freeze time if frozen is True.
        """
        return self.delta_x(pos=pos) * self.PHENO_SIZE_HOMOZYG * self.sign(pos)

    def delta_central_moment(self, nth=2, pos=True):
        """
        Returns the change in contribution to the nth central moment since the previous generation,
        (or freeze my_time if frozen is True) if nth <= 4
        Still needs to be multiplied with nth power of unit vector magnitude.
        """
        frequency = self.frequency_class(pos).frequency
        prev_frequency = self.frequency_class(pos).prev_freq

        return self._delta_central_moment(prev_frequency, frequency, nth=nth)

    def update_freq(self, frequency_pos=None,frequency_neg=None, update=None):
        """
        Updates the frequency of the aligned and/or opposing mutation.

        Updates the mutation's _frequency_class attributes: frequency, minor_status, prev_minor_status
        and maybe lifetime if update is True

        Parameters:
            frequency_pos: The new frequency of the aligned mutation
            frequency_neg: The new frequency of the opposing mutation
            update: (boolean) If true, adds a generation to the mutants total life
        """
        if update is None:
            update = True
        if frequency_pos is not None:
            self.frequency_class(True).update_freq(frequency_pos, update)
        if frequency_neg is not None:
            self.frequency_class(False).update_freq(frequency_neg, update)

    def freeze(self):
        self._frequency_class_pos.freeze()
        self._frequency_class_neg.freeze()

    def central_moment(self, nth=2,pos=True,both=False):
        """
        Returns the current contribution of the mutant to
        the nth central moment in the phenotype distribution.
        if nth <= 4
        Still needs to be multiplied with nth power of unit vector magnitude.
        """
        if not both:
            frequency = self.frequency_class(pos).frequency
            return self._central_moment(frequency, nth=nth)*self.sign(pos)**nth
        else:
            return self.central_moment(nth=nth,pos=True,both=False) + self.central_moment(nth=nth,pos=False,both=False)

    def cumulant(self, nth=4,pos=True,both=False):
        """
        Returns the current contribution of the mutant to
        the nth cumulant in the phenotype distribution.
        if nth <= 6
        Still needs to be multiplied with magnitude of nth power of unit vector magnitude.
        """
        if not both:
            frequency = self.frequency_class(pos).frequency
            return self._cumulant(frequency, nth=nth)*self.sign(pos)
        else:
            return self.cumulant(nth=nth,pos=True,both=False) + self.cumulant(nth=nth,pos=False,both=False)


    def x(self, pos=True):
        """
        Returns the fraction haplotypes with the mutation.

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
        Returns:
            (float) If not minor, the fraction of the 2*_N haplotypes with the mutation.
            Otherwise, the fraction of the 2*_N haplotypes with the minor mutation.
        """
        return self.frequency_class(pos).x(minor=False)

    def x_prev(self, pos=True):
        """
        Returns the fraction haplotypes with the mutation in the previous generation
        or at freeze time, if frozen is True.

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
            frozen: (boolean) True if we are interested in freq at freeze my_time.
        Returns:
            (float) If not minor, the fraction of the 2*_N haplotypes with the mutation
            in the previous generation.
            Otherwise, the fraction of the 2*_N haplotypes with the (currently) minor
            mutation in the previous generation.
        """
        x_prev = self.frequency_class(pos).x_prev(minor=False, frozen=False)
        return x_prev

    def x_change(self,pos=True,both=False):
        """
        Returns the fraction haplotypes with the mutation, relative to the frozen frequency.

        Parameters:
            minor: (boolean) True if we are interested in the (frozen) minor mutant
        Returns:
            (float) If not minor, the fraction of the 2*_N haplotypes with the mutation.
            Otherwise, the fraction of the 2*_N haplotypes mutation that was minor at
            freeze time.
        """
        if not both:
            prev_freq = self.initial_freq
            freq = self.frequency_class(pos).frequency
        else:
            prev_freq = self.frequency_class(False).frequency
            freq = self.frequency_class(True).frequency

        return self._delta_x(prev_frequency=prev_freq,frequency=freq)

    def is_minor(self,pos=True):
        """
        Tells us if the variant is minor. Returns boolean
        """
        return self.frequency_class(pos).is_minor()

    def is_segregating(self,pos=True):
        """
        Tells us if the variant is not yet fixed or exctinct. Returns boolean
        """
        return self.frequency_class(pos).is_segregating()

    def either_segregating(self):
        if self.is_segregating(pos=True) or self.is_segregating(pos=False):
            return True
        else:
            return False

    def is_extinct(self, pos=True):
        """
        Tells us if the mutant is extinct. Returns boolean

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
            frozen: (boolean) True if we are interested the frozen mutant.
        Returns:
            (boolean) If not minor or frozen, returns True if the mutant is extinct.
            If minor and frozen, returns True if the mutant that was minor at freeze
            my_time is now extinct.
        """
        return self.frequency_class(pos).is_extinct(minor=False, frozen=False)

    def is_fixed(self, pos=True):
        """
        Tells us if the mutant is fixed. Returns boolean

        Parameters:
            minor: (boolean) True if we are interested in the minor mutant
            frozen: (boolean) True if we are interested in the frozen mutant.
        Returns:
            (boolean) If not minor or frozen, returns True if the mutant is fixed.
            If minor and frozen, returns True if the mutant that was minor at freeze
            my_time is now fixed.
        """
        return self.frequency_class(pos).is_fixed(minor=False, frozen=False)

    def is_over_half(self,pos=True):
        """
        Tells us if the mutant has crossed half. Returns boolean

        Parameters:
            pos: (boolean) True if we are interested in the aligned mutant
            frozen: (boolean) True if we are interested in the frozen mutant.
        Returns:
            (boolean) If not minor or frozen, returns True if the mutant is over a half.
            If minor and frozen, returns True if the mutant that was minor at freeze
            my_time is now over half.
        """
        return self.frequency_class(pos).is_over_half(minor=False, frozen=False)

    def fixing_effect(self):
        """
        Returns the phenotypic effect of the derived mutant fixing.
        Still needs to be multiplied with unit vector.
        """
        return self.PHENO_SIZE_HOMOZYG


    def fixed_state_effect(self,pos=None,both=None):
        """
        Returns the contribution of the mutant to fixed state. Still needs to be multiplied with unit vector

        (The fixed state is what the mean phenotype would be if all minor mutants were
        extinct.)
        Still needs to be multiplied with unit vector.
        """
        if not both:
            minor = self.frequency_class(pos).is_minor()
            if not minor:
                return self.PHENO_SIZE_HOMOZYG*self.sign(pos)
            else:
                return 0.0
        else:
            return self.fixed_state_effect(pos=True,both=False) + self.fixed_state_effect(pos=False,both=False)
