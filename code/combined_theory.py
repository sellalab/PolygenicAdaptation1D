import numpy as np
from scipy.integrate import quad
from scipy.special import dawsn, erfi,  erf, hyp2f1
from scipy.special import gamma as gamma_funct
from math import fabs
from scipy.stats import gamma, rv_continuous

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or fabs(value - array[idx-1]) < fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]


def find_index_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or fabs(value - array[idx-1]) < fabs(value - array[idx])):
        return idx-1
    else:
        return idx

class FreqMinorDistrE(rv_continuous):
    def __init__(self, S, N=10000, a=0., b=0.5):
        super(self.__class__, self).__init__(a=a,b=b)
        self._N = N
        self._SCAL = 1.0/float(self._N)
        self._DENOMINATOR = self._SCAL/2.0
        self._S = S
        self._a = a
        self._b = b
        if self._S < 1:
            self._low_lim = 200
            self._lim = 200
        else:
            self._low_lim = 100
            self._lim = 100
        self._NORM_CONST = self._tau_norm_const_efs_E2Ns()
        #print self._NORM_CONST


    def get_XI(self, percentile):
        try:
            fr = self.ppf([percentile])[0]
            frequency = 2 * self._N * fr
            frequency_one = int(round(frequency))
            if frequency > frequency_one and frequency_one != self._N:
                frequency_two = frequency_one+1
            elif frequency < frequency_one and frequency_one != 1:
                frequency_two = frequency_one-1
            else:
                frequency_two = frequency_one
                XI = frequency * self._DENOMINATOR
            if frequency_one != frequency_two:
                XI_one = frequency_one * self._DENOMINATOR
                XI_two = frequency_two* self._DENOMINATOR
                perc_one = self.cdf(XI_one)
                perc_two = self.cdf(XI_two)
                indi = find_index_nearest([perc_one,perc_two],percentile)
                if indi == 0:
                    XI = frequency_one * self._DENOMINATOR #round(frequency_pos * self._DENOMINATOR,self.round_XI)
                else:
                    XI = frequency_two * self._DENOMINATOR
        except ValueError:
            print('Value error in get XI percentile ' + str(percentile) +' for ' + str(self._S))
            XI = 0
        return XI

    def _cdf(self,x):
        return self._ff_to_x(x)/self._NORM_CONST


    def _ff(self):
        return self._ff_to_x(self._b)

    def _ff_to_x(self,x):
        if self._a < self._DENOMINATOR < x:
            int_1 =  quad(self._folded_tau,self._a,x,points=[self._DENOMINATOR],limit=self._low_lim)
        else:
            int_1 = quad(self._folded_tau, self._a, x, limit=self._low_lim)
        return int_1[0]


    def _tau_norm_const_efs_E2Ns(self):
        """returns normalisation constant for traj times tau multiplied
        with the gamma distribution of selection coefficients"""
        const = self._ff()
        return const


    def _folded_tau(self, x, S):
        S= self._S
        if x < self._DENOMINATOR:
            return 2 * (2 * self._N )* np.exp(-S * x * (1 - x)) /  (1 - x)
        else:
            return 2 *  np.exp(-S * x * (1 - x)) / (x * (1 - x))


class VarDistrMinor(rv_continuous):
    def __init__(self, S, N=10000, a=0., b=0.5):
        self._low_a = 1.0 / (4.0 * N)
        super(self.__class__, self).__init__(a=a, b=b)
        self._N = N
        self._SCAL = 1.0 / float(self._N)
        self._DENOMINATOR = self._SCAL / 2.0
        if S < 0:
            S = -S
        self._S = S
        self._rs = np.sqrt(self._S)
        self._a = a
        self._b = b
        self._NORM_CONST = self._norm_const()
        # print self._NORM_CONST


    def get_freq(self,percentile):
        try:
            fr = self.ppf([percentile])[0]
            frequency = 2 * self._N * fr
            frequency_one = int(round(frequency))
            if frequency > frequency_one and frequency_one != self._N:
                frequency_two = frequency_one + 1
            elif frequency < frequency_one and frequency_one != 1:
                frequency_two = frequency_one - 1
            else:
                frequency_two = frequency_one
                freqi = frequency_one
            if frequency_one != frequency_two:
                XI_one = frequency_one * self._DENOMINATOR
                XI_two = frequency_two * self._DENOMINATOR
                perc_one = self.cdf(XI_one)
                perc_two = self.cdf(XI_two)
                indi = find_index_nearest([perc_one, perc_two], percentile)
                if indi == 0:
                    freqi = frequency_one   # round(frequency_pos * self._DENOMINATOR,self.round_XI)
                else:
                    freqi= frequency_two
        except ValueError:
            print('Value error in get XI percentile ' + str(percentile) + ' for ' + str(self._S))
            freqi = 0
        return freqi

    def get_XI(self, percentile):
        try:
            fr = self.ppf([percentile])[0]
            frequency = 2 * self._N * fr
            frequency_one = int(round(frequency))
            if frequency > frequency_one and frequency_one != self._N:
                frequency_two = frequency_one + 1
            elif frequency < frequency_one and frequency_one != 1:
                frequency_two = frequency_one - 1
            else:
                frequency_two = frequency_one
                XI = frequency * self._DENOMINATOR
            if frequency_one != frequency_two:
                XI_one = frequency_one * self._DENOMINATOR
                XI_two = frequency_two * self._DENOMINATOR
                perc_one = self.cdf(XI_one)
                perc_two = self.cdf(XI_two)
                indi = find_index_nearest([perc_one, perc_two], percentile)
                if indi == 0:
                    XI = frequency_one * self._DENOMINATOR  # round(frequency_pos * self._DENOMINATOR,self.round_XI)
                else:
                    XI = frequency_two * self._DENOMINATOR
        except ValueError:
            print('Value error in get XI percentile ' + str(percentile) + ' for ' + str(self._S))
            XI = 0
        return XI


    def _my_pdf(self,x):
        multi = 4.0*self._S*np.exp(-self._S*x*(1-x))/float(self._NORM_CONST)
        if x < 0 or x > 0.5:
            return 0
        elif x < self._DENOMINATOR:
            return 2.0*self._N*x*multi
        else:
            return multi


    def _norm_const(self):
        return self._the_integral_variance_per_unit_mut_input(self._b)


    def _the_integral_variance_per_unit_mut_input(self,x):
        if x <= self._a:
            return 0
        if self._a < x< self._SCAL / 2.0:
            numeratori = 2.0*self._N*(2*(self._rs*dawsn(self._rs/2.0)-1+np.exp(-self._S*x*(1-x))*(1- self._rs*dawsn(self._rs/2.0*(1.0-2.0*x)))))
        else:
            if x > self._b:
                x= self._b
            integtilldenom = 2.0*self._N*(2*(self._rs*dawsn(self._rs/2.0)-1+np.exp(-self._S*self._DENOMINATOR*(1-self._DENOMINATOR))*(1- self._rs*dawsn(self._rs/2.0*(1.0-2.0*self._DENOMINATOR)))))
            numeratori = 2.0*self._rs*np.sqrt(np.pi)*np.exp(-self._S/4.0)*(erfi(self._rs/2.0*(1.0-self._SCAL))-erfi(self._rs/2.0*(1.0-2.0*x)))
            numeratori+=integtilldenom
        return numeratori


    def _cdf(self, x):
        denom = self._NORM_CONST
        numeratori = self._the_integral_variance_per_unit_mut_input(x)
        return numeratori/denom



def v_a(a):
    """the steady-state phenotypic variance contributed by alleles with magnitude of phenotypic effect a,
    per unit mutational input"""
    a = np.abs(a)
    return 4.0*a*dawsn(a/2.0)

def v_S(S):
    """the steady-state phenotypic variance contributed by alleles with scaled selection coefficient S,
    per unit mutational input"""
    a = np.sqrt(np.abs(S))
    return v_a(a)

def v_ax(a,x0,N=10000):
    """the steady-state phenotypic variance contributed by alleles with magnitude of phenotypic effect a,
    and frequncy x0, per unit mutational input"""
    a = np.abs(a)
    denom = 1.0/float(2*N)
    if x0 < denom :
        facti = 2.0 * denom  * x0
    else:
        facti = 1.0
    return 4.0 * facti * a**2 * np.exp(-a**2 * x0 * (1 - x0))

def v_Sx(S,x0,N=10000):
    """the steady-state phenotypic variance contributed by alleles with scaled selection coefficient S,
    per unit mutational input"""
    a = np.sqrt(np.abs(S))
    return v_Sx(a,x0,N)

def f_a(a):
    return 2*a**3*np.exp(-a**2/4.0)/(np.sqrt(np.pi)*erf(a/2))

def f_S(S):
    a = np.sqrt(np.abs(S))
    return f_a(a)

def v_x(self, x0,E2Ns, V2Ns=-1):
    """the steady-state phenotypic variance contributed by alleles with MAF x0,
    per unit mutational input (Assuming a GAMMA DISTRIBUTION of squared phenotypic effects with expected value
    E2Ns, and variance V2N2)"""
    if V2Ns < 0:
        V2Ns = E2Ns**2 # in this case an exponential distribuion of squared effects
    return 4*E2Ns*(1+V2Ns/E2Ns * x0 * (1 - x0))**(-(E2Ns**2/V2Ns+1))

def get_integral_v_a(E2Ns, V2Ns=-1):
    """integrate the function v(a) over the effect size distribution,
    assuming a gamma distribution of squared effects with expected value
    E2Ns, and variance V2N2"""
    if V2Ns < 0:
        V2Ns = E2Ns**2 # in this case an exponential distribuion of squared effects
    firsthypgeo = hyp2f1(1.0, E2Ns ** 2 / V2Ns + 1, -0.5, -V2Ns / (4 * E2Ns))
    secondhypgeo = hyp2f1(1.0, E2Ns ** 2 / V2Ns + 1, +0.5, -V2Ns / (4 * E2Ns))
    firstterm = (4 * E2Ns + V2Ns) * firsthypgeo
    secterm = 2 * (E2Ns * (2 + E2Ns) + 2 * V2Ns) * secondhypgeo
    myintegral = 2 * E2Ns / (2 * E2Ns ** 2 + V2Ns) * (firstterm - secterm)

    # SHAPE_S, SCALE_S = float(E2Ns) ** 2 / float(V2Ns), float(V2Ns) / float(E2Ns)
    # S_dist = gamma(SHAPE_S, loc=0., scale=SCALE_S)
    # to_integrate = lambda ss: 4.0 * np.sqrt(np.abs(ss)) * dawsn(np.sqrt(np.abs(ss)) / 2.0) * S_dist.pdf(ss)
    # b = S_dist.ppf(0.99999999999999)
    # myintegral = quad(to_integrate, 0, b)[0]
    return myintegral

def get_integral_f_a(E2Ns, V2Ns=-1):
    """integrate the function f(a) over the effect size distribution,
    assuming a gamma distribution of squared effects  with expected value
    E2Ns, and variance V2N2"""
    if V2Ns < 0:
        V2Ns = E2Ns**2 # in this case an exponential distribuion of squared effects
    SHAPE_S, SCALE_S = float(E2Ns) ** 2 / float(V2Ns), float(V2Ns) / float(E2Ns)
    S_dist = gamma(SHAPE_S, loc=0., scale=SCALE_S)
    to_integrate = lambda ss: 2*np.sqrt(np.abs(ss))**3*np.exp(-ss/4.0)/(np.sqrt(np.pi)*erf(np.sqrt(np.abs(ss))/2))* S_dist.pdf(ss)
    b = S_dist.ppf(0.99999999999999)
    myintegral = quad(to_integrate, 0, b)[0]
    return myintegral

def get_C(E2Ns, V2Ns=-1):
    """get the allelic measure of the deviation from/ Lande (amplification)
    assuming a gamma distribution of squared effects"""
    if V2Ns < 0:
        V2Ns = E2Ns**2 # exponential distribution of squared effects
    integral_of_v = get_integral_v_a(E2Ns, V2Ns)
    integral_of_f = get_integral_f_a(E2Ns, V2Ns)
    return integral_of_v/integral_of_f-1.0

def get_variance_units_little_del_square(E2Ns, V2Ns=-1, mut_input=100):
    """get the variance assuming a gamma distribution of squared effects"""
    if V2Ns < 0:
        V2Ns = E2Ns**2
    myintegral = get_integral_v_a(E2Ns, V2Ns)
    var_0_del_square = myintegral * mut_input
    return var_0_del_square

class TheoryCurves(object):
    def __init__(self, E2Ns=None, V2Ns=None, N=5000, U=0.005, Vs=-1., shift_s0=0, var_0=None, units=None):

        self._N = N
        if Vs < 0:
            Vs = float(2*N)
        self._rootVs = np.sqrt(Vs)

        if units is None:
            self._UNITS = self._rootVs
        else:
            self._UNITS = units

        self._SCAL = 1.0 / float(self._N)
        self._U = U

        self._E2Ns = E2Ns
        self._V2Ns = V2Ns

        self._Ea = np.sqrt( self._E2Ns / float(2*self._N)) * self._rootVs
        self._Va = np.sqrt(self._V2Ns / float(2*self._N)) * self._rootVs
        self._SHIFT_s0 = shift_s0
        self._MUT_INPUT = 2.0 * self._N * self._U
        self._CONST = 4.0 * self._rootVs ** 2 * self._U
        self._DELTA_UNIT = self._rootVs * np.sqrt(self._SCAL / 2.0)

        self._VAR_0 = self._get_variance_units_little_del_square() * self._DELTA_UNIT ** 2
        self._SIGMA_0 = np.sqrt(self._VAR_0)
        self._SHIFT = self._SIGMA_0 * self._SHIFT_s0
        self._Ea_s0 = self._Ea / self._SIGMA_0
        self._Va_s0 = self._Va / self._SIGMA_0

        if var_0 is None:
            self._VAR_0_EMP = self._VAR_0
        else:
            self._VAR_0_EMP = var_0

    def set_units(self,units):
        self._UNITS = units

    def dist_lande(self, t):
        exp_decay = self._VAR_0_EMP / float(self._rootVs ** 2)
        if t < 0:
            return 0
        else:
            ans = self._SHIFT * np.exp(-exp_decay * t)
            return ans / self._UNITS

    def integral_dist_lande_over_2N_scaled(self,t):
        """the integral of D_{L}(t)/(2N), divided by (shift*delta_unit)/V_{A}(0))"""
        exp_decay = self._VAR_0_EMP / float(self._rootVs ** 2)
        if t < 0:
            return 0
        else:
            return self._DELTA_UNIT/ self._UNITS * (1- np.exp(-exp_decay * t))


    def dist_mean_fixed_over_shift(self, t):
        """Distance of the fixed background from the optimum, divided
        by the size of the shift"""
        exp_decay = 1.0 / float(2*self._N)
        if t < 0:
            return 0
        else:
            return np.exp(-exp_decay * t)

    def log_var(self, var):
        var0 = self._VAR_0_EMP / self._UNITS ** 2
        var_diff = var - var0
        if var_diff <= 0:
            var_diff = 1.
        return 2 * self._N * np.log(var_diff)

    def log_fixed_state(self, fixed_state):
        shift = self._SHIFT / self._UNITS
        fs_diff = shift - fixed_state
        if fs_diff <= 0:
            fs_diff = 1.
        return np.log(fs_diff)

    def log_mean_fixed(self, mean_fixed):
        return self.log_fixed_state(mean_fixed)

    def log_var_new(self, var):
        var0 = self._VAR_0_EMP / self._UNITS ** 2
        var_diff = var - var0
        var_sum = var + var0
        if var_diff <= 0 or var_sum <= 0:
            var_diff = 1.
        else:
            var_diff = var_diff / var_sum
        return -self._rootVs ** 2 * log(var_diff) / (2.0 * self._VAR_0_EMP)

    def diff_delta_var_dmu3(self, delta_var, mu3):
        ans = 2 * self._N * (delta_var - mu3)
        return ans

    def diff_delta_var_dmu3_rat(self, delta_var, mu3, var):
        diff = self.diff_delta_var_dmu3(delta_var=delta_var, mu3=mu3)
        var0 = self._VAR_0_EMP / self._UNITS ** 2
        var_diff = var - var0
        if var_diff <= 0:
            return 0
        else:
            return diff / var_diff

    def dist_guess_2(self, t, var, dist_guess):
        var = var * self._UNITS ** 2
        exp_decay = var / float(self._rootVs ** 2)
        if t < 0:
            lands = 0
        else:
            lands = self._SHIFT * np.exp(-exp_decay * t) / self._UNITS
        return lands + dist_guess

    def var_phase_3(self, t, t_peak=None, var_peak=None):
        if var_peak is None:
            var_peak = self.var_peak
        if t_peak is None:
            t_peak = self.t_peak
        exp_decay = (3) * self._SCAL / float(
            2)  # (self._Es+1)*self._SCAL/float(2)#(4.0)/float(2*self._N)  for 80 , 3 for 20
        var0 = self._VAR_0_EMP
        time = t - t_peak
        if time < 0:
            return 0
        else:
            ans = var0 + (var_peak - var0) * np.exp(-exp_decay * time)
            return ans / self._UNITS

    def log_dist(self, dist):
        if dist <= 0.01 * self._DELTA_UNIT:
            dist = 0.01 * self._DELTA_UNIT
        shift = self._SHIFT / self._UNITS
        try:
            ld = np.log(dist)
        except ValueError:
            ld = np.log(0.01 * self._DELTA_UNIT)
        try:
            ls = np.log(shift)
        except ValueError:
            ls = np.log(0.01 * self._DELTA_UNIT)

        return ls - ld

    def logs_dist(self, dist):
        if dist <= 0.01 * self._DELTA_UNIT:
            dist = 0.01 * self._DELTA_UNIT
        shift = self._SHIFT / self._UNITS
        try:
            ld = np.log(dist)
        except ValueError:
            ld = np.log(0.01 * self._DELTA_UNIT)

        try:
            ls = log(shift)
        except ValueError:
            ls = log(0.01 * self._DELTA_UNIT)

        return -(ld - ls) * self._rootVs ** 2 / self._VAR_0_EMP

    def get_t_var_peak(self, time_list, var_list, scale=2):
        index_var_max = var_list.index(max(var_list))
        t_var_max = time_list[index_var_max]
        t_peak = t_var_max * scale
        t_peak = find_nearest(time_list, t_peak)
        t_peak_index = time_list.index(t_peak)
        var_peak = var_list[t_peak_index]
        self.t_peak = t_peak
        self.var_peak = var_peak
        return t_peak, var_peak, t_peak_index

    def _get_variance_units_little_del_square(self):
        """
        Returns steady-state phenotypic variance, in units little delta squared
        """
        SHAPE_S, SCALE_S = float(self._E2Ns) ** 2 / float(self._V2Ns), float(self._V2Ns) / float(self._E2Ns)
        S_dist = gamma(SHAPE_S, loc=0., scale=SCALE_S)
        to_integrate = lambda ss: 4.0 * np.sqrt(np.abs(ss)) * dawsn(np.sqrt(np.abs(ss)) / 2.0) * S_dist.pdf(ss)
        b = S_dist.ppf(0.99999999999999)
        myintegral = quad(to_integrate, 0, b)[0]
        var_0_del_square = myintegral * self._MUT_INPUT
        return var_0_del_square

