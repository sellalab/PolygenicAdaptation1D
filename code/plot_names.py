from collections import defaultdict

class PlotNameClass(object):
    def __init__(self, units=None):
        if units == 's0':
            self.units = r' (units $\sqrt{V_{A}(0)}$)'
            self.units_square = r' (units $V_{A}(0)$)'
            self.units_cube = r' (units $V^{3/2}_{A}(0)$)'
            self.units_four = r' (units $V^{2}_{A}(0)$)'
            self.units_five = r' (units $V^{5/2}_{A}(0)$)'
            self.units_six = r' (units $V^{3}_{A}(0)$)'
        else:
            self.units = r' (units $\delta$)'
            self.units_square = r' (units $\delta^2$)'  # = \frac{\omega^2}{2N}$)'
            self.units_cube = r' (units $\delta^3$)'
            self.units_four = r' (units $\delta^4$)'
            self.units_five = r' (units $\delta^5$)'
            self.units_six = r' (units $\delta^6$)'


        self._make_nice_names()

    def param_text(self, name=None, value=None, units=None, digits=None):
        if units is None:
            units_text = ''
            if name is None:
                if value is None:
                    return ''
        else:
            units_text = units
        if name is None:
            the_name = ''
        else:
            the_name = name
        if value is None:
            the_value = 0
        else:
            the_value = value
        if units is not None:
            if units == 's0' or units == 'sigma_0':
                units_text = r'$\sqrt{V_{A}(0)}$'
            elif units == 'delta' or units == 'del':
                units_text = r'$\delta$'
        else:
            if len(the_name) >= 4:
                if the_name[-4:] == '_del':
                    the_name = the_name[:-4]
                    units = 's0'
                    units_text = r'$\delta$'
            if len(the_name) >= 3:
                if the_name[-3:] == '_s0':
                    units = 'del'
                    the_name = the_name[:-3]
                    units_text = r'$V_{A}(0)$'

        if the_name == 'N':
            val_str = '{0:.0f}'.format(the_value)
            name_str = r'$N$'
        elif the_name == 'w':
            val_str = '{0:.0f}'.format(the_value)
            name_str = r'$\omega$'
        elif the_name == 'cD':
            val_str = '{0:.2f}'.format(the_value)
            name_str = 'Const distance'
        elif the_name == 'Esd':
            val_str = '{0:.2f}'.format(the_value)
            name_str = r'$E_s$'
        elif the_name == 'Esd':
            val_str = '{0:.0f}'.format(the_value)
            name_str = r'$E_{sd}$'
        elif the_name == 'Vs':
            val_str = '{0:.0f}'.format(the_value)
            name_str = r'$V_s$'
        elif the_name == 'U':
            val_str = '{0:.3f}'.format(the_value)
            name_str = r'$U$'
        elif the_name == 'shift' or the_name == 'Delta':
            val_str = '{0:.0f}'.format(the_value)
            name_str = r'$\Lambda$'
        elif the_name == 'sigma_0':
            val_str = '{0:.2f}'.format(the_value)
            name_str = r'$V_{A}(0)$'
        else:
            val_str = '{0:.3f}'.format(the_value)
            name_str = the_name

        if digits is not None:
            if digits == 0:
                val_str = '{0:.0f}'.format(the_value)
            elif digits == 1:
                val_str = '{0:.1f}'.format(the_value)
            elif digits == 2:
                val_str = '{0:.2f}'.format(the_value)
            elif digits == 3:
                val_str = '{0:.3f}'.format(the_value)
            elif digits == 4:
                val_str = '{0:.4f}'.format(the_value)
            else:
                val_str = str(the_value)

        if name is None:
            texti = val_str + units_text
        elif value is None:
            texti = name_str
            if units is not None:
                texti += '(units ' + units_text + ')'
        else:
            texti = name_str + r' $=$ ' + val_str + units_text

        return texti

    def yname(self,stat):
        if stat in self.names:
            return self.names[stat]
        else:
            return stat

    def theory_yname(self,stat):
        if stat in self.theory_names:
            return self.theory_names[stat]
        else:
            return stat

    def dist_lande_yname(self,stat):
        if stat in self.dist_lande_names:
            return self.dist_lande_names[stat]
        else:
            return stat


    def _make_nice_names(self):
        self.names = defaultdict()

        self.pos_neg_names = defaultdict()

        self.theory_names = defaultdict()
        self.theory_names['dist'] = "Lande's approximation"
        self.theory_names['opt_minus_mean_fixed_over_shift'] = r"$e^{-t/(2N)}$"
        self.theory_names['dist_sum_over_2N_scaled'] = r"$[V_{A}(0)/(\Lambda \cdot \delta)]\cdot \int_{0}^{t}D_{L}(t)dt/2N$"

        self.names['var'] = r'Variance in trait value' + self.units_square
        self.names['var_square'] = r'Variance in trait value squared' + self.units_four
        self.names['num_seg'] = 'Number of segregating variants'
        self.names['mean_fixed'] =r'Ancestral state' + self.units #r'Mean trait from fixed muts' + self.units
        self.names['mean_seg'] = r'Mean trait from segregating' + self.units
        self.names['fixed_state'] = r'Fixed state' + self.units

        self.names['skewness'] = r'Skewness'
        self.names['dist'] = r'Distance from the' + '\n' + r'optimum' + self.units
        self.names['dist_square'] = r'Square distance from ' + '\n' + r'the optimum' + self.units_square
        self.names['dist_guess'] = r'$\mu_3 (t)/(2V_{A} (t))$' + self.units

        self.names['opt_minus_fixed_state_over_shift'] = r'($\Lambda-$ consensus geno)/$\Lambda$'
        self.names['opt_minus_mean_fixed_over_shift'] = r'($\Lambda-$ fixed background)/$\Lambda$'

        for key in list(self.names.keys()):
            self.names['delta_' + key] = 'Change in ' + self.names[key].lower()

        self.names['d2ax_frozen_over_shift'] = "Contribution to change from\n segregating divided by shift"
        self.names['dist_sum_over_2N_scaled'] = r"$[V_{A}(0)/(\Lambda \cdot \delta)]\cdot \int_{0}^{t}D(t)dt/2N$"

        self.names['mean_fit'] = 'Mean fitness'
        self.names['std_fit'] = 'Standard deviation in fitness'

        self.names['diff_num_seg_frac'] = 'Fraction excess pos variants'
        self.names['diff_num_seg'] = 'Number pos var - number neg var'

        self.names['c4'] = 'Fourth cumulant' + self.units_square

        self.names['berry_esseen'] = r'$\sum |\mu_{3,i}|$' + self.units
        self.names['berry_esseen_skew'] = r'$\sum |\mu_{3,i}|/ V_{A}^{3/2} (0) (t)$'

        self.names['excess_kurtosis'] = 'Excess Kurtosis'


        self.names['frac_fixed_per_seg_var'] = "Average fixation probability"
        self.names['x_per_seg_var'] = "Average allele frequency"
        self.names['dx_per_seg_var'] = "Average change in allele frequency"

        ##NEW HBSTATA
        name = "Contribution to change in mean from standing variation\nper unit mutational input, and multiplied by " \
               + r"$V_{A} (0)/(\Lambda\cdot\delta)$"
        self.names['d2ax_scaled_per_mut_input'] = name
        self.names['2a_frac_fixed_scaled_per_mut_input'] = name

        ##NEW HBSTATA
        self.names['frozen_d2ax_scaled_per_mut_input'] = name

        self.names['frozen_nm_2ax_scaled_per_mut_input'] = "Contribution to change in mean from new muts\nper unit" \
                                                         " mutational input, " \
                                                         "and multiplied by " + r"$V_{A} (0)/(\Lambda\cdot\delta)$"
        self.names['frozen_nm_2ax_scaled_per_mut_input'] = "Contribution to change in mean from new muts\nper unit" \
                                                         " mutational input, " \
                                                         "and multiplied by " + r"$V_{A} (0)/(\Lambda\cdot\delta)$"
        self.names['frozen_nm_and_standing_d2ax_scaled_per_mut_input'] = "Contribution to change in mean from standing and new\nper " \
                                                         "unit mutational input, and multiplied by " + \
                                                                         r"$V_{A} (0)/(\Lambda\cdot\delta)$"
        self.names['frozen_numseg_per_mut_input'] = "Number of standing variants, still segregating\n" \
                                                    "per unit muational input"

        self.names['frozen_d2ax_over_shift'] = "Fraction contribution to change in mean from\nstanding variation"
        self.names['frozen_nm_2ax_over_shift'] = "Fraction contribution to change in mean from\nnew mutations"
        ##END NEW HBSTATS

        #NOW tStATS
        self.names['x_per_seg_var'] = 'Average allele frequency'

        self.names['numseg_efs_bins'] = 'Number of variants'
        self.names['numseg_fbins'] = 'Number of variants'

        self.names['frozen_mean_efs_fbins'] = r'Mean scaled effect size from frozen'
        self.names['frozen_numseg_efs_bins'] = 'Number of frozen variants'
        self.names['frozen_numseg_fbins'] = 'Number of frozen variants'
        self.names['frozen_var_efs_bins'] = r'Genetic variance from frozen' + self.units_square
        self.names['frozen_var_fbins'] = r'Genetic variance from frozen' + self.units_square
        self.names['frozen_mean_f_efs_bins'] = r'Average frequency_pos'
        self.names['frozen_extinct_efs_bins'] = 'Number of extinct'
        self.names['frozen_extinct_fbins'] = 'Number of extinct'
        self.names['frozen_fixed_efs_bins'] = 'Number of fixed'
        self.names['frozen_fixed_fbins'] = 'Number of fixed'

        self.names['frozen_contr_d_ax_efs_bins'] = r'$\Sigma 2a\Delta x$ since shift' + self.units
        self.names['frozen_contr_d_ax_fbins'] = r'$\Sigma 2a\Delta x$ since shift' + self.units
        self.names['frozen_contr_d_x_fbins'] = r'$\Sigma \Delta x$ since shift'

        self.names['delta_mean_freq'] = r'Mean $\Delta x$'
        self.names['mean_fixed_state'] = r'Mean fixed state contr' + self.units
        self.names['NUM_MUTANTS'] = 'Number mutants'

        self.names['extinction_time'] = 'Traj time, given extinction'
        self.names['var_freq'] = 'Var in freq'
        self.names['mean_freq'] = 'Mean freq'


        self.names['U'] = r'$U$'
        self.names['Es'] = r'$E[a^2]$'
        self.names['Vs'] = r'$V[a^2]$'
        self.names['Delta'] = r'$\Delta$'
        self.names['shift'] = r'$\Delta$'
        self.names['Vs'] = r'$V_{S}$'
        self.names['N'] = r'$N$'
        self.names['shift'] = r'$\Lambda$'
        self.names['shift_s0'] = r'$\Lambda$'
        self.names['sigma_0'] = r'$\sqrt{V_{A}(0)}$'
        self.names['sigma'] = r'$\sqrt{V_{A}}$'
        self.names['T_O'] = r'$T_O$'

        self.dist_lande_names = defaultdict()


