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
                    units_text = r'$\sigma_0$'

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
        elif the_name == 'f1':
            val_str = '{0:.2f}'.format(the_value)
            name_str = 'Frac in 1st uniform'
        elif the_name == 'thresh':
            val_str = '{0:.0f}'.format(the_value*100)
            name_str = 'Threshold percentile'
        elif the_name == 's1':
            val_str = '{0:.0f}'.format(the_value)
            name_str = 'Peak 1'
        elif the_name == 's2':
            val_str = '{0:.0f}'.format(the_value)
            name_str = 'Peak 2'
        elif the_name == 'sigma_0':
            val_str = '{0:.2f}'.format(the_value)
            name_str = r'$\sigma_0$'
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
        #still need to do this
        # pos_neg_add = []
        # pn_units = []
        # pn_units_square = []
        # pn_units_cube = []
        #
        # self.pos_neg_names['mean_freq']

        self.theory_names = defaultdict()
        # self.theory_names['dist'] = r'lande: $\Delta e^{-\sigma_0^2 t/\omega^2}$'
        self.theory_names['dist'] = "Lande's approximation"
        self.theory_names['logsdist'] = r'lande: $y=t$'
        self.theory_names['logdist'] = r'$C+\frac{t}{2N}$'
        # self.theory_names['dist_guess_2'] = r'$\Delta e^{-\int^t_0\sigma^2 (\tau)d\tau/\omega^2 } + \mu_{3} (t)/(2\sigma^2 (t))$'


        self.names['var'] = r'Variance in trait value' + self.units_square
        self.names['var_square'] = r'Variance in trait value squared' + self.units_four
        self.names['num_seg'] = 'Number of segregating variants'
        self.names['mean'] = r'Trait mean' + self.units
        self.names['mean_fixed'] =r'Ancestral state' + self.units #r'Mean trait from fixed muts' + self.units
        self.names['mean_seg'] = r'Mean trait from segregating' + self.units
        self.names['fixed_state'] = r'Fixed state' + self.units
        self.names['mu3'] = r'The third moment of the trait' + '\n' + r'distribution' + self.units
        self.names['mu4'] = r'The fourth moment of the trait distribution' + self.units_four
        self.names['mu5'] = r'The fifth moment of the trait distribution' + self.units_five
        self.names['kappa'] = r'$\kappa$' + self.units
        self.names['skewness'] = r'Skewness'
        self.names['mu3_cube'] = r'Cube root of third moment of the trait distribution' + self.units
        self.names['dist'] = r'Distance from the' + '\n' + r'optimum' + self.units
        self.names['dist_square'] = r'Square distance from ' + '\n' + r'the optimum' + self.units_square
        self.names['dist_guess'] = r'$\mu_3 (t)/(2\sigma^2 (t))$' + self.units
        self.names['dvar'] = r'$D(t)\sigma^2 (t)$' + self.units
        self.names['dmu3'] = r'$D(t)\mu_3 (t)/\omega^2$' + self.units_square
        self.names['delta_var_minus_dmu3'] = r'$\Delta \sigma^2 (t)- D(t)\mu_3 (t)/\omega^2$' + self.units_square
        self.names['dvar_halfmu3'] = r'$-D(t)\sigma^2 (t)+ \mu_3 (t)/2$' + self.units
        self.names['halfmu3'] = r'$\mu_3 (t)/2$' + self.units
        self.names['dsig_guess'] = r'$D(t)\mu_3 (t)/\omega^2 - (2E[S]+1)(\sigma^2(t) -\sigma_0^2)/(2N)$' + self.units_square
        self.names['dvar_guess'] = r'$D(t)\mu_3 (t)/\omega^2 - (2E[S]+1)(\sigma^2(t) -\sigma_0^2)/(2N)$' + self.units_square
        self.names['mu3_death'] = r'$\mu_3 (t)$ from muts disappearing' + self.units_cube
        self.names['mu3_fix'] = r'$\mu_3 (t)$ from muts fixing' + self.units_cube
        self.names['diff_delta_var_dmu3'] = r'$2N(\Delta \sigma^2(t) - D(t)\mu_3 (t)/\omega^2)$' + self.units_square

        self.names['opt_minus_fixed_state_over_shift'] = r'($\Lambda-$ consensus geno)/$\Lambda$'

        self.names['sum_si_xi'] = r'$\sum_i S_i (1-4x_i)/(2N)$'
        self.names['sum_si_xi_square'] = r'$\sum_i S_i (1-2x_i)^2/(2N)$'
        self.names['sum_si'] = r'$\sum_i S_i/(2N) $'
        self.names['sum_si_xi_vari'] = r'$\sum_i S_i (1-4x_i)\sigma_i^2 $' + self.units_square
        self.names['sum_si_vari'] = r'$\sum_i S_i \sigma_i^2 $' + self.units_square
        self.names['sum_si_xi_square_vari'] = r'$\sum_i S_i (1-2x_i)^2\sigma_i^2 $' + self.units_square
        self.names['sum_s_var'] = r'$E[S] \sigma^2 (t)$' + self.units_square
        self.names['sum_s_var_low'] = r'$(V[S}+E[S]^2)/(3E[S]) \sigma^2 (t)$' + self.units_square
        self.names['sum_s_var_high'] = r'$(E[S]-5) \sigma^2 (t)$' + self.units_square

        self.names['mean_freq_over_shift'] = 'Mean freq / shift'
        self.names['mean_var_over_shift'] = 'Mean variance / shift'
        self.names['mean_fixed_state_over_shift'] = 'Mean fixed state / shift'
        self.names['var_freq_over_shift'] = 'Var in freq / shift'
        self.names['skewness_freq_over_shift'] = 'Skewness in freq / shift'
        self.names['mean_ax_over_shift'] = r'Mean $ax$ / shift'
        self.names['freq_det'] = r'Deterministic $E[x]$'
        self.names['kurtosis_freq'] = 'Kurtosis in freq'
        self.names['var_freq_det'] = 'Var in det freq'
        self.names['kurtosis_freq_det'] = 'Kurtosis in det freq'
        self.names['skewness_freq_det'] = 'Skewness in det freq'
        self.names['mean_freq_det'] = 'Mean det freq'

        for key in list(self.names.keys()):
            self.names['delta_' + key] = 'Change in ' + self.names[key].lower()

        self.names['mean_fit'] = 'Mean fitness'
        self.names['std_fit'] = 'Standard deviation in fitness'
        self.names['r_2'] = r'$r^{2}$'
        self.names['r_2_std'] = r'Standard deviation in $r^2$'
        self.names['LD_std'] = 'Standard deviation in LD'
        self.names['LD_quantiles'] = 'LD'
        self.names['r_2_quantiles'] = r'$r^2$'
        self.names['LD'] = 'LD'

        self.names['diff_num_seg_frac'] = 'Fraction excess pos variants'
        self.names['diff_num_seg'] = 'Number pos var - number neg var'

        self.names['kappa_neg_mean'] = r'Average $a(\frac{1}{2} -x)$ from variants with minor allele $-$' + self.units
        self.names['kappa_pos_mean'] = r'Average $a(\frac{1}{2} -x)$ from variants with minor allele $+$' + self.units
        self.names['kappa_neg'] = r'$\Sigma a(\frac{1}{2} -x)$ from variants with minor allele $-$' + self.units
        self.names['kappa_pos'] = r'$\Sigma a(\frac{1}{2} -x)$ from variants with minor allele $+$ ' + self.units
        self.names['kappa_neg_frac'] = r'Fraction $\Sigma a(\frac{1}{2} -x)$ from variants' + '\n' + 'with minor allele $-$'
        self.names[
            'kappa_pos_frac'] = r'Fraction $\Sigma a(\frac{1}{2} -x)$ from variants' + '\n' + ' with minor allele $+$'
        self.names['kappa_neg_std'] = r'Std in $a(\frac{1}{2} -x)$ from variants with minor allele $-$' + self.units
        self.names['kappa_pos_std'] = r'Std in $a(\frac{1}{2} -x)$ from variants with minor allele $+$' + self.units
        self.names['full_over_halfmu3'] = r'$|-D(t)\sigma^2 (t)+ \mu_3 (t)/2|/(\mu_3 (t)/2)$'
        self.names['mu3_diff'] = r'$\mu^{pos}_3 (t) - \mu^{neg}_3 (t)$' + self.units_cube
        self.names['mu3_abs'] = r'$|\mu|_{3} (t)$ of the trait distribution' + self.units_cube
        self.names['mu3_pos_frac'] = r'Frac. contrib. to $|\mu|_{3} (t)$ from var. with $+$ cont'
        self.names['mu3_neg_frac'] = r'Frac. contrib. to $|\mu|_{3} (t)$ from var. with $-$ cont'
        self.names['mu3_diff_mean'] = r'Mean $\mu^{pos}_{3,i} (t) - \mu^{neg}_{3,i} (t)$' + self.units_cube
        self.names['mu3_abs_mean'] = r'Mean absolute third moment of the trait distribution' + self.units_cube
        self.names['mu3_pos_mean_frac'] = r'Frac. avg contrib to absolute third moment from variants with $+$ cont'
        self.names['mu3_neg_mean_frac'] = r'Frac. avg contrib to absolute third moment from variants with $-$ cont'
        self.names['var_pos_frac'] = r'Frac. contrib. to $\sigma^2 (t)$ from variants with $+$ cont'
        self.names['var_neg_frac'] = r'Frac. contrib. to $\sigma^2 (t)$ from variants with $-$ cont'
        self.names['diff_num_seg'] = 'Diff in no. variants with $+$ and $-$ minor allele'
        self.names['diff_num_seg_frac'] = 'Diff. in no. var. with $+$ & $-$ minor all. as frac. of all'
        self.names['logsdist'] = r'$-\omega^2/\sigma_0^2 \log (D (t)/\Delta)$'
        self.names['logdist'] = r'$- \log (D (t)/\Delta)$'
        self.names['logvar'] = r'$2N \log (\sigma^2 (t)-\sigma_0^2)$'
        self.names['logdist_guess'] = r'$- \log (\mu_3(t)/(2\sigma^2(t)\Delta))$'
        self.names['logsdist_guess'] = r'$-\omega^2/\sigma_0^2 \log (\mu_3(t)/(2\sigma^2(t)\Delta))$'
        self.names['c4'] = 'Fourth cumulant' + self.units_square
        self.names['c5'] = 'Fifth cumulant' + self.units_five
        self.names['c6'] = 'Sixth cumulant' + self.units_four

        self.names['berry_esseen'] = r'$\sum |\mu_{3,i}|$' + self.units
        self.names['berry_esseen_skew'] = r'$\sum |\mu_{3,i}|/ \sigma^3 (t)$'
        self.names['c2_sum_square'] = r'$\sum \sigma^4_{i}$' + self.units_square
        self.names['excess_kurtosis'] = 'Excess Kurtosis'

        self.names['mean_efs_fbins'] = r'Mean scaled effect size'
        self.names['mean_half_kappa_fbins'] = r'Average $\frac{a}{2}(\frac{1}{2} -x)$' + self.units
        self.names['mean_half_kappa_efs_bins'] = r'Average $\frac{a}{2}(\frac{1}{2} -x)$' + self.units
        self.names['mean_f_efs_bins'] = r'Mean frequency_pos'
        self.names['numseg_efs_bins'] = 'Number of variants'
        self.names['numseg_fbins'] = 'Number of variants'
        self.names['var_efs_bins'] = r'Genetic variance' + self.units_square
        self.names['var_fbins'] = r'Genetic variance' + self.units_square
        self.names['mu3_efs_bins'] = r'Third moment' + self.units
        self.names['mu3_fbins'] = r'Third moment' + self.units
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
        self.names['frozen_mean_dax_efs_bins'] = r'Average $a\Delta x$ since shift ' + self.units
        self.names['frozen_mean_dax_fbins'] = r'Average $a\Delta x$ since shift' + self.units
        self.names['frozen_frac_fixed_efs_bins'] = 'Fraction fixed'
        self.names['frozen_frac_extinct_efs_bins'] = 'Fraction extinct'
        self.names['frozen_frac_fixed_fbins'] = 'Fraction fixed'
        self.names['frozen_frac_extinct_fbins'] = 'Fraction extinct'
        self.names['frozen_over_half_all_efs_bins'] = r'Total number freq $>1/2$'
        self.names['frozen_over_half_all_fbins'] = r'Total number freq $>1/2$'
        self.names['frozen_frac_over_half_all_efs_bins'] = r'Frac frozen with freq $>1/2$'
        self.names['frozen_frac_over_half_all_fbins'] = r'Frac frozen with freq $>1/2$'
        self.names['frozen_frac_over_half_efs_bins'] = r'Frac segreg. with freq $>1/2$'
        self.names['frozen_frac_over_half_fbins'] = r'Frac segreg.  with freq $>1/2$'

        self.names['delta_ax_efs_bins'] = r'$\Sigma a\Delta x$ since previous generation'
        self.names['delta_ax_fbins'] = r'$\Sigma a\Delta  x$ since previous generation'
        self.names['delta_x_efs_bins'] = r'$\Sigma \Delta x$ since previous generation'
        self.names['delta_x_fbins'] = r'$\Sigma \Delta  x$ since previous generation'
        self.names['delta_var_efs_bins'] = r'Change in genetic variance' \
                                           + '\n' + 'since previous generation' + self.units_square
        self.names['delta_var_fbins'] = r'Change in genetic variance' + '\n' \
                                        + 'since previous generation' + self.units_square

        self.names['frozen_contr_d_var_efs_bins'] = r'Change in genetic variance' \
                                                    + '\n' + 'since freeze' + self.units_square
        self.names['frozen_contr_d_var_fbins'] = r'Change in genetic variance' + '\n' \
                                                 + 'since freeze' + self.units_square

        self.names['frozen_contr_d_ax_efs_bins'] = r'$\Sigma a\Delta x$ since shift' + self.units
        self.names['frozen_contr_d_ax_fbins'] = r'$\Sigma a\Delta x$ since shift' + self.units
        self.names['frozen_contr_d_x_efs_bins'] = r'$\Sigma \Delta x$ since shift'
        self.names['frozen_contr_d_x_fbins'] = r'$\Sigma \Delta x$ since shift'
        self.names['frozen_mean_fr_f_fixed_efs_bins'] = 'Mean frozen freq of fixed variants'
        self.names['frozen_median_fr_f_fixed_efs_bins'] = 'Median frozen freq of fixed variants'
        self.names['frozen_var_fr_f_fixed_efs_bins'] = 'Variance in frozen freq of fixed variants'
        self.names['frozen_mean_fr_f_extinct_efs_bins'] = 'Mean frozen freq of extinct variants'
        self.names['frozen_mean_efs_fixed_fbins'] = 'Mean scaled effect size of fixed'
        self.names['frozen_mean_efs_extinct_fbins'] = 'Mean scaled effect size of extinct'
        self.names['frozen_over_half_efs_bins'] = r'Number with freq $> 1/2$'
        self.names['frozen_over_half_fbins'] = r'Number with freq $> 1/2$'

        self.names['delta_mean_freq'] = r'Mean $\Delta x$'
        self.names['traj_time'] = 'Trajectory time'
        self.names['NUM_FIXED'] = 'Number fixed'
        self.names['mean_var'] = r'Mean contrib variance' + self.units_square
        self.names['mean_fixed_state'] = r'Mean fixed state contr' + self.units
        self.names['NUM_MUTANTS'] = 'Number mutants'
        self.names['extinction_time'] = 'Traj time, given extinction'
        self.names['var_freq'] = 'Var in freq'
        self.names['mean_freq'] = 'Mean freq'
        self.names['delta_mean_var'] = r'$\Delta$ contrb variance' + self.units_square
        self.names['NUM_EXTINCT'] = 'Number fixed'
        self.names['frac_fixed'] = 'Fraction fixed'
        self.names['var_to_SE'] = 'Conv fact var to SE'
        self.names['fixation_time'] = 'Traj time, given fixation'
        self.names['frac_extinct'] = 'Fraction extinct'
        self.names['skewness_freq'] = 'Skewness in freq'
        self.names['delta_mean_ax'] = r'Mean $ \Delta ax$ prev gen' + self.units
        self.names['mean_ax'] = r'Mean $ax$' + self.units
        self.names['mean_adx'] = r'Mean $a \Delta x$ ' + self.units
        self.names['mean_freq_over_shift'] = 'Mean freq / shift'
        self.names['mean_adx_over_shift'] = r'Mean $a \Delta x$ / shift ' + self.units
        self.names['frac_fixed_over_shift'] = 'Fraction fixed/ shift'
        self.names['frac_extinct_over_shift'] = 'Fraction extinct/ shift'

        self.names['root_S_frac_extinct'] = r'$\sqrt{S}\times$ frac extinct'
        self.names['root_S_frac_fixed'] = r'$\sqrt{S}\times$ frac fixed'
        self.names['one_minus_root_S_frac_extinct'] = r'$(1-x_i )\sqrt{S}\times$ frac extinct'
        self.names['one_minus_root_S_frac_fixed'] = r'$(1-x_i )\sqrt{S}\times$ frac fixed'
        self.names['root_S_frac_extinct_over_shift'] = r'$\sqrt{S}\times$ frac extinct / shift'
        self.names['frac_extinct_over_shift'] = r'$\sqrt{S}\times$ frac extinct / shift'
        self.names['frac_fixed_over_shift'] = 'frac fixed / shift'
        self.names['root_S_frac_fixed_over_shift'] = r'$\sqrt{S}\times$ frac fixed / shift'

        self.names['fixation_time_over_shift'] = 'Fixation time / shift'
        self.names['root_S_traj_time_over_shift'] = r'$\sqrt{S}\times$ traj time / shift'
        self.names['extinction_time_over_shift'] = 'Extinction time / shift'
        self.names['root_S_traj_time'] = r'$\sqrt{S}\times$ trajectory time'
        self.names['root_S_fixation_time'] = r'$\sqrt{S}\times$ fixation time'
        self.names['root_S_extinction_time'] = r'$\sqrt{S}\times$ extinction time'
        self.names['root_S_extinction_time_over_shift'] = r'$\sqrt{S}\times$ extinct time / shift'
        self.names['root_S_fixation_time_over_shift'] = r'$\sqrt{S}\times$ fix time / shift'
        self.names['traj_time_over_shift'] = 'Trajectory time / shift'
        self.names[
            'dist_guess_2'] = r'$\Delta e^{-\int^t_0\sigma^2 (\tau)d\tau/\omega^2 } + \mu_{3} (t)/(2\sigma^2 (t))$' + self.units
        self.names['dist_guess_3'] = r'$\Delta e^{-\int^t_0\sigma^2 (\tau)d\tau/\omega^2 }$' + self.units
        self.names['var_mean'] = r'$\int^t_0\sigma^2 (\tau)d\tau /t$' + '\n' + r'$- \sigma_0^2$' + self.units_square

        self.names['density_frac_fixed'] = r'Density$(x_i )\times$ frac fixed'
        self.names['density_frac_extinct'] = r'Density$(x_i )\times$ frac extinct'
        self.names['density_root_S_frac_extinct'] = r'$\sqrt{S}\times$ density$(x_i )\times$ frac extinct'
        self.names['density_root_S_frac_fixed'] = r'$\sqrt{S}\times$ density$(x_i )\times$ frac fixed'

        self.names['dist_var'] = r'Variance in $D(t)$' + self.units_square
        self.names['logfixed_state'] = r'$\log (\Delta-\mathrm{Fixed state} )$'
        self.names['logmean_fixed'] = r'$\log (\Delta-\mathrm{Mean fixed} )$'



        self.names['U'] = r'$U$'
        self.names['Es'] = r'$E[a^2]$'
        self.names['Vs'] = r'$V[a^2]$'
        self.names['s1'] = 'Peak 1'
        self.names['s2'] = 'Peak 2'
        self.names['f1'] = 'Frac in peak 1 '
        self.names['Delta'] = r'$\Delta$'
        self.names['shift'] = r'$\Delta$'
        self.names['w'] = r'$\sqrt{V_{S}}$'
        self.names['N'] = r'$N$'
        self.names['shift'] = r'$\Lambda$'
        self.names['shift_s0'] = r'$\Lambda$'
        self.names['sigma_0'] = r'$\sqrt{V_{A}(0)}$'
        self.names['sigma'] = r'$\sqrt{V_{A}}$'
        self.names['T_O'] = r'$T_O$'

        self.dist_lande_names = defaultdict()

        s0_text = r' (units $\sqrt{V_{A}(0)}$)'
        del_text = r' (units $\delta$)'
        self.dist_lande_names['max_dist_diff_lande_over_shift'] = r'Max|$D(t) - D_{L}(t)$|/$\Delta$'

        self.dist_lande_names['max_dist_diff_lande_s0'] = r'Max|$D(t) - D_{L}(t)$|' + s0_text
        self.dist_lande_names[
            'max_dist_diff_guess_2_s0'] = r'Max|$D(t) - (\Delta e^{-\int^t_0\sigma^2 (\tau)d\tau/\omega^2 } + \mu_{3} (t)/(2\sigma^2 (t)))$|' + s0_text
        self.dist_lande_names[
            'max_dist_diff_guess_4_s0'] = r'Max|$D(t) - (D_{L}(t) + \mu_{3} (t)/(2\sigma^2 (t)))$|' + s0_text
        self.dist_lande_names[
            'max_dist_diff_guess_3_s0'] = r'Max|$D(t) - (\Delta e^{-\int^t_0\sigma^2 (\tau)d\tau/\omega^2 }$|' + s0_text
        self.dist_lande_names['max_dist_guess_s0'] = r'Max|$\mu_{3} (t)/(2\sigma^2 (t))$|' + s0_text
        self.dist_lande_names['dist_guess_integral_s0'] = r'$\int\mu_{3} (t)/(2\sigma^2 (t))dt$' + s0_text
        self.dist_lande_names[
            'var_diff_phase_1_s0'] = r'Phase I Max|$\sigma^2_0 -  \int^t_0\sigma^2 (\tau)d\tau/t$|' + r' (units $\sigma^2_0$)'
        self.dist_lande_names[
            'var_diff_phase_1_2_s0'] = r'Phase I, II Max|$\sigma^2_0 - \int^t_0\sigma^2 (\tau)d\tau/t$|' + r' (units $\sigma^2_0$)'

        self.dist_lande_names['max_dist_guess_s0'] = r'Max|$\mu_{3} (t)/(2\sigma^2 (t))$|' + s0_text

        self.dist_lande_names['slope_log_fixed_state'] = r'$2N \times$ Slope Log($\Delta$ - Fixed state)'

        self.dist_lande_names['fixed_state_at_phase_2_time_s0'] = r'$\Delta - $ Fixed state'+ s0_text +' \n at the transition time'
        self.dist_lande_names['fixed_state_at_phase_3_time_s0'] = r'$\Delta - $ Fixed state (phase 3 time)'+ s0_text
        self.dist_lande_names['fixed_state_at_max_dist_guess_time_s0'] = r'$\Delta - $ Fixed state (Max|$\mu_{3} (t)/(2\sigma^2 (t))$| time)'+ s0_text
        self.dist_lande_names['fixed_state_at_max_mu3_time_s0'] = r'$\Delta - $ Fixed state (Max|$\mu_{3} (t)$| time)'+ s0_text
        self.dist_lande_names['fixed_state_at_max_contr_big_a_time_s0'] = r'$\Delta - $ Fixed state (max contrib big $a$ time)'+ s0_text
        self.dist_lande_names['dist_at_phase_2_time_s0'] = 'Distance from the optimum' + s0_text + ' \n at the transition time'
        self.dist_lande_names['dist_at_phase_3_time_s0'] = r'$D(t)$  (phase 3 time)'+ s0_text
        self.dist_lande_names['dist_at_max_dist_guess_time_s0'] = r'$D(t)$  (Max|$\mu_{3} (t)/(2\sigma^2 (t))$| time)'+ s0_text
        self.dist_lande_names['dist_at_max_mu3_time_s0'] = r'$D(t)$  (Max|$\mu_{3} (t)$| time)'+ s0_text
        self.dist_lande_names['dist_at_max_contr_big_a_time_s0'] = r'$D(t)$  (max contrib big $a$ time)'+ s0_text

        self.dist_lande_names['max_dist_diff_lande_delta'] = r'Max|$D(t) - D_{L}(t)$|' + del_text
        self.dist_lande_names[
            'max_dist_diff_guess_2_delta'] = r'Max|$D(t) - (\Delta e^{-\int^t_0\sigma^2 (\tau)d\tau/\omega^2 } + \mu_{3} (t)/(2\sigma^2 (t)))$|' + del_text
        self.dist_lande_names[
            'max_dist_diff_guess_4_delta'] = r'Max|$D(t) - (D_{L}(t) + \mu_{3} (t)/(2\sigma^2 (t)))$|' + del_text
        self.dist_lande_names[
            'max_dist_diff_guess_3_delta'] = r'Max|$D(t) - (\Delta e^{-\int^t_0\sigma^2 (\tau)d\tau/\omega^2 }$|' + del_text
        self.dist_lande_names['max_dist_guess_delta'] = r'Max|$\mu_{3} (t)/(2\sigma^2 (t))$|' + del_text
        self.dist_lande_names['dist_guess_integral_delta'] = r'$\int\mu_{3} (t)/(2\sigma^2 (t))dt$' + del_text
        self.dist_lande_names[
            'var_diff_phase_1_delta'] = r'Phase I Max|$\sigma^2_0 - \int^t_0\sigma^2 (\tau)d\tau/t $|' + r' (units $\delta^2$)'
        self.dist_lande_names[
            'var_diff_phase_1_2_delta'] = r'Phase I, II Max|$\sigma^2_0 -  \int^t_0\sigma^2 (\tau)d\tau/t$|' + r' (units $\delta^2$)'
        r'$\Delta - $ Fixed state' + del_text + ' \n (at the transition time)'
        self.dist_lande_names['fixed_state_at_phase_2_time_delta'] = r'$\Delta - $ Fixed state'+ del_text +' \n at the transition time'
        self.dist_lande_names['fixed_state_at_phase_3_time_delta'] = r'$\Delta - $ Fixed state (phase 3 time)'+ del_text
        self.dist_lande_names['fixed_state_at_max_dist_guess_time_delta'] = r'$\Delta - $ Fixed state (Max|$\mu_{3} (t)/(2\sigma^2 (t))$| time)'+ del_text
        self.dist_lande_names['fixed_state_at_max_mu3_time_delta'] = r'$\Delta - $ Fixed state (Max|$\mu_{3} (t)$| time)'+ del_text
        self.dist_lande_names['fixed_state_at_max_contr_big_a_time_delta'] = r'$\Delta - $ Fixed state (max contrib big $a$ time)'+ del_text
        self.dist_lande_names['dist_at_phase_2_time_delta'] = 'Distance from the optimum' + del_text + ' \n at the transition time'
        self.dist_lande_names['dist_at_phase_3_time_delta'] = r'$D(t)$  (phase 3 time)'+ del_text
        self.dist_lande_names['dist_at_max_dist_guess_time_delta'] = r'$D(t)$  (Max|$\mu_{3} (t)/(2\sigma^2 (t))$| time)'+ del_text
        self.dist_lande_names['dist_at_max_mu3_time_delta'] = r'$D(t)$  (Max|$\mu_{3} (t)$| time)'+ del_text
        self.dist_lande_names['dist_at_max_contr_big_a_time_delta'] = r'$D(t)$  (max contrib big $a$ time)'+ del_text


        #times
        self.dist_lande_names['phase_2_time'] = 'Transition time'#r'Phase II time ($D(t)\sigma^2 (t) < 1.5 \mu_{3}(t)/2$)'
        self.dist_lande_names['phase_3_time'] = r'Phase II time ($D(t)\sigma^2 (t) < 1.05 \mu_{3}(t)/2$)'
        self.dist_lande_names['time_max_contr_big_a'] = 'Time that contribution of large effect \n size variants starts decreasing'
        self.dist_lande_names['time_max_mu3'] = r'Time of Max|$\mu_3 (t)$|'
        self.dist_lande_names['time_max_dist_guess'] = r'Time of Max|$\mu_{3} (t)/(2\sigma^2 (t))$|'

        #
        # 'diff_mean_fixed_state', 'pos_fixed_pheno', 'av_mean_freq', 'diff_frac_extinct', \
        # 'pos_mean_freq_det', 'delta_neg_mean_freq', 'delta_pos_mean_ax', 'neg_mean_ax', \
        # 'pos_mean_adx', 'diff_fixed_pheno', 'neg_freq_det', 'neg_fixed_pheno', 'av_mean_third_moment', 'diff_mean_ax', \
        # 'pos_extinct_pheno', 'pos_mean_third_moment', 'pos_var_freq_det', 'neg_mean_freq_det', 'av_mean_var', 'neg_mean_fixed_state', \
        # 'neg_mean_dfreq', 'pos_mean_dfreq', 'delta_neg_mean_var', 'neg_mean_var', 'neg_mean_third_moment', 'pos_mean_fixed_state', \
        # 'pos_mean_var', 'neg_frac_extinct', 'diff_mean_dfreq', 'diff_extinct_pheno', 'pos_frac_fixed', 'pos_mean_freq', 'pos_mean_ax',\
        # 'neg_extinct_pheno', 'delta_pos_mean_freq', 'delta_neg_mean_ax', 'neg_mean_freq', 'delta_pos_mean_var', \
        # 'pos_var_freq', 'neg_var_freq', 'neg_mean_adx', 'neg_frac_fixed', 'pos_freq_det', \
        # 'diff_mean_third_moment', 'neg_var_freq_det', 'diff_frac_fixed', 'pos_frac_extinct'
        #
        # beginning_label = ['pos_','neg_','diff_','av_','delta_pos_','delta_neg_']
        # beginning_extra = ['(+) ','(-) ','Diff ', 'Av. ', '(+) Change in ','(-) Change in ']
        #
        # self.pos_neg_names['mean_fixed_state'] = 'Average fixed state'
        # self.pos_neg_names['fixed_pheno'] = 'Change in mean from fixed'
        # self.pos_neg_names['mean_freq'] = 'Mean frequency_pos'
        # self.pos_neg_names['fixed_pheno'] = 'Change in mean from fixed'
        # self.pos_neg_names['fixed_pheno'] = 'Change in mean from fixed'
        # self.pos_neg_names['fixed_pheno'] = 'Change in mean from fixed'

