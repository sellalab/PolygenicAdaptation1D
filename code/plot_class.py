import os
import numpy as np
import math
from copy import copy
from plot_functions import plot_many_y, plot_many_y_break_x,plot_many_y_hist_and_many_y
from scipy.special import dawsn, erf
from plot_names import PlotNameClass
from scipy.interpolate import interp1d
from scipy.stats import gamma
from scipy.integrate import quad
from collections import defaultdict
from copy import deepcopy, copy


def make_x_y_match_x0(x0,x,y):
    """
    Takes data y corresponding to x, and returns data y0 corresponding to x0
    :param x0: (list) The x-coordinates we want the y to correspond to
    :param x: (list) The x-coordinates y currently corresponds to
    :param y: (list) The current y-coordinates
    :return: (list) The y-coordinates corresponding to x0
    """
    if not x0:
        return []
    if x0[0] < x[0]:
        x = [x0[0]] +x
        y = [y[0]]+ y
    if x0[-1] > x[-1]:
        x+= [x0[-1]]
        y+= [y[-1]]
    myfunction = interp1d(x, y)
    y0 = [myfunction(xi) for xi in x0]
    return y0



def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def slope_list(x_list,y_list):
    """calculates the change in y over x of two lists x and y"""
    y_slope =[]
    for i in range(len(x_list)-1):
        del_x = x_list[i+1]-x_list[i]
        del_y = y_list[i+1]-y_list[i]
        y_slope.append(del_y/float(del_x))
    return x_list[:-1], y_slope

def myv(a):
    a = np.abs(a)
    return 4.0*a*dawsn(a/2.0)

def myf(a):
    return 2*a**3*np.exp(-a**2/4.0)/(np.sqrt(np.pi)*erf(a/2))


class PlotClass(object):
    def __init__(self, data_classes=None, base_dir=None):

        self.indexes = []
        # A list of the data classes
        self.data_classes = data_classes

        for cl in data_classes:
            cl.index = data_classes.index(cl)
            self.indexes.append(cl.index)

        self.max_index = max(self.indexes)

        # Define the set self.stats as the set of possible stats in dataclasses in plot_Class
        self._get_stats()

        #stats for which we have a theoretical prediction
        self._theory_stats = ['dist','opt_minus_mean_fixed_over_shift','dist_sum_over_2N_scaled']
        self._theory_hbstats_at_time = ['frozen_nm_and_standing_d2ax_scaled_per_mut_input_efs_bins',
                                        'frozen_d2ax_scaled_per_mut_input_efs_bins',
                                        'frozen_nm_2ax_scaled_per_mut_input_efs_bins',
                                        'frozen_d2ax_scaled_per_mut_input_fbins',
                                        'frozen_d2ax_over_shift_fbins']


        self._dont_plot_trajfstats = ['var_to_SE','NUM_FIXED','NUM_EXTINCT','NUM_MUTANTS','NUM_SEGREGATING']


        self._make_nice_names()

        if base_dir is not None:
            self.base_dir = base_dir
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir)
        else:
            self.base_dir = os.getcwd()

    def theory_hbstat_function_and_name_short(self, data_class, hbstat, time):
        N = data_class.param_dict['N']

        if hbstat == 'frozen_nm_and_standing_d2ax_scaled_per_mut_input_efs_bins' or \
                hbstat == 'frozen_d2ax_scaled_per_mut_input_efs_bins':
            var_0 = data_class.param_dict['sigma_0'] ** 2
            Vs = data_class.param_dict['Vs']
            epsi = np.exp(-time * var_0 / float(Vs))
            if epsi < 0.01:
                const_for_time = 1.0
                functionname = r"$v(a)$"
            else:
                const_for_time = 1.0 - np.exp(-time * var_0 / float(Vs))
                functionname = r"$(1-D_{L}($" + str(int(time)) + r"$)/\Lambda)\cdot v(a)$"
            myfunction = lambda ss: const_for_time * 4.0 * np.sqrt(np.abs(ss)) * dawsn(np.sqrt(np.abs(ss)) / 2.0)

        elif hbstat == 'frozen_d2ax_scaled_per_mut_input_fbins':
            E2Ns = data_class.param_dict['E2Ns']
            V2Ns = data_class.param_dict['V2Ns']
            var_0 = data_class.param_dict['sigma_0'] ** 2
            Vs = data_class.param_dict['Vs']
            epsi = np.exp(-time * var_0 / float(Vs))
            if epsi < 0.01:
                const_for_time = 1.0
                functionname = r"$\int^{\infty}_{0}v(a,x)\cdot g(a)da$"
            else:
                const_for_time = 1.0 - np.exp(-time * var_0 / float(Vs))
                functionname = r"$(1-D_{L}($" + str(
                    int(time)) + r"$)/\Lambda)\cdot \int^{\infty}_{0}v(a,x)\cdot g(a)da$"
            myfunction = lambda x0: const_for_time * 4 * E2Ns * (1 + V2Ns / E2Ns * x0 * (1 - x0)) ** (
                -(E2Ns ** 2 / V2Ns + 1))
        else:
            myfunction = lambda ss: 0
            functionname = ''

        return myfunction, functionname

    # NEw
    def theory_hbstat_function_and_name_long(self, data_class, hbstat):

        if hbstat == 'frozen_nm_and_standing_d2ax_scaled_per_mut_input_efs_bins':
            C = data_class.get_C()
            amplification = 1.0 + C
            print("C " + str(C))
            myfunction = lambda ss: amplification * 2 * np.sqrt(np.abs(ss)) ** 3 * np.exp(-ss / 4.0) / (
                        np.sqrt(np.pi) *
                        erf(np.sqrt(np.abs(ss)) / 2))
            functionname = r"$(1+C) \cdot f(a)$"
        elif hbstat == 'frozen_d2ax_scaled_per_mut_input_efs_bins':
            A = data_class.get_A()
            amplification = 1.0 + A
            myfunction = lambda ss: amplification * 2 * np.sqrt(np.abs(ss)) ** 3 * np.exp(-ss / 4.0) / (
                    np.sqrt(np.pi) * erf(np.sqrt(np.abs(ss)) / 2))
            functionname = r"$(1+A) \cdot f(a)$"


        elif hbstat == 'frozen_nm_2ax_scaled_per_mut_input_efs_bins':
            B = data_class.get_B()
            amplification = B
            E2Ns = data_class.param_dict['E2Ns']
            V2Ns = data_class.param_dict['V2Ns']
            SHAPE_S, SCALE_S = float(E2Ns) ** 2 / float(V2Ns), float(V2Ns) / float(E2Ns)
            S_dist = gamma(SHAPE_S, loc=0., scale=SCALE_S)
            var_0_del = data_class.param_dict['sigma_0_del'] ** 2
            mut_input = 2 * data_class.param_dict['N'] * data_class.param_dict['U']
            normalizer = float(mut_input) / float(var_0_del)

            myfunction = lambda ss: amplification * 2 * np.sqrt(np.abs(ss)) ** 3 * np.exp(-ss / 4.0) / (
                    np.sqrt(np.pi) * erf(np.sqrt(np.abs(ss)) / 2)) * S_dist.pdf(ss) * normalizer
            functionname = r"$ B \cdot f(a)\cdot g(a)/\int^{\infty}_{0}v(a)\cdot g(a)da$"

        elif hbstat == 'frozen_d2ax_scaled_per_mut_input_fbins' or hbstat == 'frozen_d2ax_over_shift_fbins':
            A = data_class.get_A()
            amplification = 1.0 + A
            E2Ns = data_class.param_dict['E2Ns']
            V2Ns = data_class.param_dict['V2Ns']
            SHAPE_S, SCALE_S = float(E2Ns) ** 2 / float(V2Ns), float(V2Ns) / float(E2Ns)
            S_dist = gamma(SHAPE_S, loc=0., scale=SCALE_S)
            to_integrate = lambda ss: 2 * np.sqrt(np.abs(ss)) ** 3 * np.exp(-ss / 4.0) / (
                    np.sqrt(np.pi) * erf(np.sqrt(np.abs(ss)) / 2)) * S_dist.pdf(ss)
            b = S_dist.ppf(0.99999999999999)

            my_integral_f_a = quad(to_integrate, 0, b)[0]
            if hbstat == 'frozen_d2ax_scaled_per_mut_input_fbins':
                myfunction = lambda x0: amplification * 2 * my_integral_f_a
                functionname = r"$(1+A) \cdot 2 \int^{\infty}_{0}f(a)\cdot g(a)da$"
            else:
                normization = data_class.get_C() + 1.0
                binwidth = 0.05
                myfunction = lambda x0: 2 * amplification / normization * binwidth
                functionname = r"$2(1+A)/(1+C)\cdot$binsize $= 2(1+A)/(1+C)\cdot 0.05$"
        else:
            myfunction = lambda ss: 0
            functionname = ''

        return myfunction, functionname

    def plot_bstat(self, bstat, indexes=None, domain =None, domain_2=None, frac = None, yrange=None, er = None, label = None, theory_scale=None,long=None, std=None, other=None):
        self._plot_bstat(bstat=bstat, indexes=indexes, domain = domain, domain_2=domain_2, frac=frac, yrange= yrange, er = er, label = label,long=long, theory_scale=theory_scale, std=std, other=other)

    def plot_h_bstat_over_time_quick(self, h_bstat, index, er = None, domain = None, yrange = None, label = None, pos = None):
        """Er True means plot standard error"""

        undertext_add = False

        multiply_se_by = 1.96

        if pos is None:
            pos = 'both'
        elif pos not in ['pos','neg']:
            pos = 'both'

        if er is None:
            er = True

        if isinstance(index,int):
            index_list = [index]
        else:
            index_list =[indi for indi in index]


        data_class_list = [self.data_classes[indi] for indi in index_list]


        undertext_params = [['N', 'U'], ['shift_s0', 'sigma_0_del'], ['f1', 's1', 's2']]


        plotspecs = dict()
        plot_dict = dict()
        plotspecs['undertext_font'] =  {'color': 'black', 'weight': 'roman', 'size': 16}
        plotspecs['legend_anchor'] = 'upper left'
        plotspecs['legend_loc'] = (1.02, 1.03)
        plot_dict['savedir'] = os.path.join(self.base_dir)
        plotspecs['fsize'] = (28, 11)
        plotspecs['dpi'] = 200
        plotspecs['linewidth'] = 3.5
        plotspecs['ticksize'] = 30
        plotspecs['legend_font'] = {'size': '30'}
        plotspecs['axis_font'] = {'fontname': 'Arial', 'size': '34'}

        bin_type = data_class_list[0].bin_type[h_bstat]
        if bin_type == 'efs':
            ylabel = self.name_class.yname(h_bstat[:-9])
        else:
            ylabel = self.name_class.yname(h_bstat[:-6])


        savedir = os.path.join(plot_dict['savedir'], 'hbstats')

        if label is None:
            label = h_bstat + '_' + str(index)
        else:
            label +=  '_' +str(index)

        if pos == 'both':
            label = 'dpn_' + label
            savedir = os.path.join(savedir, 'combined')
        elif pos == 'pos':
            label ='p_' + label
            savedir = os.path.join(savedir,'positives')
        else:
            label = 'n_' + label
            savedir = os.path.join(savedir, 'negatives')


        x = []
        y = []
        yer = []


        histo_ynames_pos = list(copy(data_class_list[0].histo_labels[h_bstat]))
        histo_leg_label = copy(data_class_list[0].histo_leg_titles[h_bstat])

        if h_bstat not in data_class_list[0]._hbstats:
            print(h_bstat + "not in" + str(0))
            return

        data = data_class_list[0].read_hbstats(h_bstat)

        the_cols = sorted(data[h_bstat][pos].keys())


        histo_ynames = histo_ynames_pos

        plot_dict['ynames'] = histo_ynames
        plot_dict['legend_title'] = histo_leg_label

        vlines = [data_class_list[0].get_phase_two_time()]#, data_class_list[0].phase_3_time]
        plot_dict['vlines'] = vlines
        plotspecs['vlinecolor'] = 'black'
        plotspecs['vlineswidth'] = 3

        plotspecs['xshade'] = [0, data_class_list[0].get_phase_two_time()]

        groupings = []
        jj = 0

        for ci in the_cols:
            data = data_class_list[0].read_hbstats(h_bstat)
            times_1 = list(sorted(data[h_bstat][pos][ci].keys()))
            times = list([tt for tt in times_1])

            group = set()

            #fmult = f_multipliers[ci]

            y_1 = list([data[h_bstat][pos][ci][x_0]['mean'] for x_0 in times_1])

            # print 'the ss'
            # print data_class_list[0].hist_bins[h_bstat]

            x.append(times)
            group.add(jj)
            jj+=1

            y.append([y1i for y1i in y_1])


            if er:
                yer1 = list([multiply_se_by*data[h_bstat][pos][ci][x_0]['se'] for x_0 in times_1])
                yer.append(yer1)

            groupings.append(group)




        undertext = []
        if len(data_class_list) == 1:
            if undertext_add:
                number_runs_string = "Obtained from " + str(int(data_class_list[0].number_population_simulations())) + \
                                     " population sims with parameters:"
                undertext.append(number_runs_string)
                for listi in undertext_params:
                    text_list = self._plot_text(index_list=[index],params_list=listi)
                    if text_list:
                        text_string = ', '.join(text_list)
                        undertext.append(text_string )

                plot_dict['undertext'] = undertext#data_class.plot_text()

        plot_dict['xlabel'] = 'Time (generations)'
        plot_dict['ylabel'] = ylabel
        plot_dict['savedir'] = savedir
        plot_dict['domain']= domain
        plot_dict['yrange'] = yrange
        plot_dict['plotspecs'] = plotspecs
        plot_dict['label'] = label

        plot_dict['x'] = x
        plot_dict['y'] = y

        if er:
            plot_dict['yer'] = yer

        plot_many_y(**plot_dict)

    def plot_h_bstat_times_hist(self, h_bstat, times, index,  er = None, domain = None,
                                 yrange = None, label = None,pos=None,theory=True):
        """times is a list of times at which we want the histo"""
        data_class = self.data_classes[index]
        if h_bstat not in data_class._hbstats:
            print(h_bstat + "not in" + str(data_class.index))
            return

        multiply_se_by = 1.96

        if pos is None:
            pos = 'both'

        data = data_class.read_hbstats(h_bstat)

        times = sorted(times)
        if times[0] < 0:
            maxtime = max(data[h_bstat][pos][0].keys())
            times[0] = maxtime

        if h_bstat not in self._theory_hbstats_at_time:
            theory = False


        if er is None:
            er = True

        undertext_params = [['N', 'U'], ['shift_s0', 'sigma_0_del'], ['E2Ns', 'V2Ns']]

        data_class = self.data_classes[index]

        bin_type = data_class.bin_type[h_bstat]
        if bin_type == 'efs':
            rawname = h_bstat[:-9]
            loglinear = True
            if domain is None:
                domain = [0.1,100]
            elif domain[0] < 0.1:
                domain[0] = 0.1
            xlabel = "Effect size squared (" +r"$S=a^2$"+')'
        else:
            loglinear = False
            rawname = h_bstat[:-6]
            xlabel = "Initial MAF"
            if domain is None:
                domain = [0, 0.5]
        ylabel = self.name_class.yname(rawname)

        binedged = data_class.hist_bins[h_bstat]


        savedir = os.path.join(self.base_dir, 'hbstats_at_times')

        if label is None:
            label = h_bstat + '_' + str(index)
        else:
            label +=  '_' +str(index)


        if pos == 'both':
            label = 'dpn_' + label
            savedir = os.path.join(savedir, 'combined')
        elif pos == 'pos':
            label ='p_' + label
            savedir = os.path.join(savedir,'positives')
        else:
            label = 'n_' + label
            savedir = os.path.join(savedir, 'negatives')

        time_string = "generation_"+str(times[0])
        for ti in times[1:]:
            time_string+='_and_'+str(ti)
        savedir = os.path.join(savedir, time_string)


        plot_dict = dict()
        plot_dict['xlabel'] = xlabel
        plot_dict['savedir'] = savedir
        plot_dict['yrange'] = yrange
        plot_dict['domain'] = domain
        plot_dict['ylabel'] = ylabel


        plotspecs = dict()
        plotspecs['undertext_font'] = {'color': 'black', 'weight': 'roman', 'size': '10'}
        plotspecs['legend_anchor'] = 'upper left'
        plotspecs['legend_loc'] = (1.02, 1.03)
        plotspecs['fsize'] = (28, 11)
        plotspecs['dpi'] = 200
        plotspecs['linewidth'] = 3.5
        plotspecs['ticksize'] = 30
        plotspecs['legend_font'] = {'size': '30'}
        plotspecs['axis_font'] = {'fontname': 'Arial', 'size': '28'}
        if loglinear:
            plotspecs['xlog'] = True


        # data = data_class_list[0].read_hbstats(h_bstat)
        # the_cols = sorted(data[h_bstat][pos].keys())

        cols = sorted(data[h_bstat][pos].keys())
        times_all = sorted(data[h_bstat][pos][0].keys())

        times_real = []
        y_hist = []
        ynames_hist = []

        if er:
            yer_hist = []

        for ti in times:
            time_time = find_nearest(times_all,ti)
            times_real.append(time_time)
            label+= '_time_'+str(time_time)
            ynames_hist.append('Time: ' + str(time_time))
            #data = data_class.read_bstats(h_bstat)
            ys = [data[h_bstat][pos][co][time_time]['mean'] for co in cols]
            # print(ys)
            y_hist.append(ys)
            if er:
                yser = [multiply_se_by*data[h_bstat][pos][co][time_time]['se'] for co in cols]
                yer_hist.append(yser)

        binedges = [binedged for _ in times_real]

        plot_dict['binedges'] = binedges
        plot_dict['y_hist'] = y_hist
        if er:
            plot_dict['yer_hist'] = yer_hist

        if len(times)>1:
            plot_dict['ynames_hist'] = ynames_hist

        #some_styles = ['o','^','s','D','x','+','*','v','h']
        # time_styles = dict()
        # for i in xrange(len(times_real)):
        #     time_styles[times[i]] = some_styles[i]


        plot_dict['label'] = label

        # List with [text_top, text_bottom] containing relevant parameters

        undertext = []
        undertext = []
        if bin_type == 'efs' and rawname[-5:]=="input":
            threshold_2Ns = data_class.scaled_s_above_which_only_one_mutation_per_generation()
            warning_string = "NOTE: on average only 1 new mut per generation has " + r"$a^2>$"+\
                             str(threshold_2Ns) +". 'Per mut input' results could be inaccurate in bins"
            number_runs_string = "with few new muts unless enough simulations were done. " + str(
                int(data_class.number_population_simulations())) + \
                                 " population sims were performed with parameters:"
            undertext.append(warning_string)
            undertext.append(number_runs_string)
        else:
            number_runs_string = "Obtained from " + str(int(data_class.number_population_simulations())) + \
                                 " population sims with parameters:"
            undertext.append(number_runs_string)
        for listi in undertext_params:
            text_list = self._plot_text(index_list=[index],params_list=listi)
            if text_list:
                text_string = ', '.join(text_list)
                undertext.append(text_string )

        plot_dict['undertext'] = undertext#data_class.plot_text()
        plotspecs['undertext_font'] = {'color': 'black', 'weight': 'roman', 'size': 16}
        plot_dict['plotspecs'] = plotspecs

        #add a theory stat

        if theory:
            numxs = 100
            sfirst = domain[0]
            slast = domain[1]
            xi = np.linspace(sfirst, slast, numxs)
            xishort = np.array([ss for ss in xi])
            xilong = np.array([ss for ss in xi])
            time_for_short = min(times)
            myfunctionlong, longfname = self.theory_hbstat_function_and_name_long(data_class, h_bstat)
            yilong = np.array([myfunctionlong(ss) for ss in xilong])
            if h_bstat == 'frozen_d2ax_over_shift_fbins':
                x = [xilong]
                y = [yilong]
                ynames = [longfname]
            else:
                myfunctionshort, shortfname = self.theory_hbstat_function_and_name_short(data_class,h_bstat, time_for_short)
                yishort = np.array([myfunctionshort(ss) for ss in xishort])
                x = [xishort, xilong]
                y = [yishort, yilong]
                ynames = [shortfname, longfname]
            plot_dict['x'] = x
            plot_dict['y'] = y
            plot_dict['ynames'] = ynames

        plot_many_y_hist_and_many_y(**plot_dict)


    def paper_plot_tstats_times(self, tstat, indexes, time=None, freq=None, params = None, er = None, domain = None, yrange = None, label= None, lande=None,
                                  xi_range=None, s_range=None, pred=None, compare=None, pos =None):
        """er True means plot standard error"""

        if pos is None:
            pos = 'both'

        multiply_se_by = 1.96

        if label is None:
            label = tstat

        no_undertext = False
        no_legend = False
        if tstat == 'd2ax_per_mut_input':
            no_legend = True

        possible_compare_stats = ['d2ax_scaled_per_mut_input']

        if time is None:
            time = False

        if compare is None:
            compare = False
        if pos != 'both':
            compare = False

        if er is None:
            er = True
        if freq is None:
            freq = False
        if lande is None:
            lande = False

        if freq:
            loglinear = False
        else:
            loglinear = True

        if s_range is None:
            sr = False
        else:
            sr = True
        if xi_range is None:
            xir = False
        else:
            xir = True
        if pred is None:
            pred = False
        legend_with_dclass_param = False
        if params is not None:
            legend_with_dclass_param = True

        data_classes = [self.data_classes[indi] for indi in indexes]

        if lande is None:
            if data_classes[0]._THERE_ARE_FINAL_TRAJ_NON_LANDE:
                lande = False
            else:
                lande = True

        only_lande = False
        for dc in data_classes:
            if 'U' not in dc.param_dict:
                only_lande = True
                lande = True
            #if there are no nonlande trajectories then we must use the Lande ones
            if not data_classes[0]._THERE_ARE_FINAL_TRAJ_NON_LANDE:
                only_lande = True
                lande = True

        if only_lande:
            undertext_params = [['N', 'Vs'], ['shift_s0', 'sigma_0_del']]
        else:
            undertext_params = [['N', 'U'], ['shift_s0', 'sigma_0_del'], ['E2Ns', 'V2Ns']]

        if tstat not in data_classes[0]._tstats:
            print(str(tstat) + ' not in dataclass ')
            return

        if freq or tstat not in possible_compare_stats:# or len(data_classes) >1:
            compare = False


        plot_dict = dict()
        plotspecs = dict()
        plotspecs['legend_anchor'] = 'upper left'
        plotspecs['legend_loc'] = (1.02, 1.03)
        plot_dict['savedir'] = self.base_dir
        plotspecs['fsize'] = (28, 16)
        plotspecs['dpi'] = 200
        plotspecs['linewidth'] = 1.5
        plotspecs['ticksize'] = 30
        plotspecs['legend_font'] = {'size': '54'}
        plotspecs['axis_font'] = {'fontname': 'Arial', 'size': '28'}
        plot_dict['linestyles'] = ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-']
        plotspecs['marker_size'] = 15
        plotspecs['cap_size'] = 20
        #plotspecs['nxticks'] = 2
        plotspecs['undertext_font'] = {'color': 'black', 'weight': 'roman', 'size': '16'}
        extra_text = ''

        #print groupings

        # plotspecs['fsize'] = (17, 11)
        # plotspecs['legend_anchor'] = 'upper right'
        # plotspecs['legend_loc'] = (0.98, 0.98)
        # plotspecs['legend_anchor'] = 'upper right'
        # if tstat == 'mean_freq':
        #     plotspecs['legend_loc'] = (0.98,0.98)
        # elif tstat == 'mean_adx':
        #     plotspecs['legend_loc'] = (0.98, 0.45)
        #
        # plotspecs['ticksize'] = 35
        # plotspecs['linewidth'] = 5
        # plotspecs['marker_size'] = 20
        # #plotspecs['nyticks'] = 2
        # plotspecs['axis_font'] = {'fontname': 'Arial', 'size': '44'}
        # plotspecs['legend_font'] = {'size': '38'}


        if not no_undertext and len(data_classes)==1:
            undertext = []
            #str(int(data_class.number_population_simulations()))
            if not lande:
                number_runs_string = "Obtained from " + str(int(data_classes[0].number_allele_pairs(False,final=False))) + \
                                     " allele traj's using NonLande D(t) averaged over " \
                                        +str(int(data_classes[0].number_population_simulations()))\
                                     + " population sims with parameters:"
            elif lande and only_lande:
                number_runs_string = "Obtained from " + str(int(data_classes[0].number_allele_pairs(True, final=False))) + \
                                     " allele traj's for each effect size, with Lande D(t) and parameters:"
            elif lande and not only_lande:
                number_runs_string1st = "NonLande obtained from " + str(int(data_classes[0].number_allele_pairs(False,final=False))) + \
                                     " allele traj's with NonLande D(t) averaged over " \
                                        +str(int(data_classes[0].number_population_simulations()))+ " population sims"
                undertext.append(number_runs_string1st)
                number_runs_string = "Lande obtained from " + str(int(data_classes[0].number_allele_pairs(True, final=False))) + \
                                     " allele traj's for each effect size, with corresponding Lande D(t)"
            undertext.append(number_runs_string)
            for listi in undertext_params:
                text_list = self._plot_text(index_list=[indexes[0]],params_list=listi)
                if text_list:
                    text_string = ', '.join(text_list)
                    undertext.append(text_string )
            plot_dict['undertext'] = undertext#data_class.plot_text()

        if not no_legend:
            if len(data_classes) ==1:
                if compare:
                    if tstat == 'd2ax_scaled_per_mut_input':
                        anal_name = r"$(1-D_{L}($"+str(int(time))+ r"$)/\Lambda)\cdot v(a)$"
                    else:
                        anal_name ='Analytic'
                    plot_dict['ynames'] = ['Simulations', anal_name]
                if lande and not only_lande:
                    plot_dict['ynames'] = ['NonLande', 'lande']
                    if compare:
                        plot_dict['ynames'] = ['NonLande', 'lande','Analytic']
            else:
                if lande and not only_lande:
                    plot_dict['groupings_labels_within'] = ['NonLande', 'lande']
                    if compare:
                        plot_dict['groupings_labels_within'] = ['NonLande', 'lande','Analytic']
                if compare:
                    plot_dict['groupings_labels_within'] = ['Simulations', 'Analytic']

        if freq:
            if pred:
                plot_dict['xlabel'] = r'$x_i $ percentile'
            else:
                plot_dict['xlabel'] = r'$x_i$'
        else:
            plot_dict['xlabel'] = "Effect size squared (" + r"$S=a^2$" + ')'
            if pred:
                if not legend_with_dclass_param:
                    plot_dict['legend_title'] = 'Percentile of \n initial frequency_pos'


        plot_dict['savedir'] = os.path.join(plot_dict['savedir'],'tstats_times')
        if pred:
            plot_dict['savedir'] = os.path.join(plot_dict['savedir'],'pred')
        if freq:
            plot_dict['savedir'] = os.path.join(plot_dict['savedir'],'S_sorted')
        else:
            plot_dict['savedir'] = os.path.join(plot_dict['savedir'], 'XI_sorted')

        savedir = os.path.join(self.base_dir, 'tstats_times')

        if pred:
            savedir = os.path.join(savedir, 'pred')
        if freq:
            savedir = os.path.join(savedir, 'S_sorted')
        else:
            savedir = os.path.join(savedir, 'XI_sorted')

        if pos == 'both':
            label = 'dpn_' + label
            savedir = os.path.join(savedir, 'combined')
        elif pos == 'pos':
            label ='p_' + label
            savedir = os.path.join(savedir,'positives')
        else:
            label = 'n_' + label
            savedir = os.path.join(savedir, 'negatives')


        if domain is None:
            if not freq:
                epsl = 0.05
                epsh =2
                domain = [0.1-epsl,100+epsh ]

        plot_dict['domain']= domain
        plot_dict['yrange'] = yrange

        at_time_text = ' at generation '

        if time:
            at_time_text += str(time)
        else:
            time = data_classes[0].get_phase_two_time()

        plot_dict['ylabel'] =  self.name_class.yname(tstat)

        if tstat == 'd2ax_per_mut_input':
            if pos == 'both':
                plot_dict['ylabel'] = 'Contribution to change in mean\n phenotype '
            if data_classes[0].units_s0:
                plot_dict['ylabel'] += r' (units trait SD)'
            else:
                plot_dict['ylabel'] += r' (units $\omega/\sqrt{2N}$)'
            plot_dict['ylabel']+= ' per\n unit mutational input'
        elif tstat == 'x_per_seg_var' or tstat == 'x_per_seg_var':
            #plot_dict['ylabel'] = 'Average increased frequency of\n aligned alleles'
            if pos == 'both':
                plot_dict['ylabel'] = 'Increased frequency of aligned\n alleles per seg variant'
            else:
                plot_dict['ylabel'] = 'Average allele frequency'


        plot_dict['marker'] = True
        if compare and (not lande or only_lande):
            if len(data_classes) == 1:
                plot_dict['colors'] = ['deepskyblue','black']


        ts_string = '_' +tstat
        _mylabel = ts_string


        less = False

        x = []
        y = []
        yer = []
        ynames = []

        #maybe
        x_other = []
        y_other = []
        yer_other = []
        ynames_other = []



        tstati = tstat


        for data_class in data_classes:


            if 'var_0' in data_class.param_dict:
                var_0 = data_class.param_dict['var_0']
            else:
                var_0 = data_class.param_dict['sigma_0'] ** 2
            N = data_class.param_dict['N']
            Vs = data_class.param_dict['Vs']
            var_0_delta_square = var_0 * 2 * N / float(Vs)
            sig_0_del = np.sqrt(var_0_delta_square)

            if 'shift_s0' in data_class.param_dict:
                D_sig_0 = data_class.param_dict['shift_s0']
                D_del = sig_0_del * D_sig_0

            x_1 = defaultdict(list)
            y_1 = defaultdict(list)
            yer_1 = defaultdict(list)
            ynames_1 = dict()

            x_theory_1 = []
            y_theory_1 = []

            if lande and not only_lande:
                x_1_other = defaultdict(list)
                y_1_other = defaultdict(list)
                yer_1_other = defaultdict(list)
                ynames_1_other = dict()


            triples = deepcopy(data_class.tuples)
            if xir:
                remov = []
                xilow = xi_range[0]
                xihigh = xi_range[1]
                for trip in triples:
                    if pred:
                        XPI = data_class.xpercentile_dict[trip]
                    else:
                        XPI = trip[1]
                    if XPI < xilow or XPI > xihigh:
                        remov.append(trip)
                for trip in remov:
                    triples.remove(trip)
            if sr:
                #print 's_range', s_range
                remov = []
                slow = s_range[0]
                shigh = s_range[1]
                for trip in triples:
                    s = trip[0]
                    if s < slow or s > shigh:
                        remov.append(trip)
                for trip in remov:
                    triples.remove(trip)

            name = ''
            lib = ''
            if params is None:
                # if len(data_classes) > 1:
                #     name += str(data_class.index)
                lib += str(data_class.index)
            else:
                for param in params:
                    try:
                        name += param + ' = ' + '{0:.2f}'.format(data_class.param_dict[param]) + '  '
                        lib += param + '{0:.0f}'.format(data_class.param_dict[param]) + '_'
                    except KeyError:
                        print('KeyError: ' + param)


            stat_dict = data_class.read_tstats(tuples=triples, requiredstats=[tstati],lande=only_lande)

            if lande and not only_lande:
                stat_dict_other = data_class.read_tstats(tuples=triples, requiredstats=[tstati], lande=True)
                print('lande stat dict')


            for triple in triples:

                name_1 = ''

                times = sorted(stat_dict[triple][tstati].keys())
                if time:
                    time_real = find_nearest(times,time)

                y_val_2 = stat_dict[triple][tstati][time_real][pos]['mean']
                y_val_er_2 = stat_dict[triple][tstati][time_real][pos]['se']
                if y_val_2 == 0.0:
                    print('yval is zero for S ', triple[0])
                if lande and not only_lande:
                    y_val_2_other = stat_dict_other[triple][tstati][time_real][pos]['mean']
                    y_val_er_2_other = stat_dict_other[triple][tstati][time_real][pos]['se']
                XI = triple[1]
                if XI < 0:
                    less = True
                    XI = -XI
                else:
                    less = False
                S = triple[0]
                if freq:
                    lenformat = 1
                    if pred and not less:
                        val = data_class.xpercentile_dict[triple]
                    else:
                        val = XI
                    # if root_s:
                    #     name_1 = r'$\sqrt{S} =$ '
                    # else:
                    #     name_1 = r'$S =$ '
                    if legend_with_dclass_param:
                        key = data_class.param_dict[params[0]]
                    else:
                        key = S
                else:
                    #key =XI
                    val = S
                    lenformat = 3
                    if xir:
                        if xi_range[1] < 1:
                            lenformat = 2
                    if pred:
                        key = data_class.xpercentile_dict[triple]
                        if key < 0.91:
                            key = round(key,1)
                        elif key <0.991:
                            key = round(key,2)
                        else:
                            key = round(key,3)
                        if key < 0:
                            if legend_with_dclass_param:
                                key = data_class.param_dict[params[0]]
                            else:
                                key = -key
                        # name_1 = ''
                    else:
                        if legend_with_dclass_param:
                            key = data_class.param_dict[params[0]]
                        else:
                            key = XI
                        # name_1 = ''
                if less:
                    lenformat =0
                x_1[key].append(val)
                y_1[key].append(y_val_2)

                if lande and not only_lande:
                    x_1_other[key].append(val)
                    y_1_other[key].append(y_val_2_other)

                if er:
                    yer_1[key].append(y_val_er_2*multiply_se_by)
                    if lande and not only_lande:
                        yer_1_other[key].append(y_val_er_2_other*multiply_se_by)
                if legend_with_dclass_param:
                    name_1 += self.name_class.param_text(value=key, digits=2)
                else:
                    name_1 += self.name_class.param_text(value=key, digits=lenformat) + ' ' + name


                name_2_lande = 'lande: '+ name_1

                if name_1 not in ynames_1:
                    ynames_1[key] = name_1
                    if lande and not only_lande:
                        if name_2_lande not in ynames_1_other:
                            ynames_1_other[key] = name_2_lande

            x.append(x_1)
            y.append(y_1)
            yer.append(yer_1)
            ynames.append(ynames_1)
            if lande and not only_lande:
                x_other.append(x_1_other)
                y_other.append(y_1_other)
                yer_other.append(yer_1_other)
                ynames_other.append(ynames_1_other)

            time_string = "generation_" + str(time_real)
            savedir = os.path.join(savedir, time_string)
            plot_dict['savedir'] = savedir

            if compare:
                kkey = list(x_1.keys())[0]
                smin = min(x_1[kkey])
                smax = max(x_1[kkey])
                xtheorylowers = [xx for xx in np.linspace(smin, smin+1, 50)]
                xtheoryhighers = [xx for xx in np.linspace(smin+1, smax, 100)]
                x_theory = xtheorylowers+xtheoryhighers[1:]
                s_theory = [ssi for ssi in x_theory]
                y_theory = [0 for _ in x_theory]
                if tstat == 'd2ax_scaled_per_mut_input':
                    frac_integral_lande_dt = 1.0 - data_class._get_lande_dist_over_shift_at_time(time)
                    if pos == 'both':
                        y_theory = [frac_integral_lande_dt*myv(np.sqrt(ss)) for ss
                                    in s_theory]

                x_theory_1.append(x_theory)
                y_theory_1.append(y_theory)

        _mylabel = _mylabel + lib

        if len(data_classes)>1:
            if less:
                if not freq:
                    plot_dict['legend_title'] = r'Shift (units $\sigma_0$)'
            if legend_with_dclass_param:
                plot_dict['legend_title'] = self.name_class.param_text(name=params[0])


        x_2 =[]
        y_2 =[]
        yer_2 = []
        ynames_2 = []
        jj = 0
        groupings = []
        keys_list = []
        # for k in xrange(len(x)):
        #     keys_list+=x[k].keys()
        # keys_list = list(set(keys_list))

        for key in sorted(x[0]):
            seti = set()
            for k in range(len(x)):
                if er:
                    zipi = list(zip(*sorted(zip(x[k][key], y[k][key], yer[k][key]))))
                    x_temp = [xi for xi in zipi[0]]
                    y_temp = [yi for yi in zipi[1]]
                    yer_temp = zipi[2]
                    yer_2.append(yer_temp)
                else:
                    zipi = list(zip(*sorted(zip(x[k][key], y[k][key]))))
                    x_temp = [xi for xi in zipi[0]]
                    y_temp = [yi for yi in zipi[1]]
                x_2.append(x_temp)
                y_2.append(y_temp)
                ynames_2.append(ynames[k][key])
                seti.add(jj)
                jj += 1

                if lande and not only_lande:
                    if er:
                        zipi = list(zip(*sorted(zip(x_other[k][key], y_other[k][key], yer_other[k][key]))))
                        x_temp = [xi for xi in zipi[0]]
                        y_temp = [yi for yi in zipi[1]]
                        yer_temp = zipi[2]
                        yer_2.append(yer_temp)
                    else:
                        zipi = list(zip(*sorted(zip(x_other[k][key], y_other[k][key]))))
                        x_temp = [xi for xi in zipi[0]]
                        y_temp = [yi for yi in zipi[1]]
                    x_2.append(x_temp)
                    y_2.append(y_temp)
                    ynames_2.append(ynames_other[k][key])
                    seti.add(jj)
                    jj += 1

                if compare:
                    if x_theory_1:
                        x_2.append(x_theory_1[k])
                        y_2.append(y_theory_1[k])
                        seti.add(jj)
                        jj += 1
                        if er:
                            yer_2.append([0 for _ in x_theory_1[k]])

            groupings.append(seti)


        plot_dict['x'] = x_2
        plot_dict['y'] = y_2
        if er:
            plot_dict['yer'] = yer_2

        #experimenting
        # if len(data_classes) >1:
        #     plot_dict['groupings'] = groupings
        plot_dict['groupings'] = groupings


        linestyles =['','-']
        markerstyles = ['o','']
        if lande and not only_lande:
            markerstyles = ['o', '*','-']
            linestyles = ['','','']
        if compare:
            linestyles = ['','-','']
            markerstyles =['o','','*']
            if lande and not only_lande:
                linestyles = ['', '-','--']
                markerstyles = ['o', '','']

        plot_dict['linestyles'] = linestyles #+linestyles+linestyles+linestyles
        plot_dict['markerstyles'] = markerstyles #+markerstyles+markerstyles+markerstyles

        size_group = len(groupings[0])
        if label is None:
            if len(_mylabel) <30:
                plot_dict['label'] = _mylabel +'_many_cl'
            else:
                plot_dict['label'] = tstat
        else:
            plot_dict['label'] =  label

        if freq:
            plot_dict['label']+= '_xi_x'


        if pred:
            plot_dict['label'] += '_pred'

        if compare:
            plot_dict['label'] += '_comp'



        if time:
            plot_dict['label'] += '_t_' + str(time_real)


            #List with [text_top, text_bottom] containing relevant parameters
        # if len(data_classes) == 1:
        #     if extra_text:
        #         undertext.append(extra_text)
        #
        # if not no_undertext:
        #     plot_dict['undertext'] = undertext

        if len(data_classes)>1:
            plot_dict['ynames'] = ynames_2

        if loglinear:
            plotspecs['xlog'] = True


        plot_dict['plotspecs'] = plotspecs

        plot_many_y(**plot_dict)


    def paper_plot_final_tstats(self, ftstat, indexes, freq=None, params = None, er = None, domain = None, yrange = None, label= None, lande=None,
                                  xi_range=None, s_range=None, pred=None, compare=None, pos =None):
        """er True means plot standard error"""

        if pos is None:
            pos = 'both'

        multiply_se_by = 1.96

        if label is None:
            label = ftstat

        no_undertext = False
        no_legend = False
        # if ftstat == '2a_frac_fixed_scaled_per_mut_input':
        #     no_legend = True

        possible_compare_stats = ['2a_frac_fixed_scaled_per_mut_input']


        if compare is None:
            compare = False
        if pos != 'both':
            compare = False

        if er is None:
            er = True
        if freq is None:
            freq = False

        if freq:
            loglinear = False
        else:
            loglinear = True

        if s_range is None:
            sr = False
        else:
            sr = True
        if xi_range is None:
            xir = False
        else:
            xir = True
        if pred is None:
            pred = False
        legend_with_dclass_param = False
        if params is not None:
            legend_with_dclass_param = True

        data_classes = [self.data_classes[indi] for indi in indexes]

        if lande is None:
            if data_classes[0]._THERE_ARE_FINAL_TRAJ_NON_LANDE:
                lande = False
            else:
                lande = True

        only_lande = False
        for dc in data_classes:
            if 'U' not in dc.param_dict:
                only_lande = True
                lande = True
            #if there are no nonlande trajectories then we must use the Lande ones
            if not data_classes[0]._THERE_ARE_FINAL_TRAJ_NON_LANDE:
                only_lande = True
                lande = True

        if only_lande:
            undertext_params = [['N', 'Vs'], ['shift_s0', 'sigma_0_del']]
        else:
            undertext_params = [['N', 'U'], ['shift_s0', 'sigma_0_del'], ['E2Ns', 'V2Ns']]

        if ftstat not in data_classes[0]._ftstats:
            print(str(ftstat) + ' not in dataclass ')
            return

        if freq or ftstat not in possible_compare_stats:# or len(data_classes) >1:
            compare = False


        plot_dict = dict()
        plotspecs = dict()
        plotspecs['legend_anchor'] = 'upper left'
        plotspecs['legend_loc'] = (1.02, 1.03)
        plotspecs['fsize'] = (28, 16)
        plotspecs['dpi'] = 200
        plotspecs['linewidth'] = 1.5
        plotspecs['ticksize'] = 30
        plotspecs['legend_font'] = {'size': '54'}
        plotspecs['axis_font'] = {'fontname': 'Arial', 'size': '28'}
        plot_dict['linestyles'] = ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-']
        plotspecs['marker_size'] = 15
        plotspecs['cap_size'] = 20
        #plotspecs['nxticks'] = 2
        plotspecs['undertext_font'] = {'color': 'black', 'weight': 'roman', 'size': '16'}
        extra_text = ''

        #print groupings


        if not no_undertext and len(data_classes)==1:
            undertext = []
            final = True
            #str(int(data_class.number_population_simulations()))
            if not lande and not only_lande:
                number_runs_string = "Obtained from " + str(int(data_classes[0].number_allele_pairs(False,final=final))) + \
                                     " allele traj's using NonLande D(t) averaged over " \
                                        +str(int(data_classes[0].number_population_simulations()))\
                                     + " population sims with parameters:"
            elif lande and only_lande:
                number_runs_string = "Obtained from " + str(int(data_classes[0].number_allele_pairs(True, final=final))) + \
                                     " allele traj's for each effect size, with Lande D(t) and parameters:"
            elif lande and not only_lande:
                number_runs_string1st = "NonLande obtained from " + str(int(data_classes[0].number_allele_pairs(False,final=final))) + \
                                     " allele traj's with NonLande D(t) averaged over " \
                                        +str(int(data_classes[0].number_population_simulations()))+ " population sims"
                undertext.append(number_runs_string1st)
                number_runs_string = "Lande obtained from " + str(int(data_classes[0].number_allele_pairs(True, final=final))) + \
                                     " allele traj's for each effect size, with corresponding Lande D(t)"
            undertext.append(number_runs_string)
            for listi in undertext_params:
                text_list = self._plot_text(index_list=[indexes[0]],params_list=listi)
                if text_list:
                    text_string = ', '.join(text_list)
                    undertext.append(text_string )
            plot_dict['undertext'] = undertext#data_class.plot_text()

        if not no_legend:
            if len(data_classes) ==1:
                if compare:
                    if ftstat == '2a_frac_fixed_scaled_per_mut_input':
                        if not only_lande:
                            anal_name =  r"$ (1+A)\cdot f(a)$"
                        else:
                            anal_name = r"$ f(a)$"
                    else:
                        anal_name ='Analytic'
                    plot_dict['ynames'] = ['Simulations', anal_name]
                if lande and not only_lande:
                    plot_dict['ynames'] = ['NonLande', 'lande']
                    if compare:
                        plot_dict['ynames'] = ['NonLande', 'lande','Analytic']
            else:
                if lande and not only_lande:
                    plot_dict['groupings_labels_within'] = ['NonLande', 'lande']
                    if compare:
                        plot_dict['groupings_labels_within'] = ['NonLande', 'lande','Analytic']
                if compare:
                    plot_dict['groupings_labels_within'] = ['Simulations', 'Analytic']

        if freq:
            if pred:
                plot_dict['xlabel'] = r'$x_i $ percentile'
            else:
                plot_dict['xlabel'] = r'$x_i$'
        else:
            plot_dict['xlabel'] = "Effect size squared (" + r"$S=a^2$" + ')'
            if pred:
                if not legend_with_dclass_param:
                    plot_dict['legend_title'] = 'Percentile of \n initial frequency_pos'



        savedir = os.path.join(self.base_dir, 'ftstats')
        if pred:
            savedir = os.path.join(savedir, 'pred')
        if freq:
            savedir = os.path.join(savedir, 'S_sorted')
        else:
            savedir = os.path.join(savedir, 'XI_sorted')

        if pos == 'both':
            label = 'dpn_' + label
            savedir = os.path.join(savedir, 'combined')
        elif pos == 'pos':
            label ='p_' + label
            savedir = os.path.join(savedir,'positives')
        else:
            label = 'n_' + label
            savedir = os.path.join(savedir, 'negatives')


        if domain is None:
            if not freq:
                epsl = 0.05
                epsh =2
                domain = [0.1-epsl,100+epsh ]

        plot_dict['domain']= domain
        plot_dict['yrange'] = yrange

        plot_dict['ylabel'] =  self.name_class.yname(ftstat)

        if ftstat == 'x_per_seg_var' or ftstat == 'x_per_seg_var':
            #plot_dict['ylabel'] = 'Average increased frequency of\n aligned alleles'
            if pos == 'both':
                plot_dict['ylabel'] = 'Increased fixation prob of aligned\n alleles per seg variant'
            else:
                plot_dict['ylabel'] = 'Average fixation probability'


        plot_dict['marker'] = True
        if compare and (not lande or only_lande):
            if len(data_classes) == 1:
                plot_dict['colors'] = ['deepskyblue','black']


        ts_string = '_' +ftstat
        _mylabel = ts_string


        less = False

        x = []
        y = []
        yer = []
        ynames = []

        #maybe
        x_other = []
        y_other = []
        yer_other = []
        ynames_other = []



        tstati = ftstat


        for data_class in data_classes:


            if 'var_0' in data_class.param_dict:
                var_0 = data_class.param_dict['var_0']
            else:
                var_0 = data_class.param_dict['sigma_0'] ** 2
            N = data_class.param_dict['N']
            Vs = data_class.param_dict['Vs']
            var_0_delta_square = var_0 * 2 * N / float(Vs)
            sig_0_del = np.sqrt(var_0_delta_square)

            if 'shift_s0' in data_class.param_dict:
                D_sig_0 = data_class.param_dict['shift_s0']
                D_del = sig_0_del * D_sig_0

            x_1 = defaultdict(list)
            y_1 = defaultdict(list)
            yer_1 = defaultdict(list)
            ynames_1 = dict()

            x_theory_1 = []
            y_theory_1 = []

            if lande and not only_lande:
                x_1_other = defaultdict(list)
                y_1_other = defaultdict(list)
                yer_1_other = defaultdict(list)
                ynames_1_other = dict()


            triples = deepcopy(data_class.tuples)
            if xir:
                remov = []
                xilow = xi_range[0]
                xihigh = xi_range[1]
                for trip in triples:
                    if pred:
                        XPI = data_class.xpercentile_dict[trip]
                    else:
                        XPI = trip[1]
                    if XPI < xilow or XPI > xihigh:
                        remov.append(trip)
                for trip in remov:
                    triples.remove(trip)
            if sr:
                #print 's_range', s_range
                remov = []
                slow = s_range[0]
                shigh = s_range[1]
                for trip in triples:
                    s = trip[0]
                    if s < slow or s > shigh:
                        remov.append(trip)
                for trip in remov:
                    triples.remove(trip)

            name = ''
            lib = ''
            if params is None:
                # if len(data_classes) > 1:
                #     name += str(data_class.index)
                lib += str(data_class.index)
            else:
                for param in params:
                    try:
                        name += param + ' = ' + '{0:.2f}'.format(data_class.param_dict[param]) + '  '
                        lib += param + '{0:.0f}'.format(data_class.param_dict[param]) + '_'
                    except KeyError:
                        print('KeyError: ' + param)


            stat_dict = data_class.read_tfstats(tuples=triples, lande=only_lande)

            if lande and not only_lande:
                stat_dict_other = data_class.read_tfstats(tuples=triples, lande=True)
                print('lande stat dict')


            for triple in triples:

                name_1 = ''


                y_val_2 = stat_dict[triple][tstati][pos]['mean']
                y_val_er_2 = stat_dict[triple][tstati][pos]['se']
                if y_val_2 == 0.0:
                    print('yval is zero for S ', triple[0])
                if lande and not only_lande:
                    y_val_2_other = stat_dict_other[triple][tstati][pos]['mean']
                    y_val_er_2_other = stat_dict_other[triple][tstati][pos]['se']
                XI = triple[1]
                if XI < 0:
                    less = True
                    XI = -XI
                else:
                    less = False
                S = triple[0]
                if freq:
                    lenformat = 1
                    if pred and not less:
                        val = data_class.xpercentile_dict[triple]
                    else:
                        val = XI
                    # if root_s:
                    #     name_1 = r'$\sqrt{S} =$ '
                    # else:
                    #     name_1 = r'$S =$ '
                    if legend_with_dclass_param:
                        key = data_class.param_dict[params[0]]
                    else:
                        key = S
                else:
                    #key =XI
                    val = S
                    lenformat = 3
                    if xir:
                        if xi_range[1] < 1:
                            lenformat = 2
                    if pred:
                        key = data_class.xpercentile_dict[triple]
                        if key < 0.91:
                            key = round(key,1)
                        elif key <0.991:
                            key = round(key,2)
                        else:
                            key = round(key,3)
                        if key < 0:
                            if legend_with_dclass_param:
                                key = data_class.param_dict[params[0]]
                            else:
                                key = -key
                        # name_1 = ''
                    else:
                        if legend_with_dclass_param:
                            key = data_class.param_dict[params[0]]
                        else:
                            key = XI
                        # name_1 = ''
                if less:
                    lenformat =0
                x_1[key].append(val)
                y_1[key].append(y_val_2)

                if lande and not only_lande:
                    x_1_other[key].append(val)
                    y_1_other[key].append(y_val_2_other)

                if er:
                    yer_1[key].append(y_val_er_2*multiply_se_by)
                    if lande and not only_lande:
                        yer_1_other[key].append(y_val_er_2_other*multiply_se_by)
                if legend_with_dclass_param:
                    name_1 += self.name_class.param_text(value=key, digits=2)
                else:
                    name_1 += self.name_class.param_text(value=key, digits=lenformat) + ' ' + name


                name_2_lande = 'lande: '+ name_1

                if name_1 not in ynames_1:
                    ynames_1[key] = name_1
                    if lande and not only_lande:
                        if name_2_lande not in ynames_1_other:
                            ynames_1_other[key] = name_2_lande

            x.append(x_1)
            y.append(y_1)
            yer.append(yer_1)
            ynames.append(ynames_1)
            if lande and not only_lande:
                x_other.append(x_1_other)
                y_other.append(y_1_other)
                yer_other.append(yer_1_other)
                ynames_other.append(ynames_1_other)


            plot_dict['savedir'] = savedir

            if compare:
                kkey = list(x_1.keys())[0]
                smin = min(x_1[kkey])
                smax = max(x_1[kkey])
                xtheorylowers = [xx for xx in np.linspace(smin, smin+1, 50)]
                xtheoryhighers = [xx for xx in np.linspace(smin+1, smax, 100)]
                x_theory = xtheorylowers+xtheoryhighers[1:]
                s_theory = [ssi for ssi in x_theory]
                y_theory = [0 for _ in x_theory]
                if ftstat == '2a_frac_fixed_scaled_per_mut_input':
                    if pos == 'both':
                        myA = 0
                        if not only_lande:
                            myA = data_class.get_A()
                        y_theory = [(1.0+myA)*myf(np.sqrt(ss)) for ss
                                    in s_theory]

                x_theory_1.append(x_theory)
                y_theory_1.append(y_theory)

        _mylabel = _mylabel + lib

        if len(data_classes)>1:
            if less:
                if not freq:
                    plot_dict['legend_title'] = r'Shift (units $\sigma_0$)'
            if legend_with_dclass_param:
                plot_dict['legend_title'] = self.name_class.param_text(name=params[0])

        x_2 =[]
        y_2 =[]
        yer_2 = []
        ynames_2 = []
        jj = 0
        groupings = []
        keys_list = []
        # for k in xrange(len(x)):
        #     keys_list+=x[k].keys()
        # keys_list = list(set(keys_list))

        for key in sorted(x[0]):
            seti = set()
            for k in range(len(x)):
                if er:
                    zipi = list(zip(*sorted(zip(x[k][key], y[k][key], yer[k][key]))))
                    x_temp = [xi for xi in zipi[0]]
                    y_temp = [yi for yi in zipi[1]]
                    yer_temp = zipi[2]
                    yer_2.append(yer_temp)
                else:
                    zipi = list(zip(*sorted(zip(x[k][key], y[k][key]))))
                    x_temp = [xi for xi in zipi[0]]
                    y_temp = [yi for yi in zipi[1]]
                x_2.append(x_temp)
                y_2.append(y_temp)
                ynames_2.append(ynames[k][key])
                seti.add(jj)
                jj += 1

                if lande and not only_lande:
                    if er:
                        zipi = list(zip(*sorted(zip(x_other[k][key], y_other[k][key], yer_other[k][key]))))
                        x_temp = [xi for xi in zipi[0]]
                        y_temp = [yi for yi in zipi[1]]
                        yer_temp = zipi[2]
                        yer_2.append(yer_temp)
                    else:
                        zipi = list(zip(*sorted(zip(x_other[k][key], y_other[k][key]))))
                        x_temp = [xi for xi in zipi[0]]
                        y_temp = [yi for yi in zipi[1]]
                    x_2.append(x_temp)
                    y_2.append(y_temp)
                    ynames_2.append(ynames_other[k][key])
                    seti.add(jj)
                    jj += 1

                if compare:
                    if x_theory_1:
                        x_2.append(x_theory_1[k])
                        y_2.append(y_theory_1[k])
                        seti.add(jj)
                        jj += 1
                        if er:
                            yer_2.append([0 for _ in x_theory_1[k]])

            groupings.append(seti)


        plot_dict['x'] = x_2
        plot_dict['y'] = y_2
        if er:
            plot_dict['yer'] = yer_2

        #experimenting
        # if len(data_classes) >1:
        #     plot_dict['groupings'] = groupings
        plot_dict['groupings'] = groupings


        linestyles =['','-']
        markerstyles = ['o','']
        if lande and not only_lande:
            markerstyles = ['o', '*','-']
            linestyles = ['','','']
        if compare:
            linestyles = ['','-','']
            markerstyles =['o','','*']
            if lande and not only_lande:
                linestyles = ['', '-','--']
                markerstyles = ['o', '','']

        plot_dict['linestyles'] = linestyles #+linestyles+linestyles+linestyles
        plot_dict['markerstyles'] = markerstyles #+markerstyles+markerstyles+markerstyles

        size_group = len(groupings[0])
        if label is None:
            if len(_mylabel) <30:
                plot_dict['label'] = _mylabel +'_many_cl'
            else:
                plot_dict['label'] = ftstat
        else:
            plot_dict['label'] = label #ftstat +'_' + label

        if freq:
            plot_dict['label']+= '_xi_x'


        if pred:
            plot_dict['label'] += '_pred'

        if compare:
            plot_dict['label'] += '_comp'



            #List with [text_top, text_bottom] containing relevant parameters
        # if len(data_classes) == 1:
        #     if extra_text:
        #         undertext.append(extra_text)
        #
        # if not no_undertext:
        #     plot_dict['undertext'] = undertext

        if len(data_classes)>1:
            plot_dict['ynames'] = ynames_2

        if loglinear:
            plotspecs['xlog'] = True


        plot_dict['plotspecs'] = plotspecs

        plot_many_y(**plot_dict)

    def _plot_text(self,index_list=None,params_list=None):
        """Returns a list of texts for plots"""

        if index_list is None:
            index_list = self.indexes
        if params_list is None:
            params_list = []#['N','U']
        remove = []

        #remove the non matching params or params that aren't in all the data classes
        for param in params_list:
            if not self._param_match(index_list,param):
                remove.append(param)
        for param in remove:
            params_list.remove(param)

        text_list = []

        dclass = self.data_classes[index_list[0]]
        for param in params_list:
            text_list.append(self.name_class.param_text(name=param, value=dclass.param_dict[param]))

        return text_list


    def special_stat(self,data_class,sp_stat):
        if sp_stat == 'logdist' or sp_stat == 'dist_var' or sp_stat == 'logsdist':
            base_stat = 'dist'
        elif sp_stat == 'logvar' or sp_stat == 'dist_guess_2' or sp_stat == 'dist_guess_3' or sp_stat == 'var_mean':
            base_stat = 'var'
        elif sp_stat == 'logfixed_state':
            base_stat ='fixed_state'
        elif sp_stat == 'logmean_fixed':
            base_stat ='mean_fixed'
        elif sp_stat == 'logdist_guess' or sp_stat == 'logsdist_guess':
            base_stat = 'dist_guess'
        elif sp_stat == 'diff_delta_var_dmu3':
            base_stat = 'delta_mu3'
        elif sp_stat == 'var' or sp_stat == 'dvar' or sp_stat == 'halfmu3' or sp_stat == 'mu3' or sp_stat=='dist_guess':
            base_stat = sp_stat
        else:
            print('no base stat in special stat function')
            return
        data = data_class.read_bstats(base_stat)
        x =  sorted(data[base_stat].keys())
        y = [data[base_stat][tim]['mean'] for tim in x]

        if sp_stat == 'logsdist' or sp_stat == 'logsdist_guess':
            #print 'hi' + sp_stat + base_stat
            y = [data_class.theory_curves.logs_dist(d) for d in y]
            #y = [d for d in y]
        elif sp_stat == 'logdist' or sp_stat == 'logdist_guess':
            #print 'hi' + sp_stat + base_stat
            y = [data_class.theory_curves.log_dist(d) for d in y]
            #y = [d for d in y]
        elif sp_stat == 'logvar':
            y = [data_class.theory_curves.log_var_new(yi) for yi in y]
        elif sp_stat == 'logfixed_state':
            y = [data_class.theory_curves.log_fixed_state(yi) for yi in y]
        elif sp_stat == 'logmean_fixed':
            y = [data_class.theory_curves.log_mean_fixed(yi) for yi in y]
        elif sp_stat == 'dist_guess_2' or sp_stat =='dist_guess_3' or sp_stat =='var_mean':
            _, yvar_mean, _ = data_class.get_stat_averaging(stat = 'var',mean=True)
            if sp_stat == 'dist_guess_2':
                base_stat_2 = 'dist_guess'
                data = data_class.read_bstats(base_stat_2)
                x2 = sorted(data[base_stat_2].keys())
                y2 = [data[base_stat_2][tim]['mean'] for tim in x2]
                y = [data_class.theory_curves.dist_guess_2(t, var, dist_guess) for t, var, dist_guess in
                     zip(x, yvar_mean, y2)]
            elif sp_stat == 'dist_guess_3':
                y = [data_class.theory_curves.dist_guess_2(t, var,0) for t, var in zip(x,yvar_mean)]
            elif sp_stat == 'var_mean':
                y= [yi - data_class.theory_curves._VAR_0_EMP/data_class.theory_curves._UNITS**2 for yi in yvar_mean]
        elif sp_stat == 'dist_var':
            base_stat_2 = 'dist_square'
            data = data_class.read_bstats(base_stat_2)
            #x2 = sorted(data_class.bstat_dict[base_stat_2].keys())
            y2 = [data[base_stat_2][tim]['mean'] for tim in x]
            y = [ds-exp_ds**2 for exp_ds, ds in zip(y,y2)]
        elif sp_stat == 'diff_delta_var_dmu3':
            base_stat_2 = 'dmu3'
            data = data_class.read_bstats(base_stat_2)
            x2 = sorted(data[base_stat_2].keys())
            y2 = [data[base_stat_2][tim]['mean'] for tim in x]
            base_stat_3 = 'var'
            x3 = sorted(data[base_stat_3].keys())
            y3 = [data[base_stat_3][tim]['mean'] for tim in x]
            if x != x2 or x != x3:
                print('Different times for delta_var and dmu3')
                return x2, y2
            else:
                #y = [data_class.theory_curves.diff_delta_var_dmu3_rat(del_var,dm3,var) for del_var,dm3,var in zip(y,y2,y3)]
                y = [data_class.theory_curves.diff_delta_var_dmu3(del_var, dm3) for del_var, dm3 in zip(y, y2)]
            zero_index = x.index(0)
            y[zero_index] = 0
            y[zero_index+1] = 0
            y[zero_index + 2] = 0
            print(x[zero_index+2])
        return x, y

    def theory_stat(self,data_class,bstat,theory_scale=None,other=False):
        N =data_class.param_dict['N']

        base_stat = bstat
        data = data_class.read_bstats(base_stat)
        x =  sorted(data[base_stat].keys())
        y = [data[base_stat][tim]['mean'] for tim in x]
        if bstat == 'dist':
            y = [data_class.theory_curves.dist_lande(tim) for tim in x]
        if bstat == 'opt_minus_mean_fixed_over_shift':
            y = [data_class.theory_curves.dist_mean_fixed_over_shift(tim) for tim in x]
        if bstat == 'dist_sum_over_2N_scaled':
            y = [data_class.theory_curves.integral_dist_lande_over_2N_scaled(tim) for tim in x]
        return x, y

    def _param_match(self,index_list,param):

        if index_list is None:
            index_list = self.indexes


        dclasses = [self.data_classes[indi] for indi in index_list]

        if param in dclasses[0].param_dict:
            vali = dclasses[0].param_dict[param]
        else:
            return False

        for cl in dclasses[1:]:
            if param not in cl.param_dict:
                return False
            else:
                if vali != cl.param_dict[param]:
                    return False

        return True

    def _get_stats(self):
        """Tells us what stats have been added by updating _bstats and _hbstats and stats"""
        self.stats = set()
        self._bstats = set()
        self._h_bstats = set()
        self._tstats = set()
        self._ftstats = set()
        for cl in self.data_classes:
            for stat in cl._bstats:
                self.stats.add(stat)
                self._bstats.add(stat)
            for stat in cl._hbstats:
                self.stats.add(stat)
                self._h_bstats.add(stat)
            for stat in cl._tstats:
                self._tstats.add(stat)
                self.stats.add(stat)
            try:
                trips = cl.triples
                f_stats = cl.read_tfstats(trips,eq=False,lande=False)
                for trip in f_stats:
                    for stat in f_stats[trip]:
                        self._ftstats.add(stat)
                        self.stats.add(stat)
            except:
                AttributeError

    def _make_nice_names(self):
        if self.data_classes[0].units_s0:
            units = 's0'
        else:
            units = 'delta'
        self.name_class = PlotNameClass(units=units)


    def _plot_bstat(self, bstat, indexes=None, params = None, er = None, domain = None, domain_2=None, frac=None, yrange = None,long=None, label= None, theory_scale=None, std =None, other=None):
        """er True means plot standard error"""
        if indexes == None:
            indexes = self.indexes

        multiply_se_by = 1.96

        if er is None:
            er = False

        if other is None:
            other = False

        bstat_stat = 'mean'
        bstat_er = 'se'
        if std is None:
            std = False #Do I plot the std of the stat instead
        if std:
            bstat_stat = 'std'
        if std:
            er = False


        conference = False

        undertext_params = [['N', 'U'], ['shift_s0', 'sigma_0_del'], ['f1','s1', 's2']]
        if conference:
            undertext_params = []

        landeshort = False
        nolandelong = False

        fewticks = False

        data_classes = [self.data_classes[indi] for indi in indexes]

        if self.data_classes[0].units_s0:
            myunits = self.name_class.units
            delti = False
            if conference:
                myunits = ' (units trait SD)'
        else:
            myunits = r' (units $\delta =\omega/\sqrt{2N}$)'
            #myunits = r' (units $\delta$)'
            delti = True


        if int is None:
            long = True

        plot_dict = dict()
        plot_dict['xlabel'] = 'Time (generations)'
        plot_dict['savedir'] = os.path.join(self.base_dir)
        plot_dict['domain']= domain
        plot_dict['yrange'] = yrange
        plot_dict['linestyles'] = ['-','-','-']


        plotspecs = dict()
        if int:
            plotspecs['fsize'] = (22,5)
        else:
            plotspecs['fsize'] = (26, 14)

        plotspecs['dpi'] = 200
        plotspecs['linewidth'] = 3.5
        #plotspecs['yrotation'] = 0

        #place legend
        plotspecs['legend_loc'] = (1.01, 0.98)
        plotspecs['legend_anchor'] = 'upper left'

        if domain_2 is not None:
            if fewticks:
                plotspecs['nxticks'] = 4

        plotspecs['ticksize'] = 28

            #plotspecs['nxticks'] = 3

        if fewticks:
            plotspecs['nxticks'] = 2
            plotspecs['nyticks'] = 2
        plotspecs['legend_font'] = {'size': '32'}
        plotspecs['axis_font'] = {'fontname': 'Arial', 'size': '26'}
        if not int:
            if landeshort:
                plotspecs['linewidth'] = 9
            else:
                plotspecs['linewidth'] = 9

        if isinstance(bstat, str):
            plot_dict['ylabel'] = self.name_class.yname(bstat)
            bstats = [bstat]
            manyb = False
        elif len(bstat) == 1:
            plot_dict['ylabel'] = self.name_class.yname(bstat[0])
            bstats = bstat
            manyb = False
        else:
            manyb = True
            unit = data_classes[0]._get_units_of_stat(bstat[0])
            if unit == 1:
                plot_dict['ylabel'] = self.name_class.units
            elif unit == 2:
                plot_dict['ylabel'] = self.name_class.units_square
            elif unit == 3:
                plot_dict['ylabel'] = self.name_class.units_cube
            bstats = bstat

        _mylabel = ''

        if bstats[0] == 'dist':
            if 'dist_guess_2' not in bstats and 'dist_guess_3'  not in bstats:
                plotspecs['legend_loc'] = (0.96, 0.91)
                plotspecs['legend_anchor'] = 'upper right'
            if not delti:
                plot_dict['ylabel'] = 'Distance from optimum\n' + myunits +' \n'
            else:
                plot_dict['ylabel'] =  myunits
            if int:
                if nolandelong:
                    plotspecs['text_loc'] = [0.12, 0.4]
                    plotspecs['text_color'] = 'red'
                    plotspecs['text_size'] = 36
                    #plot_dict['text'] = 'Rapid\n decay'
                #plotspecs['text_loc'] = [0.2, 0.5]
                #plot_dict['text'] = r'Swift decrease'
            else:
                plot_dict['colors'] = ['limegreen', 'red']
                plotspecs['axis_font'] = {'fontname': 'Arial', 'size': '68'}
                plotspecs['legend_font'] = {'size': '48'}
                plotspecs['nxticks'] = 2
                if landeshort:
                    #plot_dict['text'] = r'$\longleftarrow$ Indistinguishable!'
                    plot_dict['linestyles'] = ['-','--','-','-','-','-']
                # else:
                #     plot_dict['text'] = r'$\approx$ lande'
                else:
                    plot_dict['linestyles'] = ['-', '-', '-', '-', '-', '-']
                    print(plot_dict['linestyles'], 'hi')
                plotspecs['text_loc'] = [0.62, 0.5]
                plotspecs['text_color'] = 'indianred'
                plotspecs['text_size'] = 40

        if bstats[0] == 'var':
            if int:
                plotspecs['nyticks'] = 2

            plot_dict['ylabel'] = 'Variance in trait\n distribution ' + self.name_class.units_square
            if conference:
                plot_dict['ylabel'] = 'Variance in trait\n distribution '
                plotspecs['fsize'] = (22, 5)
                plotspecs['fsize'] = (21, 5)


            plotspecs['text_color']= 'red'
            plotspecs['text_size'] = 42
            plotspecs['text_loc'] = [0.8,0.6]
            # if not landeshort:
            #     plot_dict['text'] = r'$\sigma^2 (t)$'+r'$\neq$'+   r'$\sigma^2_{0}$'
            if domain_2 is None:
                #plot_dict['text'] = 'Gradual decrease'
                plotspecs['text_color'] = 'magenta'
                plotspecs['text_size'] = 36
                plotspecs['text_loc'] = [0.7, 0.3]
        if bstats[0] == 'mu3':
            if int:
                plot_dict['ylabel'] = 'Third moment\n in trait\n distribution\n' + self.name_class.units
            else:
                plot_dict['ylabel'] = 'Third moment\n in trait\n distribution\n' + self.name_class.units
            plot_dict['text'] = r'No longer normal!'
            plotspecs['text_color']= 'red'
            plotspecs['text_size'] = 42
            plotspecs['text_loc'] = [0.7,0.6]

        if bstats == ['dist','dist_guess']:
            plot_dict['colors'] = ['limegreen', 'blue','red']
            plot_dict['linestyles'] = ['-', '--', '-', '-', '-', '-']
            #plot_dict['text'] = 'Quasi steady state'
            plotspecs['text_color']= 'forestgreen'
            plotspecs['text_size'] = 40
            plotspecs['text_loc'] = [0.9,0.35]


        if len(data_classes) ==1:
            if not landeshort and not nolandelong:
                #vlines = [data_classes[0].phase_2_time,data_classes[0].phase_3_time]
                vlines = [data_classes[0].get_phase_two_time()]
                if conference:
                    vlines = [data_classes[0].get_phase_two_time()]

            else:
                vlines = [data_classes[0].get_phase_two_time()]

            ###########JUst for lande regeime commented
            plot_dict['vlines'] = vlines
            plotspecs['vlinecolor'] = 'black'
            plotspecs['vlineswidth'] = 4


        # print plotspecs['legend_anchor']
        # print plotspecs['legend_loc']

        if conference:
            plotspecs['nyticks'] = 2

        plot_dict['plotspecs'] = plotspecs


        x = []
        y = []
        yer = []
        ynames = []

        for bstat in bstats:
            for data_class in data_classes:
                if bstat not in data_class._bstats:
                    print(bstat + "not in" + str(data_class.index))
                    return

            _mylabel += bstat

            for data_class in data_classes:
                data = data_class.read_bstats(bstat)
                times = sorted(data[bstat].keys())
                yi = [data[bstat][tim][bstat_stat] for tim in times]

                x.append(times)
                y.append(yi)
                if er:
                    yeri = [multiply_se_by*data[bstat][tim][bstat_er] for tim in times]
                    yer.append(yeri)
                name = ''
                lib = ''
                if manyb:
                    nametemp = self.name_class.yname(bstat).split("(units")
                    name += nametemp[0]
                if bstat in self._theory_stats and not nolandelong and not std:
                    name = 'From simulations'
                if params is None:

                    if len(data_classes) >1:
                        name += ' I:'+str(data_class.index)
                    lib += str(data_class.index)
                else:
                    for param in params:
                        try:
                            name += param + ' = ' + '{0:.2f}'.format(data_class.param_dict[param]) + '  '
                            lib += param + '{0:.0f}'.format(data_class.param_dict[param]) + '_'
                        except KeyError:
                            print('KeyError: ' + param)

                if bstat == 'dist_guess':
                    if conference:
                        name = 'Closed form approximation'


                ynames.append(name)
                _mylabel = _mylabel +lib

        for bstat in bstats:
            for data_class in data_classes:
                if bstat in self._theory_stats and not nolandelong and not std:

                    times, yi = self.theory_stat(data_class, bstat, theory_scale=theory_scale, other=other)

                    x.append(times)
                    y.append(yi)
                    ynami = self.name_class.theory_yname(bstat)
                    if ynami == 'none':
                        ynami = 'T: ' + name
                    ynames.append(ynami)
                    if er:
                        yer.append([0 for _ in times])

        plot_dict['x'] = x
        plot_dict['y'] = y
        if er:
            plot_dict['yer'] = yer


        if label is None:
            if len(_mylabel) <30:
                plot_dict['label'] = _mylabel +'_many_cl'
            else:
                plot_dict['label'] = "_".join(bstats)
        else:
            plot_dict['label'] = "_".join(bstats) + '_' + label

        if std:
            plot_dict['ylabel']+= ' std'
            plot_dict['label'] += '_std'

        if len(data_classes) == 1:
            #List with [text_top, text_bottom] containing relevant parameters
            undertext = []
            number_runs_string = "Obtained from " + str(int(data_classes[0].number_population_simulations())) + \
                                 " population sims with parameters:"
            undertext.append(number_runs_string)
            for listi in undertext_params:
                text_list = self._plot_text(index_list=[indexes[0]], params_list=listi)
                if text_list:
                    text_string = ', '.join(text_list)
                    undertext.append(text_string)
            plot_dict['undertext'] = undertext  # data_class.plot_text()
            plotspecs['undertext_font'] = {'color': 'black', 'weight': 'roman', 'size': 16}


        if len(ynames) >1:
            plot_dict['ynames'] = ynames
        plot_dict['undertext'] = undertext

        # print bstats
        # print ynames

        if domain_2 is not None:
            plot_dict['domain_2'] = domain_2

            if frac is not None:
                plot_dict['frac'] = frac
            plot_many_y_break_x(**plot_dict)
        else:
            plot_many_y(**plot_dict)

#######can delete the rest for online version
    def _plot_dists_figure_2a(self, indexes = None, domain=None, two_landes=None, yrange=None, long=None, vline=None, params=None, different_stat =None, nolegend =None):
        """er True means plot standard error"""

        bstat_stat = 'mean'
        bstat_er = 'se'


        if two_landes is None:
            two_landes = False

        if nolegend is None:
            nolegend = False

        if different_stat is None:
            different_stat = False
        elif not isinstance(different_stat, str):
            different_stat = 'var'

        if indexes == None:
            indexes = self.indexes

        if int is None:
            long = False

        if int:
            manyb = True
            bstats = ['dist','dist_guess']
        else:
            manyb = False
            bstats = ['dist']

        if different_stat:
            bstats = [different_stat]

        if vline is None:
            vline = False

        if params is None:
            params = False

        indexes = indexes[:2]


        undertext_params = [['N', 'U'], ['shift_s0', 'sigma_0_del'], ['f1', 's1', 's2']]
        if not params:
            undertext_params = []
            params = []
        else:
            params =undertext_params


        fewticks = False
        the_colors = ['darkgreen', 'blue','cyan','limegreen']
        the_colors_different = ['blue', 'cyan', 'darkgreen','limegreen']

        data_classes = [self.data_classes[indi] for indi in indexes]

        if self.data_classes[0].units_s0:
            myunits = self.name_class.units
            delti = False
            myunits = ' in trait SD'
        else:
            myunits = r' (units $\delta =\omega/\sqrt{2N}$)'
            # myunits = r' (units $\delta$)'
            delti = True


        plot_dict = dict()
        plot_dict['xlabel'] = 'Generations after shift'
        plot_dict['savedir'] = os.path.join(self.base_dir)
        plot_dict['domain'] = domain
        plot_dict['yrange'] = yrange

        plotspecs = dict()
        if int:
            plotspecs['fsize'] = (22, 5)
        else:
            plotspecs['fsize'] = (26, 14)
        plotspecs['fsize'] = (26, 14)

        if different_stat:
            plotspecs['fsize'] = (26, 8)

        plotspecs['dpi'] = 200
        plotspecs['linewidth'] = 3.5

        plotspecs['ticksize'] = 34

            # plotspecs['nxticks'] = 3

        if fewticks:
            plotspecs['nxticks'] = 2
            plotspecs['nyticks'] = 2

        plotspecs['linewidth'] = 8

        _mylabel = ''

        #if bstats[0] == 'dist':
        plotspecs['legend_loc'] = (0.98, 0.98)
        plotspecs['legend_anchor'] = 'upper right'
        if not delti:
            plot_dict['ylabel'] = 'Distance from optimum ' + myunits + ' \n'
        else:
            plot_dict['ylabel'] = myunits
        if different_stat:
            if different_stat == 'var':
                plot_dict['ylabel'] = 'Variance'
                plotspecs['ylabelspace'] = 1
            elif different_stat == 'skewness':
                plot_dict['ylabel'] = 'Skewness ' + r'($\mu_{3}(t)/\sigma^3 (t)$)'
            else:
                plot_dict['ylabel'] = different_stat


        plotspecs['axis_font'] = {'fontname': 'Arial', 'size': '46'}
        plotspecs['legend_font'] = {'size': '42'}

        #
        # plotspecs['text_loc'] = [0.62, 0.5]
        # plotspecs['text_color'] = 'indianred'
        # plotspecs['text_size'] = 40


        if vline:
            #vlines = [data_classes[0].phase_2_time, data_classes[0].phase_3_time]
            vlines = [data_classes[0].get_phase_two_time()]
            plot_dict['vlines'] = vlines
            plotspecs['vlinecolor'] = 'black'
            plotspecs['vlineswidth'] = 4


        if fewticks:
            plotspecs['nyticks'] = 2

        plot_dict['plotspecs'] = plotspecs

        x = []
        y = []
        yer = []
        ynames = []
        linestyles = []
        colors = []
        len_y = 0


        col_num =  0
        for bstati in bstats:
            first_data_class = True
            for data_class in data_classes:
                if bstati not in data_class._bstats:
                    print(bstati + "not in" + str(data_class.index))
                    return

            _mylabel += bstati

            for data_class in data_classes:
                data = data_class.read_bstats(bstati)
                times = sorted(data[bstati].keys())
                yi = [data[bstati][tim][bstat_stat] for tim in times]

                style = '-'
                if different_stat:
                    coli = the_colors_different[col_num]
                else:
                    coli = the_colors[col_num]

                name = ''
                lib = ''
                if manyb:
                    nametemp = self.name_class.yname(bstati).split("(units")
                    name += nametemp[0]
                if bstati in self._theory_stats:
                    name = 'Simulations'
                    if len(data_classes) == 2:
                        if first_data_class:
                            name = 'Simulations Lande'
                        else:
                            name = 'Simulations non-Lande'
                if params is None:

                    if len(data_classes) > 1:
                        name += ' I:' + str(data_class.index)
                    lib += str(data_class.index)
                else:
                    for param in params:
                        try:
                            name += param + ' = ' + '{0:.2f}'.format(data_class.param_dict[param]) + '  '
                            lib += param + '{0:.0f}'.format(data_class.param_dict[param]) + '_'
                        except KeyError:
                            print('KeyError: ' + param)

                if bstati == 'dist_guess':
                    if int:
                        name = 'Closed form approximation'
                        name = 'Quasi static approximation: ' + r'$\mu_3 (t)/(2\sigma^2 (t))$'
                        style = '--'

                if not (bstati == 'dist_guess' and first_data_class and len(data_classes) == 2):
                    col_num += 1
                    linestyles.append(style)
                    colors.append(coli)
                    len_y+=1
                    x.append(times)
                    y.append(yi)
                    ynames.append(name)
                _mylabel = _mylabel + lib
                first_data_class = False

        for bstati in bstats:
            first_data_class = True
            for data_class in data_classes:
                if different_stat:
                    coli = the_colors_different[-1]
                else:
                    coli = the_colors[-1]
                if first_data_class:
                    style = '--'
                else:
                    style = '-'
                if first_data_class or two_landes:
                    if bstati =='var' or bstati == 'skewness' or bstati == 'mu3' or bstati in self._theory_stats and bstati != 'dist_guess':
                        if bstati =='var':
                            times = x[0]
                            var_0 = data_class.param_dict['var_0']
                            var_0 = 1.0
                            yi = [var_0 for _ in times]
                            ynami = 'Equilibrium variance'
                        elif bstati == 'skewness' or bstati == 'mu3':
                            times = x[0]
                            mu0 = 0
                            yi = [mu0 for _ in times]
                            if bstati == 'skewness':
                                ynami = 'Equilibrium skewness'
                            else:
                                ynami = 'Equilibrium third moment'

                        elif bstati in self._theory_stats:
                            times, yi = self.theory_stat(data_class, bstati)
                            ynami = self.name_class.theory_yname(bstati)

                        x.append(times)
                        y.append(yi)
                        len_y += 1
                        ynames.append(ynami)
                        colors.append(coli)
                        linestyles.append(style)

                first_data_class = False

        plot_dict['x'] = x
        plot_dict['y'] = y
        plot_dict['colors'] = colors



        if len(_mylabel) < 30:
            plot_dict['label'] = _mylabel + '_many_cl'
        else:
            plot_dict['label'] = "_".join(bstats)

        # List with [text_top, text_bottom] containing relevant parameters
        undertext = []
        for listi in undertext_params:
            text_list = self._plot_text(index_list=indexes, params_list=listi)
            if text_list:
                text_string = ', '.join(text_list)
                undertext.append(text_string)
        if len(ynames) > 1 and not nolegend:
            plot_dict['ynames'] = ynames

        plot_dict['undertext'] = undertext
        plot_dict['linestyles'] = linestyles

        plot_many_y(**plot_dict)
