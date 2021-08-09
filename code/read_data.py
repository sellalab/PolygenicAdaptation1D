"""
Contains three classes which can be used to read the results of simulatins: dataClassFullSims,
dataClassTraj and dataClass.
Use dataClassFullSims if the simulation results in your results directory were generated from population simulations.
Use dataClassTraj if the simulation results in your results directory were generated from trajectory simulations.
Use dataClassif your results directory contains results generated from both population and trajectory simulations.
"""

import os
import numpy as np
import abc
import pickle
from collections import defaultdict
from combined_theory import FreqMinorDistrE, get_variance_units_little_del_square, TheoryCurves
from scipy.optimize import curve_fit
from scipy.special import dawsn,hyp2f1, erf
from scipy.stats import gamma
import warnings
from math import fabs
from operator import itemgetter
from scipy.integrate import quad
from datetime import datetime
import csv

warnings.filterwarnings('error')

#Changes a defaultdictionary to a regular one for pickling
def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or fabs(value - array[idx-1]) < fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def load(filename):
    """Reads the pickled file -- filename -- and produces an iterator over items inside it
    Used in locate_sim_pD_path below"""
    with open(filename, "rb") as f:
        while True:
            try:
                #yield pickle.load(f, encoding="bytes")
                yield pickle.load(f, encoding="latin1")
            except EOFError:
                break

def find_name_of_sim_pD_folder(save_sims_directory,param_dict=None):
    """"return the folder name in save_sims_directory
    that has simulation results run using the parameters in param_dict"""
    sim_folders = [dir for dir in os.listdir(save_sims_directory) if dir[-3:] == '_pD']
    if sim_folders is None:
        return
    # if we are not using any criteria
    if param_dict is None:
        if sim_folders:
            return sim_folders[0]
        else:
            return False
    for sfolder in sim_folders:
        spath = os.path.join(save_sims_directory,sfolder)
        files = [os.path.join(spath, 'param_dict_basic')]
        files.append(os.path.join(spath, 'extra_param_dict_full_sims'))
        the_dict = dict()
        for file in files:
            if not os.path.isfile(file):
                print('basic paramaters not there')
                pass
            iterater = load(file)
            the_dict.update(next(iterater))
        print(the_dict)
        print(param_dict)
        MAYBE_THIS_ONE = True
        for pm in param_dict:
            if pm in the_dict:
                if param_dict[pm] != the_dict[pm]:
                    MAYBE_THIS_ONE = False
                    break
        if MAYBE_THIS_ONE:
            return sfolder
    return False

class _dataClass(object, metaclass=abc.ABCMeta):
    """Class to store the data from a set of runs with particular parameters

        Parameters
        base_dir: (string) A directory containing a set of runs with particular parameters -- will end in '_pD'
            to denote 'parameter set directory'

        Attributes:
            param_dict: (dict) A dictionary of the parameters for the run
            _base_dir: (string) The base directory
            base_dir: (string) Public copy of the base directory
            _stats: List of strings
                    List of the statistics contained in the directory --- aside from those
                    connected with trajectories
            _hbstats:  List of strings
                       List of the statistics for histograms stored in the directory
            _bstats:  List of strings
                      List of the basic statistics stored in the direcory
            _summary_set: set
                            A set of extra things that the dataclass has been used to pickle
            _units: w/(2N)**(1/2). Otherwise know as little delta. Used to change units.
            """

    def __init__(self,base_dir=os.getcwd(),units=None):

        self._base_dir = base_dir

        self._set_up_some_names()

        self.index = -1 # inititialise to -1

        self._EMP = False
        self._ALPHA = 1.0

        self.param_dict = dict()
        self._unit_dict = dict()
        self._unit_text_dict = dict()

        # Read the parameters and create the _param_dict
        self._create_param_dict()
        # Read the empirical variance if there is such
        self.set_var0_emp()
        # creates the theory variance if there isn't
        self._set_theory_var0()

        self.FIXATION_CLASS_EXISTS = False

        if units is None:
            self.units_s0 = True # if false, then units are little delta (w/sqrt(2N))
        else:
            self.units_s0 = units
        self._set_units()

        # creat the unit text
        self._make_unit_text_dict()

        self._stats  = []
        self._hbstats = []
        self._bstats = []

        self._htstats =[]
        self._tstats = []

        self._traj_stats= []
        self._traj_stats_lande = []
        self._traj_stats_non_lande = []
        self._ftstats = []
        self._ftstats_stats_lande = []
        self._ftstats_stats_non_lande = []

        # Read which full simulation stats we have, if any
        self._make_list_bstats()
        # Read trajectory info if there is
        self.tractories = False
        self.tuples = set()

        self._number_pair_alleles_trajecties_combined = 0
        self._number_pair_alleles_trajecties_combined_lande = 0
        self._number_pair_alleles_trajecties_final_combined = 0
        self._number_pair_alleles_trajecties_final_combined_lande = 0

        # Are there trajectories
        self._THERE_ARE_TRAJ = False
        self._THERE_ARE_FINAL_TRAJ = False

        # Are the trajectories of the Lande or nonLande saved
        self._THERE_ARE_TRAJ_LANDE = False
        self._THERE_ARE_FINAL_TRAJ_LANDE = False
        self._THERE_ARE_TRAJ_NON_LANDE = False
        self._THERE_ARE_FINAL_TRAJ_NON_LANDE = False

        # eta is the fraction contribution to the change in mean coming from new mutations
        self._eta = 1 # Lande case, but will be overiden if nonLande case when eta is initialized
        # Note this can only be obtained from simulations, so the data class will not be able to calculate
        # this unless you collected the statistic "d2ax_frozen_over_shift"

        # _ETA_HAS_BEEN_INITIALIZED is just to make sure we only read it from simulation
        # results if we need it, but equally that we don't have to read it many times even if
        # we use it many times
        self._ETA_HAS_BEEN_INITIALIZED = False
        # C is the allelic measure of deviation from Lanade
        # 0 means there is no deviation from Lande. For nonLande sims, will calculate it
        self._C = 0  # This will be overiden in nonLande cases when eta is initialized
        # _C_HAS_BEEN_INITIALIZED is to make sure we don't calculate it many times
        self._C_HAS_BEEN_INITIALIZED = False

        #Create a set of all extra things that have been summarized
        self._summary_set = set()

    def base_dir(self):
        return self._base_dir

    def _set_up_some_names(self):
        self._run_file_name_end = '_aRun'
        self._traj_lande_ending = '_lande'
        self._traj_starts = ['pos','neg','both']
        self._run_values = ['sum', 'sum_squares', 'num']
        self._summary_values = ['mean','std','se']
        self._summary_name = 'summary'
        self._combined_runs_name = 'combined_runs'
        self._trajectory_dir_name = 'trajectories'
        self._final_stat_dir_name = 'final_stats'

        self._recording_number_runs_combined_name = "number_of_runs_combined"

    def basic_stats(self):
        """Allows us to check which basic stats are in the directory.

        Returns: The list of stats associated with the full run
        """
        return self._stats

    def traj_stats(self):
        """Allows us to check which trajectory stats are in the directory.

        Returns: The list of stats associated with different trajectories
        """
        return self._traj_stats

    def number_population_simulations(self):
        return self._number_population_runs_combined

    def number_allele_pairs(self,lande=False,final=False):
        if not final:
            if lande:
                return self._number_pair_alleles_trajecties_final_combined_lande
            else:
                return self._number_pair_alleles_trajecties_final_combined
        else:
            if lande:
                return self._number_pair_alleles_trajecties_combined_lande
            else:
                return self._number_pair_alleles_trajecties_combined


    def set_units(self,units_s0):
        "if we would like to turn units sigma 0 on or off"
        self.units_s0 = units_s0
        self._set_units()
        self._make_unit_text_dict()


    def set_emp(self,emp):
        self._EMP = emp
        self._set_units()

    def set_alpha(self,alpha):
        self._ALPHA =alpha

    def add_summary(self, summary, summary_name):
        """summary is a LIST of stuff to be pickled to summary_name"""
        self._summary_set.add(summary_name)
        file = os.path.join(self._base_dir, summary_name)
        with open(file, 'wb') as f:
            for sum in summary:
                pickle.dump(sum, f, protocol=pickle.HIGHEST_PROTOCOL)

    def read_summary(self, summary_name):
        """Returns an iterator of the stuff in summary_name."""
        if self._check_if_file_in_base_dir(summary_name):
            file = os.path.join(self._base_dir, summary_name)
            summary = self._load(file)
        else:
            summary = summary_name+ ' does not exist'
        return summary

    def summaries(self):
        """Check what summaries have been added"""
        return self._summary_set


    @abc.abstractmethod
    def stat_writer_param_dict(self):
        """
        In the base directory there is a pickled dictionary which
         was used to creat the hash for the particular set of
         simulation parameters. This function reads it
        """
        return dict()

    @abc.abstractmethod
    def make_parameter_text_to_write_file(self):
        """
        Put the parameters into a string. The string is used to label a file
        """

    def make_identifiers(self,savedir):
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        # save some identifiers to the text results data class folder
        text_file_name = "identifiers"
        text_file = os.path.join(savedir, text_file_name+ '.txt')
        with open(text_file, "w") as f:
            f.write('creation my_time : ' + str(datetime.now()) + '\n')
            f.writelines([str(k) + '\t: ' + str(v) + '\n' for k, v in list(self.param_dict.items())])
            f.write('\n')
            name = "C"+', the allelic measure of deviation from lande'
            f.write(name+ '\t: ' + str(self.get_C()) + '\n')
            name = "eta"+', the fraction contribution to change in mean from standing variation'
            f.write(name+ '\t: ' + str(self.get_eta()) + '\n')
            name = "A"+', the long-term amplification of contribution from standing variation'
            f.write(name+ '\t: ' + str(self.get_A()) + '\n')
            name = "B"+', the long-term factor mutiplying the contribution of new mutations'
            f.write(name+ '\t: ' + str(self.get_B()) + '\n')
        return savedir

    def set_up_folder_named_by_parameter_text(self,savedir):
        parameter_string = self.make_parameter_text_to_write_file()
        savefolder = os.path.join(savedir,parameter_string)
        return self.make_identifiers(savefolder)

    def params_basic(self):
        """
        Retuns the dictionary saved in 'param_dict_basic'
        """
        file = os.path.join(self._base_dir, 'param_dict_basic')
        if not os.path.isfile(file):
            print('basic paramaters not there')
            return dict()
        else:
            iterater = self._load(file)
            return next(iterater)

    def add_param(self,key,value):
        self._add_params(key,value)

    def plot_text(self):
        return self._plot_text()

    def set_var0_emp(self):
        """Read the steady state variances at shift time from all the runs,
        and record the mean and error"""
        vartotal, varsquaretotal, run_count = self._read_all_var_emp()
        run_count = float(run_count)
        self._VAR_0_EMP, var_in_var0_emp = self._mean_var(vartotal,varsquaretotal,run_count)
        if run_count > 0:
            if var_in_var0_emp >0:
                self._VAR_0_EMP_SE = np.sqrt(var_in_var0_emp/run_count)
            else:
                print('var in var neg ', var_in_var0_emp)
                self._VAR_0_EMP_SE = 0.0
        else:
            self._VAR_0_EMP_SE = 0.0
        if self._VAR_0_EMP>0:
            self.param_dict['var_0_emp'] = self._VAR_0_EMP
            #print 'first find var 0 emp ', self._VAR_0_EMP
            self.param_dict['var_0_emp_se'] = self._VAR_0_EMP_SE

    def _plot_text(self):
        """Returns a list [text_top, text_bottom] for plots"""

        text1 = r'$N =$ ' + '{0:.0f}'.format(self.param_dict['N'])
        if 'U' in self.param_dict:
            text2 = r', $U$ = ' + '{0:.3f}'.format(self.param_dict['U'])
        else:
            text2 = ''
        text3 = r', $\Delta =$' + '{0:.1f}'.format(self.param_dict['shift_s0']) + r'$\sigma_0$'


        if 'f1' in self.param_dict:
            text5 =r'$S_2$ = ' +self.param_dict['s2']
            text6 = r', $f_1$ = ' +self.param_dict['f1']
        elif 'E2Ns' in self.param_dict:
            text5 = r'$E_2Ns$ = ' + '{0:.2f}'.format(self.param_dict['E2Ns'])
            text6 = r', $V_2Ns$ = ' + '{0:.2f}'.format(self.param_dict['V2Ns'])
        else:
            text5 = ''
            text6 = ''

        text_last = r'$\sigma_0$ = ' + '{0:.2f}'.format(self.param_dict['sigma_0'] / self._DELTA_UNIT) + r'$\delta$'

        text_bottom = text1 + text2  + text3
        text_top = text5 + text6

        if not text_top:
            return [text_bottom,text_last]
        else:
            return [text_top, text_bottom,text_last]


    def _set_units(self):
        #set the units for the class
        if self.units_s0:
            if not self._EMP:
                self._units = np.sqrt(self._VAR_0_THEORY)
            else:
                if self._VAR_0_EMP> 0:
                    self._units = np.sqrt(self._VAR_0_EMP)
                else:
                    self._units = np.sqrt(self._VAR_0_THEORY)
        else:
            self._units = self._DELTA_UNIT


    @abc.abstractmethod
    def _create_param_dict(self):
        """read the parameters and create the dictionary of parameters"""
        return

    def _get_units_of_stat(self,stat):
        """Allows us to check if the stat should be in units, units squared, units cubed, etc.

        Returns: An positive float telling us which power to raise the units to
        """
        if stat in self._unit_dict:
            return float(self._unit_dict[stat])
        else:
            return 0.0


    def _make_unit_text_dict(self):
        self._unit_text_dict[0] = ''
        if self.units_s0:
            self._unit_text_dict[1]= r'$\sqrt{V_{A}(0)}$'
            self._unit_text_dict[2] = r'$V_{A}(0)$'
            self._unit_text_dict[3] = r'$V^{3/2}_{A}(0)$'
            self._unit_text_dict[4] = r'$V^{2}_{A}(0)$'
            self._unit_text_dict[5] = r'$V^{5/2}_{A}(0)$'
            self._unit_text_dict[6] = r'$V^{3}_{A}(0)$'
        else:
            self._unit_text_dict[1] = r'$\delta$'
            self._unit_text_dict[2] = r'$\delta^2$'
            self._unit_text_dict[3] = r'$\delta^3$'
            self._unit_text_dict[4] = r'$\delta^4$'
            self._unit_text_dict[5] = r'$\delta^5$'
            self._unit_text_dict[6] = r'$\delta^6$'


    def _var_emp_full_directory_path(self):
        """Finds the full path of var 0 emp if it exists

        Returns: A string with the path name
        """
        return os.path.join(self._base_dir, 'var_0_emp_vD')


    def _get_bstat_full_directory_path_with_units(self, statname):
        """Finds the full path of a given bstat

        Returns: A string with the path name
        """
        front = 'U' + str(int(self._get_units_of_stat(statname))) + '_'
        fullname = front+statname
        if statname in self._bstats:
            stat_dir = os.path.join(self._base_dir, fullname + '_sD')
        elif statname in self._hbstats:
            stat_dir = os.path.join(self._base_dir, fullname + '_H_sD')
        else:
            stat_dir = ''
        return stat_dir

    def _check_if_file_in_base_dir(self,file):
        file = os.path.join(self._base_dir, file)
        if os.path.isfile(file):
            return True
        else:
            return False

        #print self._tstats


    @staticmethod
    def _read_tuple_from_directory_name(mydirectoryname):
        if mydirectoryname[-3:] == '_tD':
            mydirectoryname = mydirectoryname[:-3]
        if mydirectoryname[:3] != 'XI_':
            print('my dir name = ', mydirectoryname)
        assert mydirectoryname[:3] == 'XI_'
        mydirectoryname = mydirectoryname[3:]
        XI, mydirectoryname = mydirectoryname.split('_S_')
        XI = XI.replace('_','.')
        XI = float(XI)
        S, mydirectoryname = mydirectoryname.split('_t_')
        S = S.replace('_','.')
        S = float(S)
        list = mydirectoryname.split('_')
        t = int(list[0])
        return (S,XI,t)

    def _read_tuple_from_path_of_run_name(self, runpath):
        #mydirectoryname = os.path.basename(os.path.dirname(runpath))
        mydirectoryname = runpath.split('/')[-3]
        return self._read_tuple_from_directory_name(mydirectoryname=mydirectoryname)

    def _read_statname_from_path_of_run_name(self,runpath):
        mydirectoryname = os.path.basename(os.path.dirname(runpath))
        return self._read_statname_from_directory_name(mydirectoryname=mydirectoryname)

    def _test_if_lande_from_path_of_run_name(self,runpath):
        if runpath.endswith(self._traj_lande_ending):
            return True
        else:
            return False

    def _read_number_of_runs_in_combined(self,stat):
        if stat in self._stats:
            stat_dir = self._get_bstat_full_directory_path_with_units(stat)
        else:
            return 0
        filename = self._recording_number_runs_combined_name + ".txt"
        filefullname = os.path.join(stat_dir,filename)
        if os.path.exists(filefullname):
            number_runs = float(self._read_first_line_file(directory=stat_dir,filename=filefullname))
        else:
            number_runs = 0
        return number_runs

    @staticmethod
    def _read_statname_from_directory_name(mydirectoryname):
        """Given the name of a bstat or a tstat directory, discover the name of the stat

        Returns: A string with the name of the stat
        """
        stat, statbody = '',''
        if len(mydirectoryname) > 3:
            if mydirectoryname[-3:] == '_sD':
                stat = mydirectoryname[:-3]
            if mydirectoryname[-4:] == '_tsD':
                stat = mydirectoryname[:-4]
        if len(stat) > 2:
            if stat[-2:] == '_H':
                stat = stat[:-2]

        if len(stat) > 3:
            #remove unit specification
            statbody =stat[3:]

        return statbody

    def _read_stat_data_from_directory_name(self,mydirectoryname):
        """Given the name of a bstat or a tstat directory, discover the name of the stat, the units of the stat
        and whether it is a tstat or a bstat and if it is a histogram stat

        Returns: A string with the name of the stat
        """
        histo_data,bstat_data,tstat_data = False, False, False
        stat, statbody = '',''
        final_stat = False

        if len(mydirectoryname) > 3:
            if mydirectoryname[-3:] == '_sD':
                bstat_data = True
                stat = mydirectoryname[:-3]
            elif mydirectoryname[-4:] == '_tsD':
                tstat_data = True
                stat = mydirectoryname[:-4]
            elif mydirectoryname[-4:] == '_tfD':
                tstat_data = True
                final_stat = True
                stat = mydirectoryname[:-4]
        if len(stat) > 2:
            if stat[-2:] == '_H':
                histo_data = True
                stat = stat[:-2]

        if not final_stat:
            statbody, units = self._read_stat_data_from_stat_name_with_units(stat)
        else:
            statbody, units = stat, 0

        return statbody, units, bstat_data, tstat_data, histo_data

    @staticmethod
    def _read_stat_data_from_stat_name_with_units(statnamewithunit):
        """Given the name of a stat with Ux_ at the beginning, discover the name of the stat,
        and the units of the stat (i.e. is it in units of trait, trait squared, trate cubed
        etc

        Returns: A string with the name of the stat, and a string of the number for the units
        """
        if len(statnamewithunit) > 3:
            units, statbody = statnamewithunit[1], statnamewithunit[3:]
        else:
            units, statbody = 0, statnamewithunit

        return statbody, units

    def _read_all_var_emp(self):
        """ Reads the var 0 emp if there is a var 0 emp.

Returns
--------
out: vartotal, nruns
"""
        # for each statistic (represented by a subdirectory)
        var0dir = self._var_emp_full_directory_path()
        vartotal, varsquaretotal, run_count = 0, 0, 0
        if os.path.isdir(var0dir):
            run_files = self._find_files_in_dir_with_ending(var0dir, self._run_file_name_end)
            filedir = os.path.join(var0dir, self._combined_runs_name)
            # If there is alread a combined runs, read it
            if os.path.isfile(filedir):
                iterater = self._load(filedir)
                vartotal =next(iterater)
                #print 'vartotal ', vartotal
                varsquaretotal =next(iterater)
                #print 'varsquaretotal ', varsquaretotal
                run_count = next(iterater)
                #print 'run_count ', run_count

            for run_name in run_files:
                iterater = self._load(run_name)
                try:
                    thisvar0 =next(iterater)[1]
                    # print 'thisvar ', thisvar0
                    # print 'run count ', run_count
                    vartotal += thisvar0
                    varsquaretotal += thisvar0**2
                    run_count+= 1
                    os.rename(run_name, run_name + '_have_read')
                except StopIteration:
                    print('Stop iteration error in:')
                    print(run_name)
            file_name = os.path.join(var0dir, self._combined_runs_name)
            #save the new vartotal and run count
            with open(file_name, 'wb') as f:
                pickle.dump(vartotal, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(varsquaretotal, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(run_count, f, protocol=pickle.HIGHEST_PROTOCOL)

        return vartotal, varsquaretotal, run_count


    def _store_stat_name_and_data_from_directoryname(self,mydirectoryname):

        statbody, units, bstat_data, tstat_data, histo_data = self._read_stat_data_from_directory_name(mydirectoryname)

        if len(statbody) > 0:
            self._unit_dict[statbody] = float(units)
            if bstat_data:
                if statbody not in self._stats:
                    self._stats.append(statbody)
                    if histo_data:
                        self._hbstats.append(statbody)
                    else:
                        self._bstats.append(statbody)
            if tstat_data:
                if statbody == self._final_stat_dir_name:
                    self._THERE_ARE_FINAL_TRAJ = True
                elif statbody not in self._tstats:
                    self._traj_stats.append(statbody)
                    if histo_data:
                        self._htstats.append(statbody)
                    else:
                        self._tstats.append(statbody)
        self._read_number_runs_of_population_simulations()
        return

    def _read_number_runs_of_population_simulations(self):
        if not self._stats:
            self._number_population_runs_combined = 0
        else:
            if 'dist' in self._stats:
                token_stat = 'dist'
            else:
                token_stat = self._stats[0]
            self._number_population_runs_combined = self._read_number_of_runs_in_combined(token_stat)


    def _make_list_bstats(self):
        # Create list of all the basic stats
        possible_stat_directories = os.listdir(self._base_dir)
        for stat_dir in possible_stat_directories:
            self._store_stat_name_and_data_from_directoryname(stat_dir)




    def _add_params(self,key,value):
        """
        In the base directory there is a pickled dictionary of 'parameters'. This function reads it, adds a param to it
        and saves it again
        """
        file = os.path.join(self._base_dir, 'parameters')
        if not os.path.isfile(file):
            print('paramaters not there')
            pass
        iterater = self._load(file)
        self.param_dict = next(iterater)
        if key in self.param_dict:
            print('replacing ', key, ' = ', self.param_dict[key], ' with ', value)
        self.param_dict[key] = value
        info_name = 'parameters'
        #"Line 1 is dictionary of " + info_name +" values"
        list = [self.param_dict]
        self.add_summary(list,info_name)




    def _read_params_basic(self):
        """
        In the base directory there is a pickled dictionary of 'basic parameters'. This function reads it and saves the param dict
        """
        file = os.path.join(self._base_dir, 'param_dict_basic')
        if not os.path.isfile(file):
            print('basic paramaters not there')
            pass
        iterater = self._load(file)
        self.param_dict.update(next(iterater))


    def _set_theory_var0(self):
        """
        creates the theory var0 if it isn't there. Probably don't need this function
        """
        print(self.param_dict)
        self._DELTA_UNIT = np.sqrt(float(self.param_dict['Vs']))/np.sqrt(2.0*self.param_dict['N'])

        if 'sigma_0' not in self.param_dict and 'sigma_0_del' not in self.param_dict:
            if 'U' in self.param_dict and 'N' in self.param_dict and 'E2Ns' in self.param_dict and 'V2Ns' in self.param_dict:
                mutinput = 2*self.param_dict['N']*self.param_dict['U']
                E2Ns = self.param_dict['E2Ns']
                V2Ns = self.param_dict['V2Ns']
                var_0_del_square = get_variance_units_little_del_square(E2Ns=E2Ns, V2Ns=V2Ns, mut_input=mutinput)
                sigma_0_del = np.sqrt(var_0_del_square)
                sigma_0 = sigma_0_del*self._DELTA_UNIT
                self.param_dict['sigma_0'] = sigma_0
                self.param_dict['sigma_0_del'] = sigma_0_del
        elif 'sigma_0' not in self.param_dict:
            self.param_dict['sigma_0'] = self.param_dict['sigma_0_del']*self._DELTA_UNIT
        else:
            self.param_dict['sigma_0_del'] = self.param_dict['sigma_0']/(self._DELTA_UNIT)

        if 'var_0' not in self.param_dict:
            self.param_dict['var_0']  = self.param_dict['sigma_0']**2

        self._VAR_0_THEORY = self.param_dict['var_0']



    @staticmethod
    def _load(filename):
        """Reads the pickled file -- filename -- and produces an iterator over items inside it"""
        with open(filename, "rb") as f:
            while True:
                try:
                    #yield pickle.load(f, encoding="bytes")
                    yield pickle.load(f, encoding="latin1")
                except EOFError:
                    break

    @staticmethod
    def _read_first_line_file(directory,filename):
        readfolderfile = os.path.join(directory,filename)
        if os.path.exists(readfolderfile):
            tf = open(readfolderfile)
            firstline = next(tf).rstrip()
            firstline.rstrip('\n')
            tf.close()
        else:
            firstline = ""
        return firstline

    @staticmethod
    def _find_files_in_dir_with_ending(dir=None,ending=None):
        """Finds all the files in directory 'dir' with names ending in 'ending'"""
        if ending is None:
            ending = ''
        size_ending = len(ending)
        if os.path.isdir(dir) and ending[0] != '.':
            run_files = [x for x in os.listdir(dir) if x[0] != '.' and len(x) > 2]
            run_files = [x for x in run_files if x[-size_ending:] == ending]
            run_files = [os.path.join(dir, x) for x in run_files]
            return run_files
        else:
            print('Directory ' + dir + ' does not exist')
            return []


    def _summarize_columns(self,dict, dir, file_name=None):
        if file_name is None:
            file_name = self._summary_name
        if os.path.isdir(dir):
            file_name = os.path.join(dir, file_name)
            with open(file_name, 'wb') as f:
                # The first line of the pickle file is the number of runs in the summary
                for column in list(dict.keys()):
                    pickle.dump(default_to_regular(dict[column]), f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _mean_var(sum, sum_squares=0.0, num = None):

        if num is None:
            mean, var = 0, 0
        else:
            num =float(num)
            sum = float(sum)
            if num > 0:
                mean = sum / num
                if num > 1:
                    var = (sum_squares - sum ** 2 / num) / (num - 1.0)
                else:
                    var = 0.0
            else:
                mean, var = 0.0,0.0

        if sum_squares == 0.0:
            var = 0.0

        return mean, var

    @staticmethod
    def _straight_line(t, m, c):
        return -m*t + c

    @staticmethod
    def _straight_line_no_intercept(t, m):
        return -m*t

    def _get_lande_dist_over_shift_at_time(self, time):

        if 'var_0' in self.param_dict and 'Vs' in self.param_dict:
            var_0 = self.param_dict['var_0']
            Vs = self.param_dict['Vs']
            if time >= 0:
                return np.exp(-time*var_0/Vs)
            else:
                return 0
        else:
            return 0.0

    # SOME theory
    @staticmethod
    def v_a(a):
        """the steady-state phenotypic variance contributed by alleles with magnitude of phenotypic effect a,
        per unit mutational input"""
        a = np.abs(a)
        return 4.0 * a * dawsn(a / 2.0)

    @staticmethod
    def v_S(S):
        """the steady-state phenotypic variance contributed by alleles with scaled selection coefficient S,
        per unit mutational input"""
        a = np.sqrt(np.abs(S))
        return 4.0 * a * dawsn(a / 2.0)

    def v_ax(self, a, x0):
        """the steady-state phenotypic variance contributed by alleles with magnitude of phenotypic effect a,
        and frequncy x0, per unit mutational input"""
        a = np.abs(a)
        facti = 1
        if 'N' in self.param_dict:
            N = self.param_dict['N']
            denom = 1.0 / float(2 * N)
            if x0 < denom:
                facti = 2.0 * denom * x0
        return 4.0 * facti * a ** 2 * np.exp(-a ** 2 * x0 * (1 - x0))

    def v_Sx(self, S, x0):
        """the steady-state phenotypic variance contributed by alleles with scaled selection coefficient S,
        per unit mutational input"""
        a = np.sqrt(np.abs(S))
        facti = 1
        if 'N' in self.param_dict:
            N = self.param_dict['N']
            denom = 1.0 / float(2 * N)
            if x0 < denom:
                facti = 2.0 * denom * x0
        return 4.0 * facti * a ** 2 * np.exp(-a ** 2 * x0 * (1 - x0))

    @staticmethod
    def f_a(a):
        return 2 * a ** 3 * np.exp(-a ** 2 / 4.0) / (np.sqrt(np.pi) * erf(a / 2))

    @staticmethod
    def f_S(S):
        a = np.sqrt(np.abs(S))
        return 2 * a ** 3 * np.exp(-a ** 2 / 4.0) / (np.sqrt(np.pi) * erf(a / 2))

    def get_A(self):
        """Returns A (for linear nonLande). (1+A) is the amplification of the contribution of
        standing variation to the contribution in mean"""
        return self.get_C() * self.get_eta()

    def get_B(self):
        """Returns B (for linear nonLande). B is factor that appears in the contribution of
        new mutations to the contribution in mean"""
        return self.get_C() * (1.0 - self.get_eta())

    def get_C(self):
        """get the allelic measure of the deviation from/ Lande (amplification)
        assuming a gamma distribution of squared effects"""
        if not self._C_HAS_BEEN_INITIALIZED:
            self._initialize_C()
            return self._C
        else:
            return self._C

    def get_eta(self):
        """Returns ,eta, which is the fraction contribution to the change in mean phenotype
        coming from standing variation."""
        if not self._ETA_HAS_BEEN_INITIALIZED:
            self._initialize_eta()
            return self._eta
        else:
            return self._eta

    @abc.abstractmethod
    def _initialize_C(self):
        """initialize the allelic measure of the deviation from/ Lande (amplification)"""

    @abc.abstractmethod
    def _initialize_eta(self):
        """find an estimate of eta, which is the fraction contribution to the change in mean phenotype
        coming from standing variation. In the Lande case, eta = 1"""

    def _initialize_C_lande(self):
        """Assuming a Lande case, initialize the allelic measure of the deviation from/ Lande (amplification)
        for pure Lande trajectories"""
        self._C_HAS_BEEN_INITIALIZED = True
        self._C = 0

    def _initialize_eta_lande(self):
        """Assuming a Lande case, initialize ,eta, the fraction contribution to the change in mean phenotype
        coming from standing variation. In the Lande case, eta = 1"""
        self._ETA_HAS_BEEN_INITIALIZED = True
        self._eta = 1

class dataClassFullSims(_dataClass):
    """
    Class to read the simulation results in base_dir, if they were generated from from population  simulations
    """
    def __init__(self,base_dir=os.getcwd(), units=None):
        _dataClass.__init__(self,base_dir = base_dir,units=units)

        self._initialise_hstats_for_which_zeros_discounted()
        self._extra_tstats = []

        self._make_theory_curves(emp=self._EMP)

        self._make_hist_bins()
        self._make_histo_labels()

    def set_units(self,units_s0):
        _dataClass.set_units(self,units_s0)
        self.theory_curves.set_units(self._units)

    def get_phase_two_time(self,alpha=None):
        return self._get_phase_two_time_lande(alpha)

    def get_distances_from_lande(self):
        self._get_distances_from_lande()

    def get_distance_dict(self,the_time = None,units=None):
        if units is None:
            units = False
        stat_dict =self.read_bstats('dist',units=units)
        use_dist_guess = False
        if the_time is None:
            the_time = max(stat_dict['dist'].keys()) +1
        elif the_time <= 0:
            the_time = max(stat_dict['dist'].keys()) + 1
        elif the_time >0:
            use_dist_guess = True

        dist_dict = dict()
        for t in stat_dict['dist']:
            if t <= the_time:
                dist_dict[t] = stat_dict['dist'][t][self._summary_values[0]]

        if use_dist_guess:
            stat_dict = self.read_bstats('dist_guess', units=units)
            for t in stat_dict['dist_guess']:
                if t > the_time:
                    dist_dict[t] = stat_dict['dist_guess'][t][self._summary_values[0]]

        return dist_dict

    def get_variance_dict(self,units=None):
        if units is None:
            units = False
        stat_dict = self.read_bstats('var',units=units)
        var_dict = dict()
        for t in stat_dict['var']:
            var_dict[t] = stat_dict['var'][t][self._summary_values[0]]
        return var_dict

    def get_var_theory_curve(self,scale = None):
        var_dict = self.get_variance_dict(units=False)
        times = sorted(var_dict.keys())

        var_list_high = []
        times_high = []
        for t in times:
            #if t >=0:
            times_high.append(t)
            var_list_high.append(var_dict[t])

        if scale is None:
            t_peak, var_peak, t_peak_index = self.theory_curves.get_t_var_peak(times_high,var_list_high)
        else:
            t_peak, var_peak, t_peak_index = self.theory_curves.get_t_var_peak(times_high, var_list_high,scale=scale)

        return t_peak, var_peak, t_peak_index

    def make_parameter_text_population_to_write_file(self):
        namebase = 'N_'+ str(int(self.param_dict['N']))
        namebase += '_U_' + str(self.param_dict['U'])
        namebase += '_Ds0_' + str(int(self.param_dict['shift_s0']))
        namebase +='_E2Ns_' + str(int(self.param_dict['E2Ns'])) +'_V2Ns_' + str(int(self.param_dict['V2Ns']))
        namebase=namebase.replace(".","_")
        return namebase

    def make_parameter_text_to_write_file(self):
        return self.make_parameter_text_population_to_write_file()

    def write_readme_file_for_bstat_text_files(self,savedir):
        time_file_name = "READ_ME_about_stats_in_this_folder"+".txt"
        timefile = os.path.join(savedir, time_file_name)
        tf = open(timefile, "w")
        tf.write('creation my_time : ' + str(datetime.now()) + '\n')
        tf.write("\n")
        tf.write("The trait is assumed to be measured in units of delta = root(Vs/(2N))")
        tf.write("\n")
        tf.write("The first line in each statistic text file are the sampling times "
                 "(the number of generations after the shift).")
        tf.write("\n")
        tf.write("The second line is value of the statistic in at the corresponding time.")
        tf.write("\n")
        tf.write("The third line is standard error of the statistic at the corresponding time.")
        tf.write("\n")
        tf.write("The fourth line is number of runs of population sims that the result for that stat are averaged over.")
        tf.write("\n")
        tf.write("If all stats were averaged over the same number of sims, then that number of sims is: "
                 +str(int(self.number_population_simulations())))
        tf.write("\n")
        tf.close()

    def write_stat_to_text_files(self,bstat,savedir,maxtime =None):

        namebase = bstat
        if bstat not in self._bstats:
            print(bstat, " not in saved stats")
            return

        if not os.path.isdir(savedir):
            print('Directory does not exist')
            return

        savedir = self.set_up_folder_named_by_parameter_text(savedir)
        savefolder = os.path.join(savedir, "bstats")

        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)

        current_units = self.units_s0
        self.set_units(False)
        data = self.read_bstats([bstat])
        self.set_units(current_units)

        times = sorted(data[bstat].keys())
        times = [t for t in times if t >=0]

        if maxtime is not None:
            if maxtime > 0:
                times = [t for t in times if t <=maxtime]

        mean = [data[bstat][tim]['mean'] for tim in times]
        se = [data[bstat][tim]['se'] for tim in times]

        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)

        #Make a readme
        self.write_readme_file_for_bstat_text_files(savefolder)

        #save the relavent number of runs
        num_runs = self._read_number_of_runs_in_combined(bstat)

        thefile = os.path.join(savefolder, namebase + '.txt')
        with open(thefile, 'w') as f:
            for item in times:
                f.write('{:.0f}'.format(item) + " ")
            f.write("\n")
            for item in mean:
                f.write('{:.12f}'.format(item) + " ")
            f.write("\n")
            for item in se:
                f.write('{:.12f}'.format(item) + " ")
            f.write("\n")
            f.write(str(int(num_runs)))

    def write_stat_to_csv_files(self,bstats,savedir,maxtime =None,units_s0=None):

        if isinstance(bstats, str):
            bstats =[bstats]

        multiply_se_by = 1.96

        bstats.sort()
        namebase = "_".join(bstats)
        for bstat in bstats:
            if bstat not in self._bstats:
                print(bstat, " not in saved stats")
                return

        if not os.path.isdir(savedir):
            print('Directory does not exist')
            return

        parameter_string = self.make_parameter_text_to_write_file()

        savefolder = os.path.join(savedir,parameter_string)
        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)

        current_units = self.units_s0
        if units_s0 is None:
            self.set_units(False)
            data = self.read_bstats(bstats)
        else:
            self.set_units(units_s0)
            data = self.read_bstats(bstats)
        self.set_units(current_units)

        times = sorted(data[bstats[0]].keys())
        times = [t for t in times if t >=0]

        if maxtime is not None:
            if maxtime > 0:
                times = [t for t in times if t <=maxtime]

        thefile = os.path.join(savefolder, namebase + '.csv')
        with open(thefile, mode='w',newline='') as f:
            cvsfilewriter = csv.writer(f, dialect='excel')
            headingsfirst = ['']
            headingssecond = ['generation after shift']
            for bstat in bstats:
                headingsfirst.append(bstat)
                headingsfirst.append('')
                headingssecond.append('mean')
                headingssecond.append('1.96*SE')
            cvsfilewriter.writerow(headingsfirst)
            cvsfilewriter.writerow(headingssecond)
            for item in times:
                thetime = '{:.0f}'.format(item)
                therow = [thetime]
                for bstat in bstats:
                    themean = '{:.12f}'.format(data[bstat][item]['mean'])
                    these = '{:.12f}'.format(data[bstat][item]['se'])
                    therow.append(themean)
                    therow.append(multiply_se_by*these)
                cvsfilewriter.writerow(therow)

    def write_sum_two_hbstat_at_time_to_csv_files(self, hbstat1,hbstat2, savedir, bins, time=None, pos =None,multiplier =1.0):

        namebase = hbstat1 + '_plus_'+hbstat2

        multiply_se_by = 1.96

        posnames = dict()
        posnames['pos'] = 'aligned'
        posnames['neg'] = 'opposing'
        posnames['both'] = 'both'

        if time is None or time < 0:
            time = -1
            shorti = "long"
        else:
            shorti = "short"

        if pos is None:
            poses = ['pos','neg','both']
        else:
            poses = [pos]

        if bins[0] == '_':
            bins = bins[1:]
        if bins[0]=='f':
            hbstat1 += "_fbins"
            hbstat2 += "_fbins"
            filie='fbins'
        else:
            hbstat1 += "_efs_bins"
            hbstat2 += "_efs_bins"
            filie = 'efs_bins'

        if hbstat1 not in self._hbstats:
            print(hbstat1, " not in saved stats")
            return
        if hbstat2 not in self._hbstats:
            print(hbstat2, " not in saved stats")
            return

        if not os.path.isdir(savedir):
            print('Directory does not exist')
            return

        parameter_string = self.make_parameter_text_to_write_file()

        current_units = self.units_s0
        self.set_units(False)
        data = self.read_hbstats([hbstat1,hbstat2])
        self.set_units(current_units)
        pairs = self.hist_bin_pairs[hbstat1]
        numkeys = len(data[hbstat1][poses[0]])
        if filie=="efs_bins":
            range_k = numkeys - 2
            offset = 1
        else:
            range_k = numkeys
            offset = 0


        times = [tt for tt in sorted(data[hbstat1][poses[0]][0].keys())]
        if time <0:
            maxtime = max(times)
            mytime = maxtime
        else:
            lande_time = find_nearest(times, time)
            mytime = lande_time

        savefolder = os.path.join(savedir,parameter_string)
        savefolder = os.path.join(savefolder, shorti)
        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)
        #save the relavent time
        timefile = os.path.join(savefolder, str(mytime)+".txt")
        tf = open(timefile, "w")
        tf.close()

        savefolder = os.path.join(savefolder,filie)
        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)

        thefile = os.path.join(savefolder, namebase + '.csv')
        with open(thefile, mode='w',newline='') as f:
            cvsfilewriter = csv.writer(f, dialect='excel')
            headingsfirst = ['','']
            if filie == "efs_bins":
                headingssecond = ['S bins','']
            elif filie == 'fbins':
                headingssecond = ['Frequency bins', '']
            else:
                headingssecond = ['Angle bins', '']
            headingsthird = ['left boundary', 'right boundary']

            headingsfirst.append(hbstat1+ ' plus '+hbstat2)
            for posi in poses:
                headingsfirst.append('')
                headingsfirst.append('')
                headingssecond.append(posnames[posi])
                headingssecond.append('')
                headingsthird.append('mean')
                headingsthird.append('1.96*SE')
            headingsfirst = headingsfirst[:-1]
            cvsfilewriter.writerow(headingsfirst)
            cvsfilewriter.writerow(headingssecond)
            cvsfilewriter.writerow(headingsthird)
            for k in range(range_k):
                itemleft =  pairs[k+offset][0]
                itemright =  pairs[k+offset][1]
                theSleft = '{:.12f}'.format(itemleft)
                theSright= '{:.12f}'.format(itemright)
                therow = [theSleft,theSright]
                for posi in poses:
                    meanplus = data[hbstat1][posi][k + offset][mytime]['mean']+ \
                               data[hbstat2][posi][k + offset][mytime]['mean']
                    sesquareplus = data[hbstat1][posi][k + offset][mytime]['se']**2 + \
                                   data[hbstat2][posi][k + offset][mytime]['se']**2
                    seplus = np.sqrt(sesquareplus)
                    themean = '{:.12f}'.format(meanplus*multiplier)
                    these = '{:.12f}'.format(seplus*multiplier)
                    therow.append(themean)
                    therow.append(multiply_se_by*these)
                cvsfilewriter.writerow(therow)

    def write_hbstat_at_time_to_csv_files(self, hbstats, savedir, bins, time=None, pos =None):
        if isinstance(hbstats, str):
            hbstats =[hbstats]
        hbstats1 = [hb for hb in hbstats]

        multiply_se_by = 1.96

        hbstats1.sort()
        nhs =len(hbstats1)
        namebase = "_".join(hbstats1)

        posnames = dict()
        posnames['pos'] = 'aligned'
        posnames['neg'] = 'opposing'
        posnames['both'] = 'both'


        if time is None or time < 0:
            time = -1
            shorti = "long"
        else:
            shorti = "short"

        if pos is None:
            poses = ['pos','neg','both']
        else:
            poses = [pos]

        if bins[0] == '_':
            bins = bins[1:]
        if bins[0]=='f':
            for k in range(nhs):
                hbstats1[k] += "_fbins"
            filie='fbins'
        else:
            for k in range(nhs):
                hbstats1[k] += "_efs_bins"
            filie = 'efs_bins'

        for hbstat in hbstats1:
            if hbstat not in self._hbstats:
                print(hbstat, " not in saved stats")
                return

        if not os.path.isdir(savedir):
            print('Directory does not exist')
            return

        parameter_string = self.make_parameter_text_to_write_file()

        current_units = self.units_s0
        self.set_units(False)
        data = self.read_hbstats(hbstats1)
        self.set_units(current_units)
        pairs = self.hist_bin_pairs[hbstats1[0]]
        numkeys = len(data[hbstats1[0]][poses[0]])
        if filie=="efs_bins":
            range_k = numkeys - 2
            offset = 1
        else:
            range_k = numkeys
            offset = 0

        times = [tt for tt in sorted(data[hbstats1[0]][poses[0]][0].keys())]
        if time <0:
            maxtime = max(times)
            mytime = maxtime
        else:
            lande_time = find_nearest(times, time)
            mytime = lande_time

        savefolder = os.path.join(savedir,parameter_string)
        savefolder = os.path.join(savefolder, shorti)
        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)
        #save the relavent time
        time_file_name = str(mytime)+".txt"
        timefile = os.path.join(savefolder, time_file_name)
        tf = open(timefile, "w")
        tf.close()

        savefolder = os.path.join(savefolder,filie)
        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)

        thefile = os.path.join(savefolder, namebase + '.csv')
        with open(thefile, mode='w',newline='') as f:
            cvsfilewriter = csv.writer(f, dialect='excel')
            headingsfirst = ['','']
            if filie == "efs_bins":
                headingssecond = ['S bins','']
            elif filie == 'fbins':
                headingssecond = ['Frequency bins', '']
            else:
                headingssecond = ['Angle bins', '']
            headingsthird = ['left boundary', 'right boundary']
            for hbstat in hbstats1:
                headingsfirst.append(hbstat)
                for posi in poses:
                    headingsfirst.append('')
                    headingsfirst.append('')
                    headingssecond.append(posnames[posi])
                    headingssecond.append('')
                    headingsthird.append('mean')
                    headingsthird.append('1.96*SE')
                headingsfirst = headingsfirst[:-1]
            cvsfilewriter.writerow(headingsfirst)
            cvsfilewriter.writerow(headingssecond)
            cvsfilewriter.writerow(headingsthird)
            for k in range(range_k):
                itemleft =  pairs[k+offset][0]
                itemright =  pairs[k+offset][1]
                theSleft = '{:.12f}'.format(itemleft)
                theSright= '{:.12f}'.format(itemright)
                therow = [theSleft,theSright]
                for hbstat in hbstats1:
                    for posi in poses:
                        themean = '{:.12f}'.format(data[hbstat][posi][k + offset][mytime]['mean'])
                        these = '{:.12f}'.format(data[hbstat][posi][k + offset][mytime]['se'])
                        therow.append(themean)
                        therow.append(multiply_se_by*these)
                cvsfilewriter.writerow(therow)

    def write_readme_file_for_hbstat_text_files(self,savedir,time):
        time_file_name = "READ_ME_about_stats_in_this_folder"+".txt"
        timefile = os.path.join(savedir, time_file_name)
        tf = open(timefile, "w")
        tf.write('creation my_time : ' + str(datetime.now()) + '\n')
        tf.write("\n")
        tf.write("The trait is assumed to be measured in units of delta = root(Vs/(2N))")
        tf.write("\n")
        tf.write("The statistics in this folder were recorded at generation "+str(int(time)))
        tf.write("\n")
        tf.write("The first line in each statistic text file is the effect size or frequency bins.")
        tf.write("\n")
        tf.write("The second line is value of the statistic in the corresponding bin.")
        tf.write("\n")
        tf.write("The third line is standard error of the statistic")
        tf.write("\n")
        tf.write("The fourth line is number of runs of population sims that the result for that stat are averaged over.")
        tf.write("\n")
        tf.write("If all stats were averaged over the same number of sims, then that number of sims is: "
                 +str(int(self.number_population_simulations())))
        tf.write("\n")
        tf.close()

    def write_hbstat_at_time_to_text_files(self, hbstat, savedir,bins, time=None,pos =None):

        namebase = hbstat
        if time is None or time < 0:
            time = -1
            shorti = "long"
        else:
            shorti = "short"

        if pos is None:
            poses = ['pos','neg','both']
        else:
            poses = [pos]

        if bins[0] == '_':
            bins = bins[1:]
        if bins[0]=='f':
            hbstat += "_fbins"
            filie='fbins'
        else:
            hbstat += "_efs_bins"
            filie = 'efs_bins'


        if hbstat not in self._hbstats:
            #print(hbstat, " not in saved stats")
            return

        if not os.path.isdir(savedir):
            print('Directory does not exist')
            return

        current_units = self.units_s0
        self.set_units(False)

        data = self.read_hbstats(hbstat)
        self.set_units(current_units)
        pairs = self.hist_bin_pairs[hbstat]
        numkeys = len(data[hbstat][poses[0]])
        if filie=="efs_bins":
            range_k = numkeys - 2
            offset = 1
        else:
            range_k = numkeys
            offset = 0

        times = [tt for tt in sorted(data[hbstat][poses[0]][0].keys())]
        if time <0:
            maxtime = max(times)
            mytime = maxtime
        else:
            lande_time = find_nearest(times, time)
            mytime = lande_time

        savefolder = self.set_up_folder_named_by_parameter_text(savedir)
        savefolder = os.path.join(savefolder, "hbstats_at_time")
        savefolder = os.path.join(savefolder, shorti)
        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)
        #save the relavent time
        self.write_readme_file_for_hbstat_text_files(savefolder,mytime)

        #save the relavent number of runs
        num_runs = self._read_number_of_runs_in_combined(hbstat)
        savefolder = os.path.join(savefolder,filie)
        for posi in poses:
            thefile = os.path.join(savefolder, posi)
            if not os.path.isdir(thefile):
                os.makedirs(thefile)
            thefile = os.path.join(thefile, namebase + '.txt')
            with open(thefile, 'w') as f:
                if bins[0] == 'f':
                    item = 0.0
                    f.write('{:.12f}'.format(item) + " ")
                for pa in pairs[:-1]:
                    item = pa[1]
                    f.write('{:.12f}'.format(item) + " ")
                if bins[0] == 'f':
                    item = 0.5
                    f.write('{:.12f}'.format(item) + " ")
                f.write("\n")
                for k in range(range_k):
                    item = data[hbstat][posi][k + offset][mytime]['mean']
                    f.write('{:.12f}'.format(item) + " ")
                f.write("\n")
                for k in range(range_k):
                    item = data[hbstat][posi][k + offset][mytime]['se']
                    f.write('{:.12f}'.format(item) + " ")
                f.write("\n")
                f.write(str(int(num_runs)))


    def read_bstats(self,requiredstats = None,units=None):

        if isinstance(requiredstats, str):
            requiredstats = [requiredstats]

        if units is None:
            units = True

        bstats = [stat for stat in requiredstats if stat in self._bstats]
        hbstats = [stat for stat in requiredstats if stat in self._hbstats]


        ## [statName][my_time]['mean'\'se']
        data = defaultdict(lambda: defaultdict(dict))
        if bstats:
            basicdata = self._read_bstats(bstats, units)
            for stat in bstats:
                data[stat] = basicdata[stat][0]
                # remove the delta D from the shift
                if stat == 'delta_dist':
                    if 0 in data[stat]:
                        data[stat][0][self._summary_values[0]] = 0.0
                        data[stat][0][self._summary_values[1]] = 0.0
                        data[stat][0][self._summary_values[2]] = 0.0
        elif hbstats:
            # [statname][col][my_time]['mean'\'std']
            data = self._read_bstats(hbstats, units)
        return data



    def read_hbstats(self,requiredstats = None,units=None):
        if requiredstats is None:
            requiredstats = self._hbstats

        if isinstance(requiredstats, str):
            requiredstats = [requiredstats]

        if units is None:
            units = True

        hbstats = [stat for stat in requiredstats if stat in self._hbstats]
        ## [statname][pos/neg/both][col][my_time]['mean'\'std'\'se']
        data = self._read_hbstats(requiredstats=hbstats, units=units)
        return data

    def summarize_bstats(self,requiredstats=None):
        """Pickles a summary for every stat in the base_dir"""
        if requiredstats is None:
            requiredstats = self._stats
        elif isinstance(requiredstats, str):
            requiredstats = [requiredstats]

        for stat in requiredstats:
            self._summarize_bstats(requiredstats=[stat])


    def fit_log_stat_after_phase_I(self,bstat='fixed_state'):

        max_time = self.param_dict['N']*2

        stat_dict = self.read_bstats(bstat)
        times = sorted(stat_dict.keys())
        times = [t for t in times if t > self.phase_3_time and t < max_time]

        stat_high = [stat_dict[t][self._summary_values[0]] for t in times]

        if bstat == 'fixed_state':
            intercept = self.theory_curves.log_fixed_state(stat_high[times[0]])
            logstat = [self.theory_curves.log_fixed_state(fi)-intercept for fi in stat_high]
        else:
            print('need to make fit_log_stat_after_phase_I work for ', bstat)
            return

        index = len(logstat)-1

        halfindex = int(index/2)

        for st in logstat[halfindex:]:
            if st == 0:
                index = logstat[halfindex:].index(st)+halfindex
                break
        if index > 0:
            times, logstat = times[:index], logstat[:index]
            popt, pcov = curve_fit(self._straight_line, times, logstat)
            #popt, pcov = curve_fit(self._straight_line_no_intercept, times, logstat)
            logstat_fit = [self._straight_line(t, *popt) for t in times]
            #logstat_fit = [self._straight_line_no_intercept(t, *popt) for t in times]
            logvar_mean = np.mean(logstat)
            residuals_squared = np.sum([(yi - yfi)**2 for yi, yfi in zip(logstat, logstat_fit)])
            ss_tot = np.sum([(yi - logvar_mean) ** 2 for yi in logstat])
            r2 = 1- residuals_squared/ss_tot
            return times, logstat_fit, popt, pcov, r2
        else:
            print('len times ', len(times))
            print('phase III time ', self.phase_3_time)
            print('index is zero in fit_log_stat_after_phase_I for ', bstat)
            return [times[0]], [stat_high[0]], [], [], 0


    def fit_log_var(self):

        stat_dict = self.read_bstats('var')
        times = sorted(stat_dict.keys())
        times = [t for t in times if t > self.phase_4_time]
        var_high = [stat_dict[t][self._summary_values[0]] for t in times]

        var_units = 1.01*self._VAR_0_EMP/self._units**2
        index = len(var_high)-1
        for vv in var_high:
            if vv < var_units:
                index = var_high.index(vv)
                break
        if index > 0:
            times, var_high = times[:index], var_high[:index]
            logvar = [self.theory_curves.log_var_new(yi) for yi in var_high]
            popt, pcov = curve_fit(self._straight_line, times, logvar)
            logvar_f = [self._straight_line(t, *popt) for t in times]
            logvar_mean = np.mean(logvar)
            residuals_squared = np.sum([(yi - yfi)**2 for yi, yfi in zip(logvar, logvar_f)])
            ss_tot = np.sum([(yi - logvar_mean) ** 2 for yi in logvar])
            r2 = 1- residuals_squared/ss_tot
            return times, logvar_f, popt, pcov, r2
        else:
            return [times[0]], [self.theory_curves.log_var_new(var_high[0])], [], [], 0

    def get_stat_averaging(self,stat = None,mean=None):
        if mean is None:
            mean = True #do we want mean or integral
        if stat is None:
            stat = 'var'
        stat_dict = self.read_bstats(stat)
        times =  sorted(stat_dict[stat].keys())
        y = [stat_dict[stat][tim][self._summary_values[0]] for tim in times]
        ySE = [stat_dict[stat][tim][self._summary_values[2]] for tim in times]
        ystat_mean_or_integral = []
        ystat_SE = []
        zero_index = times.index(0)
        indexi = 0
        for xi in times:
            if xi <= 0:
                ystat_mean_or_integral.append(y[indexi])
                ystat_SE.append(ySE[indexi])
            else:
                if indexi > zero_index:
                    inti = 0
                    inti_SE = 0
                    for indi in range(zero_index, indexi):
                        inti += (times[indi + 1] - times[indi]) * (y[indi] + y[indi + 1]) / 2.0
                        inti_SE += (times[indi + 1] - times[indi]) * (ySE[indi] + ySE[indi + 1]) / 2.0
                    if mean:
                        ystat_mean_or_integral.append(inti / float(xi))
                        ystat_SE.append(inti_SE/ float(xi))
                    else:
                        ystat_mean_or_integral.append(inti)
                        ystat_SE.append(inti_SE)
                else:
                    ystat_mean_or_integral.append(y[indexi])
                    ystat_SE.append(ySE[indexi])
            indexi += 1
        return times, ystat_mean_or_integral, ystat_SE


    def make_distances_from_lande(self):
        self.dist_from_lande_dict = defaultdict(dict)

        # self.dist_from_lande_dict['slope_log_fixed_state']['mean'] = slope_log_fixed_state
        # self.dist_from_lande_dict['slope_log_fixed_state']['SE'] = slope_log_fixed_state_SE

        #max between dist and lande

        #save the dictionary
        self.dist_from_lande_dict = default_to_regular(self.dist_from_lande_dict)
        self.add_summary([self.dist_from_lande_dict],'distance_from_lande_dict')

    def stat_writer_param_dict(self):
        """
        In the base directory there is a pickled dictionary of 'basic parameters'.
        This function reads it and saves the param dict
        """
        file = os.path.join(self._base_dir, 'extra_param_dict_full_sims')
        if not os.path.isfile(file):
            print('basic paramaters not there')
            pass
        iterater = self._load(file)
        param_dict = next(iterater)
        return param_dict


    def delete_read_var0_runs(self):
        """Deletes all the runs associated with var0 emp that have
        already been read and incorporated into a summary.
        """
        var0dir = self._var_emp_full_directory_path()
        if var0dir:
            if os.path.isdir(var0dir):
                read_runs = self._find_files_in_dir_with_ending(var0dir,'_have_read')
                for read_run in read_runs:
                    os.remove(read_run)

    def delete_all_read_bstat_runs(self):
        for stat in self._stats:
            self._delete_read_bstat_runs(stat)

    def _get_phase_two_time_lande(self, alpha=None):

        if 'shift_s0' in self.param_dict and 'var_0' in self.param_dict and 'sigma_0_del' in self.param_dict:
            shifts0 = self.param_dict['shift_s0']
            var_0 = self.param_dict['var_0']
            Vs = self.param_dict['Vs']
            sigma_0_del = self.param_dict['sigma_0_del']
            if self._EMP:
                if 'var_0_emp' in self.param_dict:
                    var_0 = self.param_dict['var_0_emp']

            if shifts0 == 0:
                return 0.0
            if alpha is None:
                alpha = self._ALPHA
            if alpha <= 0:
                return 0
            if self._EMP:
                ltime = -np.log(float(alpha) / float(shifts0 * sigma_0_del))*float(Vs) / var_0
            else:
                ltime = -np.log(float(alpha) / float(shifts0 * sigma_0_del))*float(Vs) / var_0
        else:
            ltime = 1

        return ltime




    def _get_distances_from_lande(self):
        if self._check_if_file_in_base_dir('distance_from_lande_dict'):
            iteratori = self.read_summary('distance_from_lande_dict')
            self.dist_from_lande_dict = next(iteratori)
            print('dist from lande reading')
        else:
            self.make_distances_from_lande()


    def _create_param_dict(self):
        """read the parameters and create the dictionary of parameters"""
        self._read_params_basic()
        self._read_params_full_sims()


    def _read_params_full_sims(self):
        """
        In the base directory there is a pickled dictionary of 'basic parameters'. This function reads it and saves the param dict
        """
        file = os.path.join(self._base_dir, 'extra_param_dict_full_sims')
        if not os.path.isfile(file):
            print('basic paramaters not there')
            pass
        iterater = self._load(file)
        param_dict = next(iterater)
        self.param_dict.update(param_dict)

    def _make_theory_curves(self, emp=None):
        if emp is None:
            emp = self._EMP
        if emp:
            var0 = self._VAR_0_EMP
        else:
            var0 = self._VAR_0_THEORY

        self.theory_curves = TheoryCurves(E2Ns=self.param_dict['E2Ns'], V2Ns=self.param_dict['V2Ns'], N=self.param_dict['N'],
                                          U=self.param_dict['U'], Vs=self.param_dict['Vs'],
                                          shift_s0=self.param_dict['shift_s0'],
                                          var_0=var0, units = self._units)


    def _delete_read_bstat_runs(self,statname):
        """Deletes all the runs associated with a given bstat that have already been
        read and incorporated into a summary.
        """
        if statname not in self._stats:
            return
        else:
            statdir = self._get_bstat_full_directory_path_with_units(statname)
            if statdir:
                if os.path.isdir(statdir) and statname[0]!='.':
                    read_runs = self._find_files_in_dir_with_ending(statdir,'_have_read')
                    for read_run in read_runs:
                        os.remove(read_run)

    def _read_hbstats(self,requiredstats = None,units=None):

        if isinstance(requiredstats, str):
            requiredstats = [requiredstats]

        if units is None:
            units = True

        hbstats = [stat for stat in requiredstats if stat in self._hbstats]
        # [statName][col][my_time]['mean'\'se']
        data = self._read_bstats(hbstats, units)

        # [statname][pos/neg/both][col][my_time]['mean'\'std']
        newdata = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(dict))))
        for stat in list(data.keys()):
            already_both_stat_combined = False
            lencols = len(data[stat])
            halfcols = int(lencols / 2)
            numbins = len(self.hist_bin_pairs[stat])
            if numbins == lencols:
                print('already combined for stat: ', stat)
                already_both_stat_combined= True

            for i in range(numbins):
                for time in list(data[stat][0].keys()):
                    if not already_both_stat_combined:
                        posit = data[stat][halfcols+i][time][self._summary_values[0]]
                        negit = data[stat][halfcols-1-i][time][self._summary_values[0]]
                        newdata[stat][self._traj_starts[0]][i][time][self._summary_values[0]] = posit
                        newdata[stat][self._traj_starts[1]][i][time][self._summary_values[0]] = negit
                        newdata[stat][self._traj_starts[2]][i][time][self._summary_values[0]] = posit + negit
                        for sv in self._summary_values[1:]:
                            posit_er = data[stat][halfcols+i][time][sv]
                            negit_er = data[stat][halfcols-1-i][time][sv]
                            newdata[stat][self._traj_starts[0]][i][time][sv] = posit_er
                            newdata[stat][self._traj_starts[1]][i][time][sv] = negit_er
                            newdata[stat][self._traj_starts[2]][i][time][sv] = np.sqrt(posit_er**2+negit_er**2)
                    else:
                        posit = 0.0
                        negit = 0.0
                        both = data[stat][i][time][self._summary_values[0]]
                        newdata[stat][self._traj_starts[0]][i][time][self._summary_values[0]] = posit
                        newdata[stat][self._traj_starts[1]][i][time][self._summary_values[0]] = negit
                        newdata[stat][self._traj_starts[2]][i][time][self._summary_values[0]] = both
                        for sv in self._summary_values[1:]:
                            posit_er = 0
                            negit_er = 0
                            both_er = data[stat][i][time][sv]
                            newdata[stat][self._traj_starts[0]][i][time][sv] = posit_er
                            newdata[stat][self._traj_starts[1]][i][time][sv] = negit_er
                            newdata[stat][self._traj_starts[2]][i][time][sv] = both_er

        return newdata

    def _read_bstats(self, requiredstats = None,units = None):
        """ returns dictionary [statname][col][my_time]['mean'\'std']. There will only be more than one col
    if the stat is a histogrom (one col for every bin).
    Also returns dictionary with n_runs_dict[statname] = nRuns in summary"""

        if units is None:
            units = True

        # [statname][col][my_time]['mean'\'std']
        data = defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))

        if requiredstats != None:
            stats = [s for s in self._stats if s in requiredstats]
            for s in requiredstats:
                if s not in stats:
                    print(s, 'not in', self._base_dir)
                    return
        else:
            return

        # for each statistic (represented by a subdirectory)
        for statname in stats:
            stat_dir = self._get_bstat_full_directory_path_with_units(statname)
            if os.path.isdir(stat_dir) and statname[0]!='.':
                filedir = os.path.join(stat_dir,self._summary_name)
                #If there is no summary, create one
                if not os.path.isfile(filedir):
                    self._summarize_bstats([statname])
                iterater = self._load(filedir)
                c= 0
                for column in iterater:
                    data[statname][c] = column
                    c +=1
        if units:

            #changing to units of sigma_0 or delta
            data = self._unit_change(data, self._units)

        return data

    def _unit_change(self, data, factor):
        """Applied to data[stat] for the stats in statList, this function changes the units by deviding everythin by factor"""

        statlist = [stati for stati in data]
        doubles = ['var']
        for stat in statlist:
            unit_int = self._unit_dict[stat]
            one_over_this_factor = 1.0/float(factor**unit_int)
            for c in list(data[stat].keys()):
                for time in list(data[stat][c].keys()):
                    for vali in self._summary_values:
                        if vali in data[stat][c][time]:
                            data[stat][c][time][vali]=data[stat][c][time][vali]*one_over_this_factor
                    for vali in doubles:
                        if vali in data[stat][c][time]:
                            data[stat][c][time][vali] = data[stat][c][time][vali] * one_over_this_factor**2


        return data


    def _summarize_bstats(self, requiredstats=None):
        """Creates a pickle file for every stat in requiredStats
         1. 'summary': The first line is
the number of runs in the summary. The second line a dictionary of dict[stat][my_time] = summary
If the stat is for  a histogram the next few lines will be a dictionary (dict[stat][my_time]) for
each column
    """
        # self._read_list_basic_stats()

        data = self._read_all_bstats(requiredstats=requiredstats)
        for stat in list(data.keys()):
            if stat in self._stats:
                stat_dir = self._get_bstat_full_directory_path_with_units(stat)
            else:
                print(stat + ' not in bstats or hbstats')
                return
            file_name = os.path.join(stat_dir, self._summary_name)
            with open(file_name, 'wb') as f:
                for column in list(data[stat].keys()):
                    pickle.dump(data[stat][column], f, protocol=pickle.HIGHEST_PROTOCOL)

    def _read_all_bstats(self, requiredstats=None):
        """ Reads EVERY run in the base directory for each of the requiredStats.

Returns
--------
out: dictionary [statname][col][my_time]['mean'\'std']. There will only be more than one col
    if the stat is a histogrom (one col for every bin)
out: dictionary[statname] = number of runs in the statdirectory
"""

        # [statname][col][my_time]['mean'\'std']
        data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        # # [statname][col]['mean'\'std'] (averaged over times and runs)
        # data_single = defaultdict(lambda:defaultdict(dict))


        if requiredstats != None:
            stats = [s for s in self._stats if s in requiredstats]
            for s in requiredstats:
                if s not in stats:
                    print(s, 'not in', self._base_dir)
                    return
        else:
            print('requiredStats = None')
            return

        # for each statistic (represented by a subdirectory)
        for statname in stats:
            if statname in self._stats:
                statdir = self._get_bstat_full_directory_path_with_units(statname)
            if os.path.isdir(statdir) and statname[0] != '.':
                # print 'starting with the %s statistic'%statname
                # read run logs file by file:
                run_dirs = self._find_files_in_dir_with_ending(statdir, self._run_file_name_end)
                if run_dirs:
                    data_little = self._combine_all_from_run_files(run_dirs)
                    data[statname] = data_little
        return data

    def _save_combined_runs_bstats(self, stat, combined_runs_data):
            """Creates a pickle file for every stat in requiredStats
             1. 'summary': The first line is
    the number of runs in the summary. The second line a dictionary of dict[stat][my_time] = summary
    If the stat is for  a histogram the next few lines will be a dictionary (dict[stat][my_time]) for
    each column
        """
            # self._read_list_basic_stats()

            if stat in self._stats:
                stat_dir = self._get_bstat_full_directory_path_with_units(stat)
            else:
                print(stat + ' not in bstats or hbstats')
                return
            file_name = os.path.join(stat_dir, self._combined_runs_name)
            list_columns = list(combined_runs_data.keys())

            # record how many runs are being combined
            a_column = list_columns[0]
            a_time = next(iter(combined_runs_data[a_column]))
            number_of_runs = combined_runs_data[a_column][a_time][self._run_values[2]]
            self._save_text_file_number_of_runs_in_combined(stat=stat, number_runs=number_of_runs)
            self._number_population_runs_combined = number_of_runs

            # save the combined runs dictionary
            with open(file_name, 'wb') as f:
                for column in list_columns:
                    pickle.dump(default_to_regular(combined_runs_data[column]), f, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_text_file_number_of_runs_in_combined(self,stat,number_runs):
        stat_dir = self._get_bstat_full_directory_path_with_units(stat)
        name = self._recording_number_runs_combined_name
        savefolderfile = os.path.join(stat_dir,name+".txt")
        tf = open(savefolderfile, "w")
        tf.write(str(number_runs))
        tf.write("\n")
        tf.write(str(number_runs) + " runs of the population simulations were combined to create the "+
                 self._combined_runs_name + " file for "+stat)
        tf.close()

    def _read_combined_runs_bstats(self, statname):
        """ returns dictionary [statname][col][my_time]['sum'\'sum_squares/num']. There will only be more than one col
    if the stat is a histogrom (one col for every bin).
    Also returns dictionary with n_runs_dict[statname] = nRuns in summary"""

        table = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        if statname not in self._stats:
            print(statname, 'not in', self._base_dir)
            return

        # for each statistic (represented by a subdirectory)
        stat_dir = self._get_bstat_full_directory_path_with_units(statname)
        if os.path.isdir(stat_dir) and statname[0]!='.':
            filedir = os.path.join(stat_dir,self._combined_runs_name)
            #If there is alread a combined runs, read it
            if os.path.isfile(filedir):
                iterater = self._load(filedir)
                c= 0
                for column in iterater:
                    table[c] = column
                    c +=1

        return table

    def _combine_all_from_run_files(self, run_files=None):

        data_little = defaultdict(lambda: defaultdict(dict))
        table = self._read_all_from_run_files(run_files)

        discount_zeros = False
        #print run_files
        # if run_files is not None:
        #     #print run_files
        #     for hstat in self._hstats_for_which_zeros_discounted:
        #         if hstat in run_files[0]:
        #             discount_zeros = True

        nCols, nTimes = len(table), len(table[0])
        # summarize mean and std for the statistic
        for col in range(nCols):
            for time in list(table[col].keys()):
                sum = table[col][time][self._run_values[0]]
                sum_squares = table[col][time][self._run_values[1]]
                num_runs = table[col][time][self._run_values[2]]
                # if discount_zeros:
                #     num_zeros = table[col][time]['num_zeros']
                #     num_runs-=num_zeros

                mean, var = self._mean_var(sum,sum_squares=sum_squares,num=num_runs)

                data_little[col][time][self._summary_values[0]] = mean
                if num_runs >1:
                    try:
                        data_little[col][time][self._summary_values[1]] = np.sqrt(var)
                        data_little[col][time][self._summary_values[2]] = np.sqrt(var) / np.sqrt(num_runs)
                    except Warning as e:
                        print(e)
                        print("var: ",var)
                        print("mean: ", mean)
                        print("sum, sum_squares, num ", sum, sum_squares, num_runs)
                        print("a run file: ", run_files[0])
                        data_little[col][time][self._summary_values[1]] = 0.0
                        data_little[col][time][self._summary_values[2]] = 0.0
                else:
                    data_little[col][time][self._summary_values[1]]= 0
                    data_little[col][time][self._summary_values[2]] = 0.0


        return data_little


    def _read_all_from_run_files(self, run_files=None):


        if run_files is None:
            run_files = []

        if run_files:
            stati = self._read_statname_from_path_of_run_name(run_files[0])
        else:
            stati = ''
        table = self._read_combined_runs_bstats(stati)

        for run_name in run_files:
            iterater = self._load(run_name)
            try:
                for time, vals in next(iterater)[1].items():
                    #if it is a histo stat
                    if isinstance(vals, list):
                        for col in range(len(vals)):
                            vali = vals[col]
                            table[col][time][self._run_values[0]]+= vali
                            table[col][time][self._run_values[1]] += vali ** 2
                            table[col][time][self._run_values[2]] += 1
                            # if vali == 0:
                            #     table[col][time]['num_zeros'] += 1
                    #if it is not a histo stat
                    else:
                        table[0][time][self._run_values[0]] += vals
                        table[0][time][self._run_values[1]] += vals ** 2
                        table[0][time][self._run_values[2]] += 1
                        # if vals == 0:
                        #     table[0][time]['num_zeros'] += 1
                os.rename(run_name, run_name +'_have_read')
            except StopIteration:
                print('Stop iteration error in:')
                print(run_name)
        if table:
            self._save_combined_runs_bstats(stati,table)
        return table

    def _read_hist_bins(self):
        """
        In the base directory there might be a pickled dictionary of 'hist_bins'. This function reads it and saves the param dict
        Returns a dictionary hist_bin_dict, with keys hist_bin_dict['fbins'/'efsbins'] = list_of_boundaries_between_bins
        """
        file = os.path.join(self._base_dir, 'hist_bins')
        if not os.path.isfile(file):
            print('basic paramaters not there')
            return dict()
        else:
            iterater = self._load(file)
            hist_bins_dict = next(iterater)
            return hist_bins_dict

    def _make_hist_bins(self):

        self.hist_bins = dict()
        self.hist_bin_pairs = dict()
        self.bin_type = dict()

        hist_bin_dict = self._read_hist_bins()
        #print hist_bin_dict

        # Also interested in how many muts in each bin
        self.frac_muts_in_hist_bins = dict()
        self.one_over_mutinput_in_hist_bins = dict()


        if hist_bin_dict:
            for h_bstat in self._hbstats:

                if h_bstat[-5:] == 'fbins':
                    self.hist_bins[h_bstat] = hist_bin_dict['fbins']
                    self.bin_type[h_bstat] = 'f'
                    if self.hist_bins[h_bstat]:
                        self.hist_bins[h_bstat] = [0] + self.hist_bins[h_bstat] + [0.5]

                    self.frac_muts_in_hist_bins[h_bstat] = hist_bin_dict['frac_muts_in_fbins']
                    self.one_over_mutinput_in_hist_bins[h_bstat] = hist_bin_dict['one_over_mutinput_in_fbins']

                elif h_bstat[-8:] == 'efs_bins':
                    self.hist_bins[h_bstat] = hist_bin_dict['efs_bins']
                    self.bin_type[h_bstat] = 'efs'
                    nbins = len(self.hist_bins[h_bstat])
                    max_efs = max(self.hist_bins[h_bstat])
                    min_efs = 0
                    efs_width = 2*(max_efs - min_efs)/float(nbins)
                    max_efs+= efs_width
                    self.hist_bins[h_bstat] = [min_efs] +self.hist_bins[h_bstat]+ [max_efs]

                    self.frac_muts_in_hist_bins[h_bstat] = hist_bin_dict['frac_muts_in_efs_bins']
                    self.one_over_mutinput_in_hist_bins[h_bstat] = hist_bin_dict['one_over_mutinput_in_efs_bins']

                self.hist_bin_pairs[h_bstat] = [[f, s] for f, s in zip(self.hist_bins[h_bstat][:-1],
                                                                       self.hist_bins[h_bstat][1:])]
        else:
            return

    def _make_histo_labels(self):


        #[stat][pos/neg/both]
        self.histo_labels = defaultdict(list)
        self.histo_leg_titles = dict()

        x_epsilon = 0.000001


        for h_bstat in self._hbstats:

            histo_label_pos = []

            list_pairs = self.hist_bin_pairs[h_bstat]
            #print list_pairs

            if h_bstat[-5:] == 'fbins':
                leg_label = 'Frequency range'

                for i in range(len(list_pairs)):
                    histo_label_pos.append('{0:.2f}'.format(list_pairs[i][0]) + ' - ' + '{0:.2f}'.format(list_pairs[i][1]))


            elif h_bstat[-8:] == 'efs_bins' or h_bstat == 'LD_quantiles' or h_bstat == 'r_2_quantiles':

                leg_label = 'Scaled S (2Ns)'
                for i in range(len(list_pairs)):
                    histo_label_pos.append('{0:.1f}'.format(list_pairs[i][0])+' - '+ '{0:.1f}'.format(list_pairs[i][1]))

            else:
                print('Unknown hbstat ', h_bstat)
                return

            self.histo_labels[h_bstat] = histo_label_pos
            self.histo_leg_titles[h_bstat] = leg_label


    def _initialise_hstats_for_which_zeros_discounted(self):
        self._hstats_for_which_zeros_discounted = []
        self._hstats_for_which_zeros_discounted = ['mean_fr_f_fixed','var_fr_f_fixed','mean_fr_f_extinct','mean_efs_fixed','mean_efs_extinct']

# SOME theory
    def scaled_s_above_which_only_one_mutation_per_generation(self):
        """Returns the value of a^2 above which there is on average only
        one new mutation per generation in the populution"""
        V2Ns = self.param_dict['V2Ns']
        E2Ns = self.param_dict['E2Ns']
        N = self.param_dict['N']
        U = self.param_dict['U']
        one_over_mut_input = 1.0 / float(2 * N * U)
        SHAPE_S, SCALE_S = float(E2Ns) ** 2 / float(V2Ns), float(V2Ns) / float(E2Ns)
        S_dist = gamma(SHAPE_S, loc=0., scale=SCALE_S)
        if one_over_mut_input < 1:
            scaled_s = S_dist.ppf(1.0 - one_over_mut_input)
        else:
            scaled_s = 0
        return scaled_s

    def v_x(self, x0):
        """the steady-state phenotypic variance contributed by alleles with MAF x0,
        per unit mutational input (Assuming a GAMMA DISTRIBUTION of squared phenotypic effects)"""
        E2Ns = float(self.param_dict['E2Ns'])
        V2Ns = float(self.param_dict['V2Ns'])
        return 4*E2Ns*(1+V2Ns/E2Ns * x0 * (1 - x0))**(-(E2Ns**2/V2Ns+1))

    def get_integral_v_a(self):
        """integrate the function v(a) over the effect size distribution,
        with the GAMMA DISTRIBUTION of squared effects"""
        E2Ns = float(self.param_dict['E2Ns'])
        V2Ns = float(self.param_dict['V2Ns'])

        firsthypgeo = hyp2f1(1.0,E2Ns**2/V2Ns+1, -0.5,-V2Ns/(4*E2Ns))
        secondhypgeo = hyp2f1(1.0,E2Ns**2/V2Ns+1, +0.5,-V2Ns/(4*E2Ns))
        firstterm = (4*E2Ns+V2Ns)*firsthypgeo
        secterm = 2*(E2Ns*(2+E2Ns)+2*V2Ns)*secondhypgeo
        myintegral = 2*E2Ns/(2*E2Ns**2+V2Ns)*(firstterm-secterm)

        # SHAPE_S, SCALE_S = float(E2Ns) ** 2 / float(V2Ns), float(V2Ns) / float(E2Ns)
        # S_dist = gamma(SHAPE_S, loc=0., scale=SCALE_S)
        # to_integrate = lambda ss: 4.0 * np.sqrt(np.abs(ss)) * dawsn(np.sqrt(np.abs(ss)) / 2.0) * S_dist.pdf(ss)
        # b = S_dist.ppf(0.99999999999999)
        # myintegral = quad(to_integrate, 0, b)[0]
        return myintegral

    def get_integral_f_a(self):
        """integrate the function f(a) over the effect size distribution,
        with the gamma distribution of squared effects"""
        E2Ns = self.param_dict['E2Ns']
        V2Ns = self.param_dict['V2Ns']
        SHAPE_S, SCALE_S = float(E2Ns) ** 2 / float(V2Ns), float(V2Ns) / float(E2Ns)
        S_dist = gamma(SHAPE_S, loc=0., scale=SCALE_S)
        to_integrate = lambda ss: 2 * np.sqrt(np.abs(ss)) ** 3 * np.exp(-ss / 4.0) / (
                    np.sqrt(np.pi) * erf(np.sqrt(np.abs(ss)) / 2)) * S_dist.pdf(ss)
        b = S_dist.ppf(0.99999999999999)
        myintegral = quad(to_integrate, 0, b)[0]
        return myintegral

    def _initialize_C(self):
        """initialize C the allelic measure of the deviation from Lande (amplification)"""
        self._initialize_C_nonlande()

    def _initialize_eta(self):
        """initialze ,eta, which is the fraction contribution to the change in mean phenotype
        coming from standing variation.
        Note that eta can only be obtained from simulations, so the data class will not be able to calculate
        # this unless you collected the statistic "d2ax_frozen_over_shift"
        """
        self._initialize_eta_nonlande()

    def _initialize_C_nonlande(self):
        """initialize the allelic measure of the deviation from/ Lande (amplification)
        assuming a gamma distribution of squared effects"""
        integral_of_v = self.get_integral_v_a()
        integral_of_f = self.get_integral_f_a()
        self._C = integral_of_v / integral_of_f - 1.0

        self._C_HAS_BEEN_INITIALIZED = True

    def _initialize_eta_nonlande(self):
        """find ,eta, which is the fraction contribution to the change in mean phenotype
        coming from standing variation. In the Lande case, eta = 1.
        Note that eta can only be obtained from simulations, so the data class will not be able to calculate
        # this unless you collected the statistic "d2ax_frozen_over_shift"
        """
        stat = 'd2ax_frozen_over_shift'
        stat_dict =self.read_bstats(stat)
        if stat_dict:
            the_time = max(stat_dict[stat].keys())
            eta = stat_dict[stat][the_time][self._summary_values[0]]
        else:
            print('No '+stat + ' was recorded')
            print("Therefore cannot find, eta, the fraction contribution to change in mean\n"
                  "coming from standing variation. So returning eta=1, even though probably\n"
                  "wrong")
            eta = 1.0
        self._ETA_HAS_BEEN_INITIALIZED = True
        self._eta = eta

class dataClassTraj(_dataClass):
    """
    Class to read the simulation results in base_dir, if they were generated from from trajectory simulations
    """
    def __init__(self,base_dir=os.getcwd(), units=None):
        _dataClass.__init__(self,base_dir = base_dir,units=units)

        self._traj_class_set_up()


    def _traj_class_set_up(self):

        self._make_trajectory_dirs_and_get_tuples()

        self._store_lande_and_non_lande_stats()

        self._extra_tstats = []
        if self._THERE_ARE_TRAJ:
            #self._initialise_tfstats_for_amending()
            self._make_percentile_dict()
            # for stat in self._add_amended_tfstat:
            #     if stat in self._tstats:
            #         self._extra_tstats.append(stat+'_over_shift')


    def _store_lande_and_non_lande_stats(self):

        for lande in [True, False]:

            for stat in self._traj_stats:
                for tup in self.tuples:
                    are_there_trajs = self._check_if_lande_or_nonlande(stat=stat, tup=tup, lande=lande)
                    if are_there_trajs:
                        if lande:
                            if stat not in self._traj_stats_lande:
                                self._traj_stats_lande.append(stat)
                        else:
                            if stat not in self._traj_stats_non_lande:
                                self._traj_stats_non_lande.append(stat)
            if lande:
                if self._traj_stats_lande:
                    self._THERE_ARE_TRAJ_LANDE = True
                    self._THERE_ARE_TRAJ = True
            else:
                if self._traj_stats_non_lande:
                    self._THERE_ARE_TRAJ_NON_LANDE = True
                    self._THERE_ARE_TRAJ = True
        if self._THERE_ARE_TRAJ_LANDE and not self._THERE_ARE_TRAJ_NON_LANDE:
            self._LANDE_STATUS = True
        else:
            self._LANDE_STATUS = False


    def _check_if_ending_traj(self, stat, tup, ending):
        # Use 'final_stats' to check for final stats

        are_there_trajs = False
        if tup not in self.tuples:
            print("Tuple ", tup, " not in tuples")
            return are_there_trajs
        if stat not in self._traj_stats:
            print(stat, " not in traj stats")
            return are_there_trajs
        if stat in self._tstats:
            stat_dir = self._get_tstat_full_directory_path_with_units(tup=tup, statname=stat)
            # stat_dir = os.path.join(self._tuple_dirs[tup], stat + '_tsD')
        # dont need this elif
        elif stat in self._htstats:
            stat_dir = self._get_tstat_full_directory_path_with_units(tup=tup, statname=stat)
            # stat_dir = os.path.join(self._tuple_dirs[tup], stat + '_H_tsD')
        elif stat == self._final_stat_dir_name:
            stat_dir = os.path.join(self._tuple_dirs[tup], self._final_stat_dir_name + '_tfD')
        else:
            return are_there_trajs
        run_dirs_lande = self._find_files_in_dir_with_ending(stat_dir, ending)
        if len(run_dirs_lande) >= 1:
            are_there_trajs = True
        return are_there_trajs

    def _check_if_lande_or_nonlande(self, stat, tup, lande=None):
        # Use 'final_stats' to check for final stats
        if lande is None:
            lande = self._LANDE_STATUS
        if self._check_if_lande_or_nonlande_summary(stat=stat, tup=tup, lande=lande):
            return True
        elif self._check_if_lande_or_nonlande_traj_runs_unsummarised(stat=stat, tup=tup, lande=lande):
            return True
        else:
            return False

    def _check_if_lande_or_nonlande_summary(self, stat, tup, lande=None):
        # Use 'final_stats' to check for final stats
        if lande is None:
            lande = self._LANDE_STATUS
        ending = self._summary_name
        if lande:
            ending += self._traj_lande_ending
        are_there_trajs = self._check_if_ending_traj(stat=stat, tup=tup, ending=ending)

        return are_there_trajs

    def _check_if_lande_or_nonlande_traj_runs_unsummarised(self, stat, tup, lande=None):
        # Use 'final_stats' to check for final stats
        if lande is None:
            lande = self._LANDE_STATUS
        ending = self._run_file_name_end
        if lande:
            ending += self._traj_lande_ending

        are_there_trajs = self._check_if_ending_traj(stat=stat, tup=tup, ending=ending)


        return are_there_trajs


    def read_tstats(self, requiredstats, tuples=None, units=None, lande=None):

        if units is None:
            units = True
        if lande is None:
            lande = self._LANDE_STATUS

        if isinstance(requiredstats, str):
            requiredstats = [requiredstats]

        tstatdict = defaultdict(lambda :defaultdict(lambda : defaultdict(lambda: defaultdict(dict))))

        tstats = [stat for stat in requiredstats if stat in self._tstats]

        if lande:
            print("we are in Lande")
            print(tstats)
        else:
            print("no lande")
            print(tstats)
        data = self._read_traj_bstats(tstats, tuples=tuples, units=units, lande=lande)

        if not data:
            print("no data")
        else:
            print("yes data")

        # keys = data.keys()
        if lande:
            print('after _read_traj_bstats')

        if data:
            for tup in tuples:
                # if lande:
                #     print tup
                for stat in tstats:
                    #print data.keys()

                    tstatdict[tup][stat] = data[tup][stat][0]

        return tstatdict


    def read_tfstats(self, tuples=None, lande=None, units=None):
        if lande is None:
            lande = self._LANDE_STATUS
        return self._read_traj_fstats(tuples=tuples, lande=lande, units=units)

    def make_parameter_text_lande_to_write_file(self):
        namebase = 'N_'+ str(int(self.param_dict['N']))
        namebase += '_Ds0_' + str(int(self.param_dict['shift_s0']))
        s0_del = self.param_dict['sigma_0_del']
        if int(s0_del) == s0_del:
            s0_string = str(int(self.param_dict['sigma_0_del']))
        elif int(10*s0_del) == 10*s0_del:
            s0_string = str(round(self.param_dict['sigma_0_del'],1))
        else:
            s0_string = str(round(self.param_dict['sigma_0_del'], 2))
        namebase +='_sigma_0_del_' + s0_string
        namebase=namebase.replace(".","_")
        return namebase

    def make_parameter_text_to_write_file(self):
        return self.make_parameter_text_lande_to_write_file()

    def set_up_folder_named_by_parameter_text_lande(self,savedir):
        parameter_string = self.make_parameter_text_lande_to_write_file()
        savefolder = os.path.join(savedir,parameter_string)
        return self.make_identifiers(savefolder)

    #Now
    def write_readme_file_for_tstat_text_files(self,savedir,time,lande):
        if not os.path.exists(savedir):
            os.makedirs(savedir)
            print(savedir)
        if time < 0:
            final  = True
        else:
            final = False
        if final:
            time_file_name = "READ_ME_about_final_time_stats_in_this_folder.txt"
        else:
            time_file_name = "READ_ME_about_time_stats_in_this_folder.txt"
        timefile = os.path.join(savedir, time_file_name)
        tf = open(timefile, "w")
        tf.write('creation my_time : ' + str(datetime.now()) + '\n')
        tf.write("\n")
        if final:
            tf.write("The statistics in this folder were recorded from the trajectory simulations")
            tf.write(" after every allele was fixed or lost ")
        else:
            tf.write("The statistics in this folder were recorded at generation " + str(time))
        tf.write("\n")
        tf.write("The trait is assumed to be measured in units of delta = root(Vs/(2N))")
        tf.write("\n")
        tf.write("The first line in each statistic text file is the effect sizes")
        tf.write("\n")
        tf.write("The second line is value of the statistic for alleles of the corresponding effect size")
        tf.write("\n")
        tf.write("The third line is the standard error of the statistic")
        tf.write("\n")
        tf.write("The statistics were generated with the trajectory simulations")
        tf.write("\n")
        tf.write("The fourth line is number of runs of allele trajectories for each pair of aligned and opposing "
                 "effect sizes that the result for that stat are averaged over.")
        tf.write("\n")
        tf.write("If all stats for all tuples were averaged over the same number of sims, then that number of sims is: "
                 +str(int(self.number_allele_pairs(lande=lande,final=final))))
        tf.close()

    def write_tfstat_efs_to_text_files(self, savedir, tfstat=None, tuple_time=0, pos =None, lande=True):

        # all_tfstats is true if we want to record all the tfstats
        all_tftsats = False
        if tfstat is None:
            all_tftsats = True

        mytups = [tup for tup in self.tuples if tup[2] == tuple_time]
        mytups = sorted(mytups, key=itemgetter(0))

        if lande:
            if not self._THERE_ARE_FINAL_TRAJ_LANDE:
                return
        else:
            if not self._THERE_ARE_FINAL_TRAJ_NON_LANDE:
                return

        shorti = "final"

        if pos is None:
            poses = ['pos','neg','both']
        else:
            poses = [pos]

        if tuple_time == 0:
            filie = 'standing'
        elif tuple_time<0:
            filie = 'new'
        else:
            filie = 'new_time_'+str(int(tuple_time))

        if not os.path.isdir(savedir):
            print('Directory does not exist')
            return

        if lande:
            savefolder = self.set_up_folder_named_by_parameter_text_lande(savedir)
        else:
            savefolder = self.set_up_folder_named_by_parameter_text(savedir)

        current_units = self.units_s0
        self.set_units(False)
        data = self.read_tfstats(tuples=mytups, lande=lande, units=True)

        if not data:
             return

        self.set_units(current_units)

        if all_tftsats:
            namebase_list = data[mytups[0]].keys()
        else:
            namebase_list = [tfstat]

        savefolder = os.path.join(savefolder, "tstats")
        savefolder = os.path.join(savefolder, shorti)
        savefolder = os.path.join(savefolder,filie)

        self.write_readme_file_for_tstat_text_files(savefolder,lande=lande,time=-1)

        for tfstati in namebase_list:
            for posi in poses:
                thefile = os.path.join(savefolder, posi)
                if not os.path.isdir(thefile):
                    os.makedirs(thefile)
                thefile = os.path.join(thefile, tfstati + '.txt')
                with open(thefile, 'w') as f:
                    for tupi in mytups:
                        item = tupi[0]
                        f.write('{:.12f}'.format(item) + " ")
                    f.write("\n")
                    for tupi in mytups:
                        item = data[tupi][tfstati][posi]['mean']
                        f.write('{:.12f}'.format(item) + " ")
                    f.write("\n")
                    for tupi in mytups:
                        item = data[tupi][tfstati][posi]['se']
                        f.write('{:.12f}'.format(item) + " ")
                    f.write("\n")
                    for tupi in mytups:
                        item = self._read_number_of_allele_pairs_in_combined(tup=tupi,stat=self._final_stat_dir_name,lande=lande )
                        f.write(str(int(item)) + " ")


    def write_tfstat_efs_to_csv_files(self, tfstats, savedir, tuple_time=0, pos =None, lande=True, multiplier=1.0):

        if isinstance(tfstats, str):
            tfstats =[tfstats]

        multiply_se_by = 1.96

        tfstats.sort()
        namebase = "_".join(tfstats)

        posnames = dict()
        posnames['pos'] = 'aligned'
        posnames['neg'] = 'opposing'
        posnames['both'] = 'both'
        mytups = [tup for tup in self.tuples if tup[2] == tuple_time]
        mytups = sorted(mytups, key=itemgetter(0))


        shorti = "final"

        if pos is None:
            poses = ['pos','neg','both']
        else:
            poses = [pos]

        if tuple_time == 0:
            filie = 'standing'
        elif tuple_time<0:
            filie = 'new'
        else:
            filie = 'new_time_'+str(int(tuple_time))

        if not os.path.isdir(savedir):
            print('Directory does not exist')
            return

        if lande:
            parameter_string = self.make_parameter_text_lande_to_write_file()
        else:
            parameter_string = self.make_parameter_text_to_write_file()

        current_units = self.units_s0
        self.set_units(False)
        data = self.read_tfstats(tuples=mytups, lande=lande, units=True)
        self.set_units(current_units)

        savefolder = os.path.join(savedir,parameter_string)
        savefolder = os.path.join(savefolder, shorti)
        savefolder = os.path.join(savefolder,filie)

        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)

        thefile = os.path.join(savefolder, namebase + '.csv')
        with open(thefile, mode='w',newline='') as f:
            cvsfilewriter = csv.writer(f, dialect='excel')
            headingsfirst = ['']
            headingssecond = ['']
            headingsthird = ['S']
            for tfstat in tfstats:
                headingsfirst.append(tfstat)
                for posi in poses:
                    headingsfirst.append('')
                    headingsfirst.append('')
                    headingssecond.append(posnames[posi])
                    headingssecond.append('')
                    headingsthird.append('mean')
                    headingsthird.append('1.96*SE')
                headingsfirst = headingsfirst[:-1]
            cvsfilewriter.writerow(headingsfirst)
            cvsfilewriter.writerow(headingssecond)
            cvsfilewriter.writerow(headingsthird)
            for tupi in mytups:
                item = tupi[0]
                theS = '{:.12f}'.format(item)
                therow = [theS]
                for tfstat in tfstats:
                    for posi in poses:
                        themean = '{:.12f}'.format(data[tupi][tfstat][posi]['mean']*multiplier)
                        these = '{:.12f}'.format(multiply_se_by*data[tupi][tfstat][posi]['se']*multiplier)
                        therow.append(themean)
                        therow.append(these)
                cvsfilewriter.writerow(therow)


    def write_tstat_efs_at_time_to_text_files(self, savedir, tstat=None, tuple_time=0, time=None, pos =None, lande=True):

        if tstat is None:
            tstat_list = self._tstats
        else:
            if tstat not in self._tstats:
                print(tstat, " not in saved stats")
                return
            tstat_list = [tstat]

        if lande:
            if not self._THERE_ARE_TRAJ_LANDE:
                return
        else:
            if not self._THERE_ARE_TRAJ_NON_LANDE:
                return


        mytups = [tup for tup in self.tuples if tup[2] == tuple_time]
        mytups = sorted(mytups, key=itemgetter(0))

        if time is None or time < 0:
            time == -1
            shorti = "long"
        else:
            shorti = "short"

        if pos is None:
            poses = ['pos','neg','both']
        else:
            poses = [pos]

        if tuple_time == 0:
            filie = 'standing'
        elif tuple_time<0:
            filie = 'new'
        else:
            filie = 'new_time_'+str(int(tuple_time))

        if not os.path.isdir(savedir):
            print('Directory does not exist')
            return

        if lande:
            savefolder = self.set_up_folder_named_by_parameter_text_lande(savedir)
        else:
            savefolder = self.set_up_folder_named_by_parameter_text(savedir)

        current_units = self.units_s0
        self.set_units(False)
        data = self.read_tstats(tuples= mytups, requiredstats=tstat_list,lande=lande)
        self.set_units(current_units)


        if not data:
            return

        times = [tt for tt in sorted(data[mytups[0]][tstat_list[0]].keys())]
        if time <0:
            maxtime = max(times)
            mytime = maxtime
        else:
            lande_time = find_nearest(times, time)
            mytime = lande_time

        savefolder = os.path.join(savefolder,"tstats")
        savefolder = os.path.join(savefolder, shorti)
        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)

        # save the relavent time and other details in a readme
        self.write_readme_file_for_tstat_text_files(savefolder, lande=lande, time=mytime)

        savefolder = os.path.join(savefolder,filie)


        for tstati in tstat_list:
            namebase = tstati
            for posi in poses:
                thefile = os.path.join(savefolder, posi)
                if not os.path.isdir(thefile):
                    os.makedirs(thefile)
                thefile = os.path.join(thefile, namebase + '.txt')
                with open(thefile, 'w') as f:
                    for tupi in mytups:
                        item = tupi[0]
                        f.write('{:.12f}'.format(item) + " ")
                    f.write("\n")
                    for tupi in mytups:
                        item = data[tupi][tstati][mytime][posi]['mean']
                        f.write('{:.12f}'.format(item) + " ")
                    f.write("\n")
                    for tupi in mytups:
                        item = data[tupi][tstati][mytime][posi]['se']
                        f.write('{:.12f}'.format(item) + " ")
                    f.write("\n")
                    for tupi in mytups:
                        item = self._read_number_of_allele_pairs_in_combined(tup=tupi,stat=tstati,lande=lande )
                        f.write(str(int(item)) + " ")


    def write_tstat_efs_at_time_to_csv_files(self, tstats, savedir, tuple_time=0, time=None, pos =None, lande=True, multiplier=1.0):

        if isinstance(tstats, str):
            tstats =[tstats]

        tstats.sort()
        namebase = "_".join(tstats)

        multiply_se_by = 1.96

        posnames = dict()
        posnames['pos'] = 'aligned'
        posnames['neg'] = 'opposing'
        posnames['both'] = 'both'
        mytups = [tup for tup in self.tuples if tup[2] == tuple_time]
        mytups = sorted(mytups, key=itemgetter(0))

        if time is None or time < 0:
            time == -1
            shorti = "long"
        else:
            shorti = "short"

        if pos is None:
            poses = ['pos','neg','both']
        else:
            poses = [pos]

        if tuple_time == 0:
            filie = 'standing'
        elif tuple_time<0:
            filie = 'new'
        else:
            filie = 'new_time_'+str(int(tuple_time))

        for tstat in tstats:
            if tstat not in self._tstats:
                print(tstat, " not in saved stats")
                return

        if not os.path.isdir(savedir):
            print('Directory does not exist')
            return

        if lande:
            parameter_string = self.make_parameter_text_lande_to_write_file()
        else:
            parameter_string = self.make_parameter_text_to_write_file()

        current_units = self.units_s0
        self.set_units(False)
        data = self.read_tstats(tuples= mytups, requiredstats=tstats)
        self.set_units(current_units)

        times = [tt for tt in sorted(data[mytups[0]][tstats[0]].keys())]
        if time <0:
            maxtime = max(times)
            mytime = maxtime
        else:
            lande_time = find_nearest(times, time)
            mytime = lande_time

        savefolder = os.path.join(savedir,parameter_string)
        savefolder = os.path.join(savefolder, shorti)
        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)
        #save the relavent time
        time_file_name = str(mytime)+".txt"
        timefile = os.path.join(savefolder, time_file_name)
        tf = open(timefile, "w")
        tf.write("The statistics in this folder were recorded at generation "+str(mytime))
        tf.close()

        savefolder = os.path.join(savefolder,filie)
        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)

        thefile = os.path.join(savefolder, namebase + '.csv')
        with open(thefile, mode='w',newline='') as f:
            cvsfilewriter = csv.writer(f, dialect='excel')
            headingsfirst = ['']
            headingssecond = ['']
            headingsthird = ['S']
            for tstat in tstats:
                headingsfirst.append(tstat)
                for posi in poses:
                    headingsfirst.append('')
                    headingsfirst.append('')
                    headingssecond.append(posnames[posi])
                    headingssecond.append('')
                    headingsthird.append('mean')
                    headingsthird.append('1.96*SE')
                headingsfirst = headingsfirst[:-1]
            cvsfilewriter.writerow(headingsfirst)
            cvsfilewriter.writerow(headingssecond)
            cvsfilewriter.writerow(headingsthird)
            for tupi in mytups:
                item = tupi[0]
                theS = '{:.12f}'.format(item)
                therow = [theS]
                for tstat in tstats:
                    for posi in poses:
                        themean = '{:.12f}'.format(data[tupi][tstat][mytime][posi]['mean']*multiplier)
                        these = '{:.12f}'.format(multiply_se_by*data[tupi][tstat][mytime][posi]['se']*multiplier)
                        therow.append(themean)
                        therow.append(these)
                cvsfilewriter.writerow(therow)



    def summarize_all_traj(self):
        """Pickles a summary for every stat in the base_dir"""
        for lande in [False,True]:
            if self.tuples is not None:
                for tup in self.tuples:
                    for stat in self._traj_stats:
                        if self._check_if_lande_or_nonlande_traj_runs_unsummarised(stat, tup, lande):
                            self._summarize_traj_bstats(requiredstats = [stat], tuples=[tup], lande=lande)
            else:
                print('No attribute tuples for '+self._base_dir)

    def summarize_traj_final(self):
        """Pickles a summary for every stat in the base_dir"""
        for lande in [False,True]:
            if self.tuples is not None:
                self._summarize_final_traj_bstats(self.tuples, lande=lande)
            else:
                print('No attribute tuples for '+self._base_dir)


    def stat_writer_param_dict(self):
        """
        In the base directory there are pickled dictionaries used to create the hash in the statwriter.
        This function reads it
        """
        stat_param_dict = dict()
        files = [os.path.join(self._base_dir, 'param_dict_basic')]#, os.path.join(self._base_dir, 'extra_param_dict_full_sims')]
        for file in files:
            if os.path.isfile(file):
                iterater = self._load(file)
                stat_param_dict.update(next(iterater))
            else:
                print('some paramaters not there')
        return stat_param_dict

    def delete_all_read_tstat_runs(self):
        for tup in self.tuples:
            self._delete_read_final_tstat_runs(tup)
            for stat in self._tstats:
                self._delete_read_tstat_runs(statname=stat, tup=tup)

    def final_tdir(self, tup):
        """Finds the full path of the final tstats

        Returns: A string with the name of the path of the directory containing the final tstats
        """
        dir = self._tuple_dirs[tup]
        dir = os.path.join(dir,'final_stats_tfD')
        return dir


    def _create_param_dict(self):
        """read the parameters and create the dictionary of parameters"""
        self._read_params_basic()
        self._read_params_traj()


    def _read_params_traj(self):
        """
        In the base directory there is a pickled dictionary of 'basic parameters'. This function reads it and saves the param dict
        """
        file = os.path.join(self._base_dir, 'param_dict_traj')
        if not os.path.isfile(file):
            print('traj paramaters not there')
            pass
        else:
            iterater = self._load(file)
            self.param_dict.update(next(iterater))
        print('reading param dict')
        self.param_dict['var_0_emp']=self.param_dict['var_0']


    def _make_trajectory_dirs_and_get_tuples(self):
        """
        Makes the following attributes
            trajectories: (boolean) True if there are trajs. False otherwise
            _traj_directory: (string) The directory contain the trajectory stuff
            tuples: (set) A set of all the tuples (XI, S, t) on which we have traj dirs
            _tuple_dirs: (dict) Keys are tuples and values are the dirs contain traj info
                for that tup
            _htstats: (list) histo stats for trajectories (none at the moment)
            _tstats: (list) basic trajectory stats
            _traj_stats: (set) A set of all the trajectory stats
        """
        base_dir = self._base_dir
        traj_directory = os.path.join(base_dir, self._trajectory_dir_name)

        if os.path.exists(traj_directory):
            self._traj_directory = traj_directory
            self._tuple_dirs = dict()
            tuple_types = os.listdir(self._traj_directory)
            tuple_types = [tr for tr in tuple_types if len(tr) > 3]
            tuple_types = [tuptype for tuptype in tuple_types if tuptype[-3:] == '_tD']
            if tuple_types:
                self._THERE_ARE_TRAJ = True
                for tuple_type in tuple_types:
                    tup = self._read_tuple_from_directory_name(tuple_type)
                    S, XI, time = tup
                    if S > 0:
                        self.tuples.add(tup)
                        dir = os.path.join(self._traj_directory, tuple_type )
                        self._tuple_dirs[tup] = dir
                a_tup = self._read_tuple_from_directory_name(tuple_types[0])
                trajectory_stats_dir_list = os.listdir(self._tuple_dirs[a_tup])
                for stat_dir in trajectory_stats_dir_list:
                    self._store_stat_name_and_data_from_directoryname(stat_dir)
                if self._THERE_ARE_FINAL_TRAJ:
                    a_tup = list(self.tuples)[0]
                    for lande in [True,False]:
                        # when we read the final traj stats, we also record them
                        # that is the purpose of the below line
                        self._read_traj_fstats(tuples=[a_tup], lande=lande)
        else:
            print('No trajectories in ' + base_dir)
        self._read_number_runs_of_allele_pair_simulations()

    def _delete_read_tstat_runs(self, statname, tup):
        """Deletes all the runs associated with a given tstat that have already been
        read and incorporated into a summary.
        """
        if statname not in self._traj_stats:
            return
        else:
            statdir = self._get_tstat_full_directory_path_with_units(statname, tup)
            if statdir:
                if os.path.isdir(statdir) and statname[0]!='.':
                    read_runs = self._find_files_in_dir_with_ending(statdir,'_have_read')
                    for read_run in read_runs:
                        os.remove(read_run)

    def _delete_read_final_tstat_runs(self, tup):
        """Deletes all the runs associated with a given tstat that have already been
        read and incorporated into a summary.
        """

        statdir = self.final_tdir(tup)
        if statdir:
            if os.path.isdir(statdir):
                read_runs = self._find_files_in_dir_with_ending(statdir,'_have_read')
                for read_run in read_runs:
                    os.remove(read_run)


    def _get_tstat_full_directory_path_with_units(self, statname, tup):
        """Finds the full path of a given tstat

        Returns: A string with the path name
        """
        front = 'U' + str(int(self._get_units_of_stat(statname))) + '_'
        if statname in self._tstats:
            stat_dir = os.path.join(self._tuple_dirs[tup], front + statname + '_tsD')
        elif statname in self._htstats:
            stat_dir = os.path.join(self._tuple_dirs[tup], front + statname + '_H_tsD')
        else:
            stat_dir = ''
        return stat_dir


    def _get_xpercentile(self, tup):
        s, xi, _ = tup
        N = self.param_dict['N']
        if xi > 0:
            freqdist = FreqMinorDistrE(S=s, N=N)
            try:
                perc = freqdist.cdf(xi)
            except ValueError:
                perc = 0
            return perc
        else:
            return xi


    def _get_spercentile(self, tup):
        s, xi, _ = tup
        try:
            E2Ns = self.param_dict['E2Ns']
            V2Ns = self.param_dict['V2Ns']
            shape = float(E2Ns) ** 2 / float(V2Ns)
            scale = float(V2Ns) / float(E2Ns)
            effect_size_dist = gamma(shape, loc=0., scale=scale)
            try:
                perc = effect_size_dist.cdf(s)
            except ValueError:
                perc = 0
        except KeyError:
            perc = 0
        return perc

    def _make_percentile_dict(self):
        self.xpercentile_dict = dict()
        self.spercentile_dict = dict()
        for tup in self.tuples:
            self.xpercentile_dict[tup] = self._get_xpercentile(tup)
            self.spercentile_dict[tup] = self._get_spercentile(tup)

    def _read_traj_bstats(self, requiredstats = None, tuples=None, lande=None, units = None):
        """ returns dictionary [statname][col][my_time]['mean'\'std']. There will only be more than one col
    if the stat is a histogrom (one col for every bin).
    Also returns dictionary with n_runs_dict[statname] = nRuns in summary"""

        if units is None:
            units = True
        if lande is None:
            lande = self._LANDE_STATUS

        sum_file_name = self._summary_name
        if lande:
            sum_file_name += self._traj_lande_ending

        # [statname][col][my_time]['pos'/'neg'/'both']['mean'\'std'/'se']
        data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(dict))))

        if requiredstats != None:
            stats = [s for s in self._traj_stats if s in requiredstats]
            for s in requiredstats:
                if s not in stats:
                    print(s, 'not in', self._base_dir)
                    return
        else:
            print('required stats are None')
            return

        if tuples != None:
            the_tuples = [s for s in tuples if s in self.tuples]
            for s in tuples:
                if s not in the_tuples:
                    print(s, 'not in', self._base_dir)
                    return
        else:
            print('tuples are None')
            return

        first = True
        # for each statistic (represented by a subdirectory)
        for tuple in tuples:
            for statname in stats:
                continue_loops = True
                if statname in self._tstats:
                    stat_dir = self._get_tstat_full_directory_path_with_units(tup=tuple, statname=statname)
                elif statname in self._htstats:
                    stat_dir = self._get_tstat_full_directory_path_with_units(tup=tuple, statname=statname)
                if os.path.isdir(stat_dir) and statname[0]!='.':
                    filedir = os.path.join(stat_dir,sum_file_name)
                    #If there is no summary, create one
                    if not os.path.isfile(filedir):
                        print(filedir + ' not there yet')
                        if self._check_if_lande_or_nonlande_traj_runs_unsummarised(statname, tuple, lande=lande): #check if they exist
                            self._summarize_traj_bstats(requiredstats=[statname], tuples=[tuple], lande=lande)
                        else:
                            continue_loops = False #if they don't stop
                    if continue_loops:
                        #print 'loading file dir for tup ', tup
                        iterater = self._load(filedir)
                        #print 'loaded file dir for tup ', tup
                        c= 0
                        for column in iterater:
                            data[tuple][statname][c] = column
                            c +=1
                    else:
                        print('not continuing loops ', statname)
        if data:
            if units:
                #changing to units of sigma_0
                for tuple in tuples:
                    data[tuple] = self._unit_change_traj(data[tuple], self._units)
        return data


    def _read_traj_fstats(self, tuples=None, lande=None, units = None):
        """ returns dictionary [statname][tup]['pos'/'neg'/'both']['mean'\'std'\'se']. There will only be more than one col
    if the stat is a histogrom (one col for every bin).
    Also returns dictionary with n_runs_dict[statname] = nRuns in summary"""


        if lande is None:
            lande = self._LANDE_STATUS
        if units is None:
            units = True

        summary_name = self._summary_name
        if lande:
            summary_name+= self._traj_lande_ending

        # [tup][statname]['pos'/'neg'/'both']['mean'\'var','SE']
        final_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        # [tup][statbody]['pos'/'neg'/'both']['mean'\'var','SE']
        final_dict_statbody = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        stats_found = set()

        if tuples != None:
            the_tuples = [s for s in tuples if s in self.tuples]
            for s in tuples:
                if s not in the_tuples:
                    print(s, 'not in', self._base_dir)
                    return
        else:
            the_tuples = self.tuples

        # for each statistic (represented by a subdirectory)
        for tuple in the_tuples:
            continue_loops = True
            stat_dir = self.final_tdir(tuple)

            if os.path.isdir(stat_dir):

                filedir = os.path.join(stat_dir,summary_name)

                #If there is no summary, create one
                if not os.path.isfile(filedir):
                    if lande:
                        if self._check_if_lande_or_nonlande_traj_runs_unsummarised(self._final_stat_dir_name, tuple, lande=True):  # check if they exist
                            self._summarize_final_traj_bstats(tuples=[tuple])
                        else:
                            continue_loops = False  # if they don't stop
                    else:
                        if self._check_if_lande_or_nonlande_traj_runs_unsummarised(self._final_stat_dir_name, tuple, lande=False):  # check if they exist
                            self._summarize_final_traj_bstats(tuples=[tuple])
                        else:
                            continue_loops = False  # if they don't stop


                if continue_loops:
                    iterater = self._load(filedir)

                    final_dict[tuple] = next(iterater)

        # print 'tups ', final_dict.keys()
        # print 'stats ', final_dict[final_dict.keys()[0]]

        for tuple in the_tuples:
            for stat in final_dict[tuple]:
                units, statbody = float(stat[1]), stat[3:]
                # If we have never read the units for this stat, read them
                if statbody not in self._unit_dict:
                    self._unit_dict[statbody] = units
                final_dict_statbody[tuple][statbody] = final_dict[tuple][stat]

        # if the dictionary is empty
        if final_dict_statbody:
            #final_dict = self._add_missing_fstats(final_dict)
            if lande:
                self._THERE_ARE_FINAL_TRAJ_LANDE = True
            else:
                self._THERE_ARE_FINAL_TRAJ_NON_LANDE = True
            if units:
                #changing to units of sigma_0 or delta
                for tuple in the_tuples:
                    final_dict_statbody[tuple] = self._unit_change_traj_final(final_dict_statbody[tuple], self._units)

            for tuple in final_dict_statbody:
                for st in final_dict_statbody[tuple]:
                    stats_found.add(st)

            # Record what the final stats are
            for st in stats_found:
                if st not in self._ftstats:
                    self._ftstats.append(st)
                if lande:
                    if st not in self._ftstats_stats_lande:
                        self._ftstats_stats_lande.append(st)
                else:
                    if st not in self._ftstats_stats_non_lande:
                        self._ftstats_stats_non_lande.append(st)

        return final_dict_statbody


    def _unit_change_traj(self, data, factor):
        """Applied to data[stat] for the stats in statList, this function changes the units by deviding everythin by factor"""

        statlist = list(data.keys())
        doubles = ['var']
        for stat in statlist:
            unit_int = self._unit_dict[stat]
            one_over_this_factor = 1.0/float(factor**unit_int)
            for c in list(data[stat].keys()):
                for time in list(data[stat][c].keys()):
                    for st in self._traj_starts:
                        for vali in self._summary_values:
                            data[stat][c][time][st][vali]=data[stat][c][time][st][vali]*one_over_this_factor
                        # for vali in doubles:
                        #     data[stat][c][time][st][vali] = data[stat][c][time][st][vali] * one_over_this_factor**2
        return data


    def _unit_change_traj_final(self, final_data, factor):
        """Applied to final_data[stat] for the stats in statList, this function changes the units by deviding everythin by factor"""

        statlist = list(final_data.keys())
        doubles = ['var']
        for stat in statlist:
            unit_int = self._unit_dict[stat]
            one_over_this_factor = 1.0/float(factor**unit_int)
            for st in self._traj_starts:
                for vali in self._summary_values:
                    final_data[stat][st][vali]= final_data[stat][st][vali] * one_over_this_factor
                for vali in doubles:
                    if vali in final_data[stat][st]:
                        final_data[stat][st][vali] = final_data[stat][st][vali] * one_over_this_factor ** 2
        return final_data

    def _summarize_traj_bstats(self, requiredstats=None, tuples=None, lande=None):
        """Creates a pickle file for every stat in requiredStats
         1. 'summary': The first line is
the number of runs in the summary. The second line a dictionary of dict[stat][my_time] = summary
If the stat is for  a histogram the next few lines will be a dictionary (dict[stat][my_time]) for
each column
    """
        if lande is None:
            lande = self._LANDE_STATUS

        data = self._read_all_traj_bstats(requiredstats=requiredstats, tuples=tuples, lande=lande)

        summary_name = self._summary_name
        if lande:
            summary_name += self._traj_lande_ending

        for tup in data:
            for stat in data[tup]:
                if stat in self._tstats:
                    stat_dir =self._get_tstat_full_directory_path_with_units(tup=tup, statname=stat)
                elif stat in self._htstats:
                    stat_dir = self._get_tstat_full_directory_path_with_units(tup=tup, statname=stat)
                else:
                    print(stat, ' not in traj stats for ', tup)
                    return
                self._summarize_columns(dict=data[tup][stat], dir=stat_dir,file_name=summary_name)


    def _summarize_final_traj_bstats(self, tuples=None, lande=None):
        """Creates a pickle file for every stat in requiredStats
         1. 'summary': The first line is
    the number of runs in the summary. The second line a dictionary of dict[stat][my_time] = summary
If the stat is for  a histogram the next few lines will be a dictionary (dict[stat][my_time]) for
each column
    """
        # self._read_list_basic_stats()
        if tuples is None:
            tuples = self.tuples
        if lande is None:
            lande = self._LANDE_STATUS

        final_dict= self._read_all_final_traj_bstats(tuples=tuples, lande=lande)

        if lande:
            if self._ftstats_stats_lande:
                self._THERE_ARE_FINAL_TRAJ_LANDE = True
            else:
                print('There are no final Lande stats')
                return
        else:
            if self._ftstats_stats_non_lande:
                self._THERE_ARE_FINAL_TRAJ_NON_LANDE = True
            else:
                print('There are no final nonLande stats')
                return

        save_name = self._summary_name
        if lande:
            save_name += self._traj_lande_ending
        for tup in list(final_dict.keys()):
            dir = self.final_tdir(tup)

            if not os.path.exists(dir):
                print(dir, ' does not exist')
                return
            file_name = os.path.join(dir, save_name)
            if final_dict[tup]:
                with open(file_name, 'wb') as f:
                    pickle.dump(default_to_regular(final_dict[tup]), f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print('Nothing in final dict of ', tup)
                return



    def _read_all_traj_bstats(self, requiredstats=None, tuples=None, lande=None):
        """ Reads EVERY run in the base directory for each of the requiredStats.

Returns
--------
out: dictionary [statname][col][my_time]['mean'\'std']. There will only be more than one col
    if the stat is a histogrom (one col for every bin)
out: dictionary[statname] = number of runs in the statdirectory
"""
        if lande is None:
            lande = self._LANDE_STATUS
        ending = self._run_file_name_end
        if lande:
            ending += self._traj_lande_ending


        if requiredstats != None:
            stats = [s for s in self._traj_stats if s in requiredstats]
            for s in requiredstats:
                if s not in stats:
                    print(s, 'not in', self._base_dir)
                    return
        else:
            return

        if tuples != None:
            the_tuples = [s for s in tuples if s in self.tuples]
            for s in tuples:
                if s not in the_tuples:
                    print(s, 'not in', self._base_dir)
                    return
        else:
            return

        # [tup][statname][col][my_time]['pos'/'neg'/'both]['mean'\'std'\'se']
        data =  defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
        #n_runs_dict = defaultdict(dict)
        for tup in the_tuples:
            # for each statistic (represented by a subdirectory)
            for statname in stats:
                if statname in self._tstats:
                    stat_dir = self._get_tstat_full_directory_path_with_units(tup=tup, statname=statname)
                elif statname in self._htstats:
                    stat_dir = self._get_tstat_full_directory_path_with_units(tup=tup, statname=statname)
                if os.path.isdir(stat_dir) and statname[0] != '.':
                    run_dirs = self._find_files_in_dir_with_ending(stat_dir, ending)
                    n_runs = len(run_dirs)
                    if n_runs > 0:
                        data_little = self._combine_all_traj_from_run_files(run_dirs)
                        #n_runs_dict[tup][statname] = n_runs
                        data[tup][statname] = data_little
        return data

    def _read_all_final_traj_bstats(self, tuples=None, lande=None):
        """ Reads EVERY run in the base directory for each of the requiredStats.

Returns
--------
out: dictionary [statname][col][my_time]['mean'\'std']. There will only be more than one col
    if the stat is a histogrom (one col for every bin)
out: dictionary[statname] = number of runs in the statdirectory
"""
        if lande is None:
            lande = self._LANDE_STATUS
        ending = self._run_file_name_end
        if lande:
            ending += self._traj_lande_ending
        if tuples != None:
            the_tuples = [s for s in tuples if s in self.tuples]
            for s in tuples:
                if s not in the_tuples:
                    print(s, 'not in', self._base_dir)
                    return
        else:
            return

        # [tup][statname]['pos'/'neg'/'both']['mean'\'std','SE']
        data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda :defaultdict(float))))

        stats_found = set()

        for tup in the_tuples:
            # for each statistic (represented by a subdirectory)
            final_dir = self.final_tdir(tup)
            # if not os.path.exists(final_dir):
            #     print 'No ' + final_dir
            if os.path.isdir(final_dir):
                run_dirs = self._find_files_in_dir_with_ending(final_dir, ending)
                data_little = self._combine_all_final_traj_from_run_files(run_dirs)
                data[tup] = data_little
                for st in list(data[tup].keys()):
                    stats_found.add(st)

        # Record what the final stats are
        for st in stats_found:
            units, statbody = float(st[1]), st[3:]
            # If we have never read the units for this stat, read them
            if statbody not in self._unit_dict:
                self._unit_dict[statbody] = units

            if st not in self._ftstats:
                self._ftstats.append(statbody)
            if lande:
                if st not in self._ftstats_stats_lande:
                    self._ftstats_stats_lande.append(statbody)
            else:
                if st not in self._ftstats_stats_non_lande:
                    self._ftstats_stats_non_lande.append(statbody)

        if lande:
            if self._ftstats_stats_lande:
                self._THERE_ARE_FINAL_TRAJ_LANDE = True
        else:
            if self._ftstats_stats_non_lande:
                self._THERE_ARE_FINAL_TRAJ_NON_LANDE = True
        return data


    def _combine_all_traj_from_run_files(self, run_files=None):

        #['pos'/'neg'/'both']['sum'/'sum_squares'/'num']

        data_little = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: defaultdict(lambda: defaultdict(float)))))
        table = self._read_all_traj_from_run_files(run_files)
        first = True
        nCols, nTimes = len(table), len(table[0])
        # summarize mean and std for the statistic
        for col in range(nCols):
            #think its run_files[0] not run_files[col]
            stati = self._read_statname_from_path_of_run_name(run_files[0])
            #print stati
            for time in list(table[col].keys()):
                # ['pos'/'neg'/'both']
                for st in self._traj_starts:
                    sum = float(table[col][time][st][self._run_values[0]])
                    sum_squares = float(table[col][time][st][self._run_values[1]])
                    num_muts = float(float(table[col][time][st][self._run_values[2]]))
                    mean, var = self._mean_var(sum,sum_squares,num_muts)

                    if var >=0:
                        std = np.sqrt(var)
                        se =  std/np.sqrt(num_muts)
                    else:
                        std = 0.0
                        se = 0.0
                        if time > 0 and first:
                            print('stat: ', stati)
                            print('start ', st)
                            print('problem var < 0 ')
                            print('time ', time)
                            print('mean ', mean)
                            print('var ', var)
                            print('start ', st)
                            print('nmuts ' , num_muts)
                            first = False
                    if num_muts <= 1:
                        if first:
                            print('num muts ',num_muts)
                            first = False
                        se = 0
                    data_little[col][time][st][self._summary_values[0]] = mean
                    data_little[col][time][st][self._summary_values[1]] = std
                    data_little[col][time][st][self._summary_values[2]] = se
        return data_little


    def _combine_all_final_traj_from_run_files(self, run_dirs=None):

        if run_dirs is None:
            run_dirs =[]
        # [statname]['pos'/'neg'/'both']['sum'\'sum_squares,'num_muts']

        sum_etc_final_dict = self._read_all_final_traj_from_run_files(run_dirs)

        # [statname]['pos'/'neg'/'both']['mean'\'std','SE']
        combined_final_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        stats = list(sum_etc_final_dict.keys())
        for stati in stats:

            for st in self._traj_starts:
                sum = float(sum_etc_final_dict[stati][st][self._run_values[0]])
                sum_squares = float(sum_etc_final_dict[stati][st][self._run_values[1]])
                num_muts = float(sum_etc_final_dict[stati][st][self._run_values[2]])
                mean, var = self._mean_var(sum, sum_squares, num_muts)
                if var > 0:
                    std = np.sqrt(var)
                    if np.isnan(std):
                        std = 0
                        se = 0
                    else:
                        se = std / np.sqrt(num_muts)
                        if np.isnan(se):
                            se = 0
                else:
                    std = 0.0
                    se = 0.0
                if stati == 'U0_NUM_MUTANTS' or stati == 'U0_NUM_SEGREGATING':
                    mean = num_muts
                    std = 0.0
                    se = 0.0
                combined_final_dict[stati][st][self._summary_values[0]] = mean
                combined_final_dict[stati][st][self._summary_values[1]] = std
                combined_final_dict[stati][st][self._summary_values[2]] = se

        return combined_final_dict

    def _save_combined_runs_traj_bstats(self, tup, stat, combined_runs_data, lande):
        if stat in self._traj_stats:
            stat_dir = self._get_tstat_full_directory_path_with_units(tup=tup, statname=stat)
        else:
            print(stat + ' not in traj bstats or hbstats')
            return
        if lande:
            file_name = os.path.join(stat_dir, self._combined_runs_name+self._traj_lande_ending)
        else:
            file_name = os.path.join(stat_dir, self._combined_runs_name)

        list_columns = list(combined_runs_data.keys())

        # record how many runs are being combined
        a_column = list_columns[0]
        a_time = next(iter(combined_runs_data[a_column]))
        a_traj_start = self._traj_starts[0]
        number_of_runs = combined_runs_data[a_column][a_time][a_traj_start][self._run_values[2]]

        self._save_text_file_number_of_alleles_in_combined(tup=tup, stat=stat, number_runs=number_of_runs,lande=lande)
        if lande:
            self._number_pair_alleles_trajecties_combined = number_of_runs
        else:
            self._number_pair_alleles_trajecties_combined_lande = number_of_runs

        # table_new[0][time][st][theval]

        with open(file_name, 'wb') as f:
            for column in list_columns:
                pickle.dump(default_to_regular(combined_runs_data[column]), f, protocol=pickle.HIGHEST_PROTOCOL)

    def _read_number_of_allele_pairs_in_combined(self, tup, stat, lande):
        if stat == self._final_stat_dir_name:
            stat_dir = self.final_tdir(tup)
        else:
            stat_dir = self._get_tstat_full_directory_path_with_units(tup=tup, statname=stat)
        name = self._recording_number_runs_combined_name
        if lande:
            name += self._traj_lande_ending
        if os.path.exists(os.path.join(stat_dir, name + ".txt")):
            number_string = self._read_first_line_file(directory=stat_dir,filename=name+ ".txt")
            number_alleles = float(number_string)
        else:
            number_alleles = 0

        return number_alleles


    def _read_number_runs_of_allele_pair_simulations(self):
        if not self.tuples:
            self._number_pair_alleles_trajecties_combined = 0
            self._number_pair_alleles_trajecties_combined_lande = 0
            self._number_pair_alleles_trajecties_final_combined = 0
            self._number_pair_alleles_trajecties_final_combined_lande = 0
        else:
            a_tuple = list(self.tuples)[0]
            if self._traj_stats:
                token_stat = self._traj_stats[0]
                self._number_pair_alleles_trajecties_combined = \
                    self._read_number_of_allele_pairs_in_combined(a_tuple,token_stat,lande=False)
                self._number_pair_alleles_trajecties_combined_lande = \
                    self._read_number_of_allele_pairs_in_combined(a_tuple,token_stat,lande=True)

            if self._THERE_ARE_FINAL_TRAJ_NON_LANDE:
                self._number_pair_alleles_trajecties_final_combined = \
                    self._read_number_of_allele_pairs_in_combined(a_tuple,self._final_stat_dir_name,lande=False)
            if self._THERE_ARE_FINAL_TRAJ_LANDE:

                self._number_pair_alleles_trajecties_final_combined_lande = \
                    self._read_number_of_allele_pairs_in_combined(a_tuple,self._final_stat_dir_name,lande=True)


    def _save_text_file_number_of_alleles_in_combined(self, tup, stat, number_runs,lande):
        if stat == self._final_stat_dir_name:
            stat_dir = self.final_tdir(tup)
        else:
            stat_dir = self._get_tstat_full_directory_path_with_units(tup=tup, statname=stat)
        name = self._recording_number_runs_combined_name
        if lande:
            name +=  self._traj_lande_ending
        savefolderfile = os.path.join(stat_dir, name + ".txt")
        tf = open(savefolderfile, "w")
        tf.write(str(number_runs))
        tf.write("\n")
        tf.write("Trajectory simulations (OA) results for " + str(number_runs)
                 + " pairs of aligned and opposing alleles were combined to create the "
                 + self._combined_runs_name + " file for " + stat + ", for this tuple")
        tf.close()

    def _save_combined_runs_final_traj_bstats(self, tup, combined_runs_data, lande):
        if tup not in self.tuples:
            print(tup, ' not in tuples')
            return
        # for each tup (represented by a subdirectory)
        final_dir = self.final_tdir(tup)
        if lande:
            file_name = os.path.join(final_dir, self._combined_runs_name+self._traj_lande_ending)
        else:
            file_name = os.path.join(final_dir, self._combined_runs_name)

        list_stats = list(combined_runs_data.keys())

        a_stat = list_stats[0]
        if a_stat == 'NUM_SEGREGATING' and len(list_stats) > 1:
            a_stat = list_stats[1]
        a_traj_start = self._traj_starts[0]

        if list_stats:
            # record how many runs are being combined
            number_of_runs = combined_runs_data[a_stat][a_traj_start][self._run_values[2]]

            self._save_text_file_number_of_alleles_in_combined(tup=tup, stat= self._final_stat_dir_name, number_runs=number_of_runs,lande=lande)
            if lande:
                self._number_pair_alleles_trajecties_final_combined_lande = number_of_runs
            else:
                self._number_pair_alleles_trajecties_final_combined = number_of_runs

        with open(file_name, 'wb') as f:
            pickle.dump(default_to_regular(combined_runs_data), f, protocol=pickle.HIGHEST_PROTOCOL)

    def _read_combined_runs_traj_bstats(self, tup, statname, lande):
        table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))

        if statname not in self._traj_stats:
            print(statname, 'not in', self._base_dir)
            return
        # for each statistic (represented by a subdirectory)
        stat_dir = self._get_tstat_full_directory_path_with_units(tup=tup, statname=statname)
        if os.path.isdir(stat_dir) and statname[0] != '.':
            if lande:
                filedir = os.path.join(stat_dir, self._combined_runs_name+self._traj_lande_ending)
            else:
                filedir = os.path.join(stat_dir, self._combined_runs_name)
            # If there is alread a combined runs, read it
            if os.path.isfile(filedir):
                iterater = self._load(filedir)
                c = 0
                for column in iterater:
                    table[c] = column
                    c += 1
        return table

    def _read_combined_runs_final_traj_bstats(self, tup, lande):

        table = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        if tup not in self.tuples:
            print(tup, ' not in tuples')
            return
        # for each tup (represented by a subdirectory)
        final_dir = self.final_tdir(tup)
        if os.path.isdir(final_dir):
            if lande:
                filedir = os.path.join(final_dir,self._combined_runs_name+self._traj_lande_ending)
            else:
                filedir = os.path.join(final_dir, self._combined_runs_name)
            # If there is alread a combined runs, read it
            if os.path.isfile(filedir):
                iterater = self._load(filedir)
                table = next(iterater)
        return table


    def _read_all_traj_from_run_files(self, run_files=None):

        if run_files is None:
            run_files = []

        if run_files:
            stati = self._read_statname_from_path_of_run_name(run_files[0])
            tup = self._read_tuple_from_path_of_run_name(run_files[0])
            lande_status = self._test_if_lande_from_path_of_run_name(run_files[0])
        else:
            return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))

        #list_dict = table[col][time]
        #table = defaultdict(lambda: defaultdict(list))

        #table_new = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))

        table_new = self._read_combined_runs_traj_bstats(tup=tup, statname=stati,lande=lande_status)

        for run_name in run_files:
            iterater = self._load(run_name)
            try:
                for time,a_dict in next(iterater)[0].items():
                    #so if it is a histostat
                    if isinstance(a_dict, list):
                        for col in range(len(a_dict)):
                            a_dict_single = a_dict[col]
                            # ['pos'/'neg'/'both']
                            for st in self._traj_starts:
                                # sum, sum squares, number of mutants
                                for theval in self._run_values:
                                    table_new[col][time][st][theval] += a_dict_single[st][theval]
                    # so if it is a basic stat #more likely
                    else:
                        # ['pos'/'neg'/'both']
                        for st in self._traj_starts:
                            # sum, sum squares, number of mutants
                            for theval in self._run_values:
                                try: #REMOVE THIS
                                    table_new[0][time][st][theval] += a_dict[st][theval]
                                except KeyError:
                                    print('Key error ')
                                    print('time ', time)
                                    print('start ', st)
                                    print('the value ', theval)
                                    print(a_dict)
                os.rename(run_name, run_name + '_have_read')
            except StopIteration:
                print('Stop iteration error in:')
                print(run_name)
        if table_new:
            self._save_combined_runs_traj_bstats(tup=tup, stat=stati, combined_runs_data=table_new,lande=lande_status)
        return table_new

    def _read_all_final_traj_from_run_files(self, run_dirs=None):

        if run_dirs is None:
            run_dirs = []
        if not run_dirs:
            return defaultdict(lambda: defaultdict(lambda : defaultdict(float)))
        else:
            lande_status = self._test_if_lande_from_path_of_run_name(runpath=run_dirs[0])
            tup = self._read_tuple_from_path_of_run_name(run_dirs[0])
        # [statname]['pos'/'neg'/'both']['sum'\'sum_squares','num']
        comb_final_dict = self._read_combined_runs_final_traj_bstats(tup, lande=lande_status)

        if not run_dirs:
            return comb_final_dict
            # tup = self._read_tuple_from_path_of_run_name(run_dirs[0])

        final_dict_list = []
        for run_name in run_dirs:
            # [statname]['pos'/'neg'/'both']['sum'\'sum_squares','num']
            temp_final_dict = defaultdict(lambda: defaultdict(dict))
            iterater = self._load(run_name)
            try:
                # nmuts = next(iterater)
                temp_final_dict = next(iterater)[0]

                final_dict_list.append(temp_final_dict)
                #TEMP BLOCK OUT
                os.rename(run_name, run_name + '_have_read')
            except StopIteration:
                print('Stop iteration error in:')
                print(run_name)

        # Gather possible stats
        stats = list(comb_final_dict.keys())
        for fd in final_dict_list:
            for key in fd:
                if key not in stats:
                    stats.append(key)

        for stati in stats:

            for st in self._traj_starts:
                # sum, sum squares, number of mutants
                sum_and_squares_and_num = [0 for _ in self._run_values]
                for i in range(len(self._run_values)):
                    for dicti in final_dict_list:
                        if stati in dicti:
                            #front = 'U' + str(self._get_units_of_stat(stati)) + '_'
                            sum_and_squares_and_num[i] += dicti[stati][st][self._run_values[i]]

                sum = float(sum_and_squares_and_num[0])
                sum_squares = float(sum_and_squares_and_num[1])
                num_muts = float(sum_and_squares_and_num[2])

                comb_final_dict[stati][st][self._run_values[0]] += sum
                comb_final_dict[stati][st][self._run_values[1]] += sum_squares
                comb_final_dict[stati][st][self._run_values[2]] += num_muts
        #TEMP BLOCK OUT
        if comb_final_dict:
            self._save_combined_runs_final_traj_bstats(tup=tup, combined_runs_data=comb_final_dict, lande=lande_status)
        return comb_final_dict

# SOME theory
    def _initialize_C(self):
        """initialize the allelic measure of the deviation from/ Lande (amplification)
        for pure Lande trajectories"""
        self._initialize_C_lande() #if we ONLY have trajectories simulations, we need to use the Lande C

    def _initialize_eta(self):
        """find ,eta, which is the fraction contribution to the change in mean phenotype
        coming from standing variation. In the Lande case, eta = 1"""
        self._initialize_eta_lande() # if we ONLY have trajectories simulations, we need to use the Lande eta

class dataClass(dataClassFullSims,dataClassTraj):
    """
    Class to read the simulation results in base_dir, if they were generated from both population
    and trajectory simulations
    """
    def __init__(self, base_dir=os.getcwd(), units=None):
        dataClassFullSims.__init__(self,base_dir=base_dir,units=units)
        self._traj_class_set_up()


    def stat_writer_param_dict(self):
        """
        In the base directory there is a pickled dictionary of 'basic parameters'. This function reads it and saves the param dict
        """
        param_dict = super(dataClassFullSims, self).stat_writer_param_dict()
        return param_dict

    def _create_param_dict(self):
        """read the parameters and create the dictionary of parameters"""
        self._read_params_basic()
        self._read_params_full_sims()
        #self._read_params_traj()
        print(self.param_dict)

    def make_parameter_text_to_write_file(self):
        namebase = 'N_'+ str(int(self.param_dict['N']))
        namebase += '_U_' + str(self.param_dict['U'])
        namebase += '_Ds0_' + str(int(self.param_dict['shift_s0']))
        namebase +='_E2Ns_' + str(int(self.param_dict['E2Ns'])) +'_V2Ns_' + str(int(self.param_dict['V2Ns']))
        namebase=namebase.replace(".","_")
        return namebase

# SOME theory
    def _initialize_C(self):
        """initialize C the allelic measure of the deviation from Lande (amplification)"""
        self._initialize_C_nonlande()

    def _initialize_eta(self):
        """initialze ,eta, which is the fraction contribution to the change in mean phenotype
        coming from standing variation.
        Note that eta can only be obtained from simulations, so the data class will not be able to calculate
        # this unless you collected the statistic "d2ax_frozen_over_shift"
        """
        self._initialize_eta_nonlande()



