from random import getrandbits
from datetime import datetime
import pickle
import os
import hashlib


class statWriter(object):
    """
    Easy-to-use writer for various statistics.

    Statistics are not defined in advance; usage: w.write('statName',*vals)

    Parameters:
        base_dir: (string) The base statistics directory. Most commonly results are written to:
                   base_dir/id_DIR/statistic/runFile; where id_DIR is shared between
                   runs with the same identifiers; statistic is a directory for each saved
                    statistic 'statName'
        param_dict: (dict) dict by keywoards. all values must be hashable.

     Attributes:
         _dir:
         _writers: (dict) keys: writer
         _runID: (string) ID for this particular run: form 'ID_aRun'
    """
    
    def __init__(self, base_dir, param_dict):
        
        # map the identifiers specified to a unique directory (sorting required for deterministic hashing order)
        hashfunct = hashlib.blake2s(digest_size=10) # define the deterministic hash function
        hashfunct.update(str(tuple(sorted(param_dict.items()))).encode()) # encode paramdict items in hash
        self._hashID = hashfunct.hexdigest() # this hash is the same for every run with the same parameters
        self._dir = base_dir + '/' + self._hashID + '_pD' + '/'
        # if the directory doesn't exist, create it and save the identifiers
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
            with open(self._dir+'identifiers.txt', "w") as f:
                f.write('creation my_time : ' + str(datetime.now()) + '\n')
                f.writelines([str(k) + '\t: ' + str(v) + '\n' for k,v in list(param_dict.items())])

        self._writers = {}
        self._runIDINT = getrandbits(64) # each run has a unique identifier
        self._runID = str(self._runIDINT) + '_aRun'

    def hashID(self):
        return self._hashID

    def write_info(self, info, info_name):
        filedir = self._dir + info_name
        if not os.path.isfile(filedir):
            with open(filedir, 'wb') as f:
                #Line 1 is dictionary of " + info_name +" values"
                pickle.dump(info, f, protocol=pickle.HIGHEST_PROTOCOL)

    def write(self, key, dir_type = None,lande=None, *vals):
        """
        Use this to record data from runs. If dir_type is _sD then we are writing a
        basic stat and the directory will end with _sD. If dir_type = None then
        no particular end to file name

        Parameters:
            key: (float) The stat we are recording or, in the traj case, contains info about the tuple and the stat
            dir_type: (float) The type of directory 'e.g.' _sD (for stat directory)
              If dir type is 'traj' then put in folder called traj
            vals: () Thing to be pickled
        """

        if lande is None:
            lande = False

        if lande:
            key_reference = key + '_lande'
        else:
            key_reference = key


        # if there is no writer for this key, create it
        if key_reference not in self._writers:
            if dir_type == 'traj': # traj stats over time
                dir_key = self._dir + 'trajectories/'
                dir_key = dir_key + str(key)  +'_tsD/'
            elif dir_type == 'traj_tfD': # final traj stats dictionary
                dir_key = self._dir + 'trajectories/'
                dir_key = dir_key  + str(key)  +'_tfD/'
            elif dir_type is not None:
                dir_key = self._dir + str(key) + dir_type  +'/'
            else:
                dir_key = self._dir + str(key) +'/'
            # if there is no directory for this key, create it
            if not os.path.exists(dir_key):
                os.makedirs(dir_key)
            # create writer
            if lande:
                self._writers[key_reference] = open(dir_key + self._runID + '_lande', 'wb')
            else:
                self._writers[key_reference] = open(dir_key + self._runID ,'wb')
        #Write object vals to file using pickle
        pickle.dump(vals, self._writers[key_reference], protocol=pickle.HIGHEST_PROTOCOL)

    def write_explanation(self, info_dict, dir_type, tup_file=None):
        """
        Use this to record data from runs. If dir_type is _sD then we are writing a
        basic stat and the directory will end with _sD. If dir_type = None then
        no particular end to file name

        Parameters:
            tup_file: (float) In the traj case, contains info about the tuple
            dir_type: (float) The type of directory 'e.g.' _sD (for stat directory)
              If dir type is 'traj' then put in folder called traj
            info_dict: (dict) Thing info to be recorded in a text file
        """

        if tup_file is None:
            tup_file = ''
        else:
            tup_file = str(tup_file) + '/'

        dir = self._dir
        if dir_type == 'traj':  # traj stats over time
            dir = dir + 'trajectories/'
            dir = dir + tup_file
            text_file_name = 'READ_ME_about_traj_stats.txt'

        elif dir_type == 'traj_tfD':  # final traj stats dictionary
            dir= dir + 'trajectories/'
            dir = dir + tup_file
            text_file_name = 'READ_ME_about_final_traj_stats.txt'
        else:
            text_file_name = 'READ_ME_about_stats.txt'

        # if the directory doesn't exist, create it and save the identifiers
        print(dir)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir+text_file_name, "w") as f:
            f.write(self._make_some_explanation_lines(dir_type) + '\n')
            f.writelines([str(k) + '\t: ' + str(v) + '\n' for k,v in list(info_dict.items())])

    def get_traj_key(self, stat, tup):
        """
        Parameters:
            stat: (float) The stat we are recording
            tup: (tup) (S,x,t)
        """
        file = self.get_traj_tuple_key(tup)
        key = file +'/' + stat
        return key

    def get_traj_tuple_key(self,tup):
        """
        Parameters:
            tup: (tup) (S,x,t)
        Generate the folder name for this tuple
        """
        S= tup[0]
        XI = tup[1]
        t = tup[2]
        XI_string = str(XI)
        S_string = str(S)
        XI_string = XI_string.replace('.','_')
        S_string = S_string.replace('.','_')
        t_string = str(t)
        file = 'XI_' + XI_string + '_S_' + S_string + '_t_' + t_string + '_tD'
        return file


    def close_writers(self):
        """
        Closes all the writers.
        """
        for _,w in self._writers.items():
            w.close()

    def read(self, key, bstat=None):
        """
        Reads the stuff in the 'key file and returns an iterator with everythin in it. If bstat is 'B' then we are writing a
        basic stat and the directory will end with _sD. If bstat = 'nB' then we are
        not reading a basic stat

        WARNING: only to be used after the writers have been closed
        """
        if bstat is None:
            bstat = True

        if key not in self._writers:
            print(key, 'not in writers')
            pass
        if bstat:
            dirKey = self._dir + str(key) +'_sD/'
        else:
            assert bstat == False
            dirKey = self._dir + str(key) +'/'
        return self._load(dirKey + self._runID)

    def delete_file(self, key, bstat =None):
        """Allows us to delete the file for our run in tmp when we no longer need it.

        If bstat is True then we are writing a
        basic stat and the directory will end with _sD. If bstat is False then we are
        not deleting a basic stat"""
        if bstat is None:
            bstat = False
        if bstat:
            dirKey = self._dir + str(key) +'_sD/'
        else:
            assert bstat == False
            dirKey = self._dir + str(key) +'/'
        fileKey = dirKey+ self._runID
        os.remove(fileKey)

    @staticmethod
    def _load(filename):
        """
        Reads the pickled file and produces an iterator over items inside it
        """
        with open(filename, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    def _make_some_explanation_lines(self,dir_type):
        """Generates the text to be saved in readme type files explaining how the various statistics are saved"""
        some_explantion_lines = ''
        if dir_type == 'traj':  # traj stats over time
            some_explantion_lines += "INFORMATION ABOUT THE TRAJECTORY STATISTICS SAVED IN EACH TUPLE FOLDER\n"
            some_explantion_lines += "(we call them trajectory statistics because they were generated with the trajectory (or OA)" \
                                     "simulator)\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "HOW TO READ THE TRAJECTORY STATISTICS\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "The trajectory stats saved here are saved using cpickle, which can be read " \
                                     "using the classes in read_data.py. \n "
            some_explantion_lines += "Specifically you should use: the class called 'dataClassTraj' " \
                                     "(for pure Lande trajectories),\n"
            some_explantion_lines += " or the class called 'dataClass' (for trajectories saved in a folder that also " \
                                     "has population simulation results). \n"
            some_explantion_lines += "\n"
            some_explantion_lines += "It is recommended that you use summarize_save_argparse.py to summarize " \
                                     "the results of your simulations. This will also record *some* of the summarized" \
                                     " results in .txt files in a folder called 'text_results', which you may prefer \n"
            some_explantion_lines += "\n"
            some_explantion_lines += "ABOUT THE TUPLE FOLDERS\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "Each tuple folder is of the form 'XI_xi_S_a2_t_time_tD'.\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "SQUARED EFFECT: The value 'a2' after 'S_' means alleles in that tuple have 2Ns at \n"
            some_explantion_lines += "steady-state of 'a2'. e.i. the have squared effects in units of V_s/(2N) of size a2.\n" \
                          " Within a2 an underscore ('_') denotes a decimal point (e.g. S_1_32_... means a2 = 1.32)\n"
            some_explantion_lines += "INITIAL MAF: The value 'xi' after 'XI_' means that the initial MAF of alleles " \
                          "in that tuple were 'xi'.\n"
            some_explantion_lines += "NB NB NB: If xi <0, then alleles in that tuple had initial frequencies sampled from" \
                          " the steady-state MAF distribution of alleles with 2Ns at steady-state of 'a2'. \n"
            some_explantion_lines += "I ALWAYS take xi < 0. It is thus possible the code is a bit buggy if xi > 0."
            some_explantion_lines += "TIME: The value 'time' after 't_' means that that we are looking at alleles " \
                                     "from generation 'time'.\n"
            some_explantion_lines += "If time = 0, then we are considering alleles that were segregating at " \
                          "generation  0 (i.e. at the time of shift).\n"
            some_explantion_lines += "If time > 0 then we are considering *mutations* that arose in generation time.\n"
            some_explantion_lines += "I ALWAYS take time = 0. It is thus possible the code is a bit buggy if time != 0.\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "ABOUT THE HOW THE STATS IN THE TUPLE FOLDERS ARE STORED\n"
            some_explantion_lines += "\n"
            some_explantion_lines += 'Many folders inside each tuple folder have the form UX_statname_tsD\n'
            some_explantion_lines += "The folder names imply that the statistic 'statname' is recorded inside that " \
                                     "folder\n"
            some_explantion_lines += "The UX_ at the beginning of the name means that the statistic has units of " \
                                     "(trait)^X.\n"
            some_explantion_lines += "So, for instance, U2 means that the statistic has units of trait squared, and U0" \
                                     " means that the statistic is unitless.\n"
            some_explantion_lines += "E.g. if the statistic was phenotypic variance it would start with U2, skewness " \
                                     "would start with U0_\n"
            some_explantion_lines += "Inside the folder statname_tsD you may find the results of various runs the" \
                                     "trajecotry simulations (OA)\n"
            some_explantion_lines += "The results of each run, are recorded in a file either called 'RUN_ID_aRun' or '"\
                                     "RUN_ID_aRun_lande.\n"
            some_explantion_lines += "RUN_ID is a string of numbers uniquely identifying that specific run of the " \
                                     "simulator \n"
            some_explantion_lines += "The '_lande' means that run RUN_ID was done using a lande's approx for D(t)'\n"
            some_explantion_lines += "If it just ends '_aRun' than RUN_ID was simulated using D(t) from the " \
                                     "population sims which are saved in folder containing this directory'\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "For each RUN_ID, the statistic statname is recorded at various sample times as\n"
            some_explantion_lines += "dict[sample_time]['pos'/'neg'/'both']['sum'/'sum_squares'/'num'] = value\n"
            some_explantion_lines += "Here 'pos' means we are looking at the statistic only for alleles with effects " \
                                     " aligned to the shift. "
            some_explantion_lines += "And 'neg' means we are looking only at the statistic only for alleles with"\
                                     "effects opposing the shift.\n"
            some_explantion_lines += "And 'both' means we are looking at the combined statistic for alleles with "\
                                     "effects aligned AND opposing the shift.\n"
            some_explantion_lines += "'sum' means that we are looking at the sum of that statistic, where the sum is " \
                                     "taken over the all the alleles simulated for the relavent tuple in RUN_ID\n"
            some_explantion_lines += "of which there are 'num' of them.\n"
            some_explantion_lines += "'sum_squares' means that we are looking at the sum of squares of that statistic " \
                                     "taken over alleles, of which there are 'num' of them.\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "If you have used sum_and_save_argparse.py to summarize and save the " \
                                     "reults of sims,\n"
            some_explantion_lines += "then instead (or in addition to) the results of individual simulations (RUN_ID_aRun)\n"
            some_explantion_lines += ", you will find the results of many simulations combined in the " \
                                       "file 'combined_runs.'\n"
            some_explantion_lines +=  "You will also find the file 'summary', which is a summary of the sim results in" \
                                       "the file 'combined_runs'.\n"
            some_explantion_lines += "In 'summary' the statistic statename is recorded at various sample times as\n"
            some_explantion_lines += "dict[sample_time]['pos'/'neg'/'both']['mean'/'se'/'std'] = value\n"
            some_explantion_lines += "The keys 'mean', 'se' and 'std' denote that value is the average, standard" \
                                     " error or standard deviation of the statistic, respectively, \n"
            some_explantion_lines += " for all of the alleles whose results have been combined into 'combined_runs'.\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "ABOUT WHAT THE STATS ARE\n"

        elif dir_type == 'traj_tfD':  # final traj stats dictionary
            some_explantion_lines += "INFORMATION ABOUT THE FINAL TRAJECTORY STATISTICS SAVED IN 'final_stats_tfD' " \
                                     "INSIDE EACH TUPLE FOLDER\n"
            some_explantion_lines += "(we call them *final* trajectory statistics because they were generated with the " \
                                     "trajectory (or OA) simulator, and recorded only after every allele simulated " \
                                     "was lost or fixed)\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "ABOUT THE HOW THE FINAL STATS IN THE TUPLE FOLDERS ARE STORED\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "Inside each tuple folder is the folder 'final_stats_tfD'\n"
            some_explantion_lines += "Inside 'final_stats_tfD' you may find the results of various runs the" \
                                     "trajecotry simulations (OA)\n"
            some_explantion_lines += "The results of each run, are recorded in a file either called 'RUN_ID_aRun' or '"\
                                     "RUN_ID_aRun_lande.\n"
            some_explantion_lines += "RUN_ID is a string of numbers uniquely identifying that specific run of the " \
                                     "simulator \n"
            some_explantion_lines += "The '_lande' means that run RUN_ID was done using a lande's approx for D(t)'\n"
            some_explantion_lines += "If it just ends '_aRun' than RUN_ID was simulated using D(t) from the " \
                                     "population sims which are saved in folder containing this directory'\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "For each RUN_ID, the a final statistic dictionary is recorded as\n"
            some_explantion_lines += "final_dict[statistic]['pos'/'neg'/'both']['sum'/'sum_squares'/'num'] = value\n"
            some_explantion_lines += "'statistic' is just some stat that we recorded after every allele was fixed or lost"
            some_explantion_lines += "And 'pos' means we are looking at the statistic only for alleles with effects " \
                                     " aligned to the shift. "
            some_explantion_lines += "And 'neg' means we are looking only at the statistic only for alleles with "\
                                     "effects opposing the shift.\n"
            some_explantion_lines += "And 'both' means we are looking at the combined statistic for alleles with "\
                                     "effects aligned AND opposing the shift.\n"
            some_explantion_lines += "'sum' means that we are looking at the sum of that statistic, where the sum is " \
                                     "taken over the all the alleles simulated for the relevant tuple in RUN_ID\n"
            some_explantion_lines += "of which there are 'num' of them.\n"
            some_explantion_lines += "'sum_squares' means that we are looking at the sum of squares of that statistic taken over alleles," \
                                     "of which there are 'num' of them.\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "If you have used sum_and_save_argparse.py to summarize and save the " \
                                     "results of sims,\n"
            some_explantion_lines += "then instead (or in addition to) the results of individual simulations (RUN_ID_aRun)\n"
            some_explantion_lines += ", you will find the results of many simulations combined in the " \
                                       "file 'combined_runs.'\n"
            some_explantion_lines +=  "You will also find the file 'summary', which is a summary of the sim results in" \
                                       "the file 'combined_runs'.\n"
            some_explantion_lines += "In 'summary' the statistic statname is recorded at various sample times as\n"
            some_explantion_lines += "dict[statistic]['pos'/'neg'/'both']['mean'/'se'/'std'] = value\n"
            some_explantion_lines += "The keys 'mean', 'se' and 'std' denote that value is the average, standard" \
                                     " error or standard deviation of the statistic, respectively, \n"
            some_explantion_lines += " for all of the alleles whose results have been combined into 'combined_runs'.\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "ABOUT WHAT THE STATS ARE\n"
            some_explantion_lines += "(in the each dictionary of final statistics, you will find the following)\n"
        else:
            some_explantion_lines += "INFORMATION ABOUT ALL THE BASIC OR HIST STATISTICS SAVED IN FOLDERS THAT END " \
                                     "_sD \n"
            some_explantion_lines += "(we call them basic or histo statistics because they were generated with the " \
                                     "population simulator (validation or AA))\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "HOW TO READ THE BASIC AND HISTO STATISTICS\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "The basic and histo stats saved here are saved using cpickle, which can be read " \
                                     "using the classes in read_data.py. \n "
            some_explantion_lines += "Specifically you could use: the class called 'dataClassFull' " \
                                     "(for just population simulations),\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "It is recommended that you use summarize_save_argparse.py to summarize " \
                                     "the results of your simulations. This will also record *some* of the summarized" \
                                     " results in .txt files in a folder called 'text_results', which you may prefer \n"
            some_explantion_lines += "\n"
            some_explantion_lines += "ABOUT THE HOW THE STATS IN THE _sD FOLDERS ARE STORED\n"
            some_explantion_lines += "\n"
            some_explantion_lines += 'Many folders have the form UX_statname_sD or ' \
                                     'UX_statname_Ybins_H_sD\n'
            some_explantion_lines += "The folder names imply that the statistic 'statname' is recorded inside that " \
                                     "folder\n"
            some_explantion_lines += "The UX_ at the beginning of the name means that the statistic has units of " \
                                     "(trait)^X.\n"
            some_explantion_lines += "So, for instance, U2 means that the statistic has units of trait squared, and U0" \
                                     " means that the statistic is unitless.\n"
            some_explantion_lines += "E.g. if the statistic was phenotypic variance it would start with U2, skewness " \
                                     "would start with U0_\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "Inside the folders UX_statname_sD and UX_statname_Ybins_H_sD you may find the " \
                                     "results of various runs of the population simulations (AA or validation)\n"
            some_explantion_lines += "If the folder name ends with results Ybins_H_sD, then the stat collected was a " \
                                     "histo stat\n "
            some_explantion_lines += "This means that it was binned by MAF at the time of the shift ( 'Y' = f, so ending " \
                                     "_fbins_H_sD) or it was binned by squared effect sizes (Y = 'efs_' so " \
                                     "ending _efs_bins_H_sD)\n "
            some_explantion_lines += "If the folder name starts with UX_frozen_stat..., then the histo stat collected " \
                                     "was calculated from alleles that were segregating at the time of the shift " \
                                     "(i.e. alleles frozen at the time of the shift)\n "
            some_explantion_lines += "If the folder name starts with UX_frozen_nm..., then the histo stat collected " \
                                     "was calculated for new mutations that arose in the 5N generations " \
                                     "following the shift\n "
            some_explantion_lines += "\n"
            some_explantion_lines += "The MAFs and effect sizes that the stats were binned by are recorded " \
                                     "in the file 'hist_bins' which contains a dictionary of the form\n"
            some_explantion_lines += "hist_bin_dict['fbins' / 'efsbins'] = list_of_boundaries_between_bins\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "The results of each run, are recorded in a file called 'RUN_ID_aRun\n"
            some_explantion_lines += "RUN_ID is a string of numbers uniquely identifying that specific run of the " \
                                     "simulator \n"
            some_explantion_lines += "\n"
            some_explantion_lines += "For each RUN_ID, the statistic statname is recorded in 'RUN_ID_aRun' at various " \
                                     "sample times as a dictionary of the form:\n"
            some_explantion_lines += "dict[sample_time] = stat_value, for a bstat, or " \
                                     "dict[sample_time] = list_of_stat_values for a histo bstat \n"
            some_explantion_lines += "\n"
            some_explantion_lines += "If you have used sum_and_save_argparse.py to summarize and save the " \
                                     "results of simulationss (recommended),\n"
            some_explantion_lines += "then instead of (or in addition to) the results of individual simulations " \
                                     "(i,e. RUN_ID_aRun)\n"
            some_explantion_lines += ", you will find the results of many simulations combined in the " \
                                     "file 'combined_runs'.\n"
            some_explantion_lines += "They are saved as dict[0][sample_time][sum/sum_squares/num] = value for bstats\n"
            some_explantion_lines += "and dict[colum][sample_time][sum/sum_squares/num] = value for histo bstats\n"
            some_explantion_lines += "For a histo bstat 'colum' is  an integer for 0 to the number of bins that the stat"
            some_explantion_lines += "was collected in.\n "
            some_explantion_lines += "'sum' means that value is the value of the sum of that statistic, where the " \
                                     "sum is taken over the number of runs of the population simulations\n"
            some_explantion_lines += " (e.i. the number of RUN_ID_aRuns's that were combined, of which there are " \
                                     "'num' of them)\n"
            some_explantion_lines += "'sum_squares' means that value is the the sum of squares of that statistic.\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "You will also find the file 'summary', which is a summary of the simulation " \
                                     "results in the file 'combined_runs'.\n"
            some_explantion_lines += "In 'summary' the statistic statname is recorded at various sample times as\n"
            some_explantion_lines += "dict[sample_time]['mean'/'se'/'std'] = value\n"
            some_explantion_lines += "The keys 'mean', 'se' and 'std' denote that value is the average, standard" \
                                     " error or standard deviation of the statistic, respectively, \n"
            some_explantion_lines += " for all of the runs of evolution whose results have been combined into " \
                                     "'combined_runs'.\n"
            some_explantion_lines += "\n"
            some_explantion_lines += "ABOUT WHAT THE STATS ARE\n"

        return some_explantion_lines



