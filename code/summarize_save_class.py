"""
contains the class to summarize the statistics collected over many simulations; and to save these summaries
"""
import os
import shutil
import errno
import numpy as np
from read_data import dataClassFullSims, dataClassTraj, dataClass

class SumAndSave(object):
   def __init__(self, parent_dir=None,save_output_dir=None):
      if parent_dir is None:
         self._PARENT_FOLDER = os.getcwd()
      else:
         self._PARENT_FOLDER = parent_dir
      if save_output_dir is None:
         self._SAVE_OUTPUT_DIR = self._PARENT_FOLDER
      else:
         if not os.path.exists(save_output_dir):
            os.makedirs(save_output_dir)
         self._SAVE_OUTPUT_DIR = save_output_dir

      self._SAVE_TEXT_FILES_FOLDER = "text_results"
      self._SAVE_TEXT_FILES_DIR = os.path.join(self._SAVE_OUTPUT_DIR, self._SAVE_TEXT_FILES_FOLDER)

      results_folders = [x for x in os.listdir(self._PARENT_FOLDER) if x[0]!='.' and len(x)>3]
      results_folders = [x for x in results_folders if x[-3:] == '_pD']
      # save the set of directory addresses of folders in the parent folder that contain
      # simulation results
      self._RESULTS_DIRECTORIES_LIST = set([os.path.join(self._PARENT_FOLDER, p) for p in results_folders])
      # Set of all the sim parameter dicts corresponding to directories in parent folder
      self._PARAM_DICT_LIST = []
      # Set of all the sim parameter dicts corresponding directories with tajectories in parent folder
      self._PARAM_DICT_TRAJ_LIST = []
      # Set of all the sim parameter dicts corresponding directories with full sims in parent folder
      self._PARAM_DICT_FULL_LIST = []

      # dictionary with dict[set param tuples] = dataclass
      self._DATA_CLASS_DICT = dict()

      self._make_data_classes()

   def param_dict_list(self):
      return self._PARAM_DICT_LIST

   def turn_dict_into_dict_key(self,mydict):
      return frozenset(mydict.items())

   def get_data_class(self,frozenparamdict):
      return self._DATA_CLASS_DICT[frozenparamdict]

   def get_lande_time(self,frozenparamdict):
      """Returns the time at which Lande's approximation for D(t) is 1\delta away from the new optimum"""
      myAlpha = 1
      Vs = self._DATA_CLASS_DICT[frozenparamdict].param_dict['Vs']
      s0del = self._DATA_CLASS_DICT[frozenparamdict].param_dict['sigma_0_del']
      s0 = self._DATA_CLASS_DICT[frozenparamdict].param_dict['sigma_0']
      shifts0 = self._DATA_CLASS_DICT[frozenparamdict].param_dict['shift_s0']
      mylandtime = int(-np.log(myAlpha / (shifts0 * s0del))*Vs/ s0 ** 2)
      return mylandtime

   def write_bstats_to_text_file(self,frozenparamdict):
      internal_folder_name = self.make_folder_name_for_data_class(frozenparamdict)
      save_folder = os.path.join(self._SAVE_TEXT_FILES_DIR,internal_folder_name)
      if not os.path.exists(save_folder):
         os.makedirs(save_folder)
      tmp = self._DATA_CLASS_DICT[frozenparamdict]
      for bstati in tmp._bstats:
         tmp.write_stat_to_text_files(bstati, savedir=save_folder)

   def write_final_tstats_to_text_file(self,frozenparamdict,tfstat=None):
      internal_folder_name = self.make_folder_name_for_data_class(frozenparamdict)
      save_folder = os.path.join(self._SAVE_TEXT_FILES_DIR,internal_folder_name)
      if not os.path.exists(save_folder):
         os.makedirs(save_folder)
      tmp = self._DATA_CLASS_DICT[frozenparamdict]

      for lands in [True, False]:
         if tfstat is None:
            tmp.write_tfstat_efs_to_text_files(savedir=save_folder,lande=lands)
         else:
            tmp.write_tfstat_efs_to_text_files(tfstat=tfstat, savedir=save_folder, lande=lands)

   def write_some_hbstats_at_lande_and_end_time_to_text_file(self,frozenparamdict):
      internal_folder_name = self.make_folder_name_for_data_class(frozenparamdict)
      save_folder = os.path.join(self._SAVE_TEXT_FILES_DIR,internal_folder_name)
      if not os.path.exists(save_folder):
         os.makedirs(save_folder)

      lande_time=self.get_lande_time(frozenparamdict)

      tmp = self._DATA_CLASS_DICT[frozenparamdict]

      stat_list = ['frozen_d2ax_times_n_scaled_per_mut_input', 'frozen_d2ax_over_shift','frozen_nm_2ax_times_n_scaled_per_mut_input']

      stat_list = ['frozen_d2ax_scaled_per_mut_input', 'frozen_d2ax_over_shift',
                   'frozen_nm_2ax_scaled_per_mut_input','frozen_nm_and_standing_d2ax_scaled_per_mut_input']
      bin_list = ['e', 'f','a']

      for stata in stat_list:
         for bini in bin_list:
            tmp.write_hbstat_at_time_to_text_files(hbstat=stata, savedir=save_folder, bins=bini, time=lande_time)
            tmp.write_hbstat_at_time_to_text_files(hbstat=stata, savedir=save_folder, bins=bini)


   def write_tstats_at_lande_time_to_text_file(self,frozenparamdict,tstat=None):
      internal_folder_name = self.make_folder_name_for_data_class(frozenparamdict)
      print(internal_folder_name)
      save_folder = os.path.join(self._SAVE_TEXT_FILES_DIR,internal_folder_name)
      if not os.path.exists(save_folder):
         os.makedirs(save_folder)

      lande_time=self.get_lande_time(frozenparamdict)

      tmp = self._DATA_CLASS_DICT[frozenparamdict]
      # stat_list = ['dx_per_seg_var','x_per_seg_var', 'd2ax_per_mut_input']

      for lands in [True,False]:
         if tstat is None:
            tmp.write_tstat_efs_at_time_to_text_files(savedir=save_folder,lande=lands, time=lande_time)
         else:
            tmp.write_tstat_efs_at_time_to_text_files(tstat=tstat, savedir=save_folder, lande=lands, time=lande_time)

   def write_final_tstats_to_text_file_all(self,tfstat=None):
      for param_dict in self._PARAM_DICT_TRAJ_LIST:
         self.write_final_tstats_to_text_file(self.turn_dict_into_dict_key(param_dict),tfstat=tfstat)

   def write_tstats_at_lande_time_to_text_file_all(self,tstat=None):
      for param_dict in self._PARAM_DICT_TRAJ_LIST:
         self.write_tstats_at_lande_time_to_text_file(self.turn_dict_into_dict_key(param_dict),tstat=tstat)

   def write_bstats_to_text_file_all(self):
      for param_dict in self._PARAM_DICT_FULL_LIST:
         self.write_bstats_to_text_file(self.turn_dict_into_dict_key(param_dict))

   def write_some_hbstats_at_lande_and_end_time_to_text_file_all(self):
      for param_dict in self._PARAM_DICT_FULL_LIST:
         self.write_some_hbstats_at_lande_and_end_time_to_text_file(self.turn_dict_into_dict_key(param_dict))

   def summarize_bstat_data_all(self):
      for param_dict in self._PARAM_DICT_FULL_LIST:
         self.summarize_bstat_data_in_directory(self.turn_dict_into_dict_key(param_dict))

   def summarize_traj_data_all(self):
      for param_dict in self._PARAM_DICT_TRAJ_LIST:
         self.summarize_traj_data_in_directory(self.turn_dict_into_dict_key(param_dict))
         self.summarize_traj_final_data_in_directory(self.turn_dict_into_dict_key(param_dict))

   def summarize_bstat_data_in_directory(self,frozenparamdict):
      self._DATA_CLASS_DICT[frozenparamdict].summarize_bstats()

   def summarize_traj_data_in_directory(self,frozenparamdict):
      self._DATA_CLASS_DICT[frozenparamdict].summarize_all_traj()

   def summarize_traj_final_data_in_directory(self,frozenparamdict):
      self._DATA_CLASS_DICT[frozenparamdict].summarize_traj_final()

   def delete_full_sim_stats_that_have_been_read_all(self):
      for param_dict in self._PARAM_DICT_FULL_LIST:
         self.delete_full_sim_stats_that_have_been_read(self.turn_dict_into_dict_key(param_dict))

   def delete_traj_stats_that_have_been_read_all(self):
      for param_dict in self._PARAM_DICT_TRAJ_LIST:
         self.delete_traj_stats_that_have_been_read(self.turn_dict_into_dict_key(param_dict))

   def delete_full_sim_stats_that_have_been_read(self, frozenparamdict):
      self._DATA_CLASS_DICT[frozenparamdict].delete_all_read_bstat_runs()
      self._DATA_CLASS_DICT[frozenparamdict].delete_read_var0_runs()


   def delete_traj_stats_that_have_been_read(self, frozenparamdict):
      self._DATA_CLASS_DICT[frozenparamdict].delete_all_read_tstat_runs()

   def _make_data_classes(self):
      for dir in self._RESULTS_DIRECTORIES_LIST:
         if self._is_it_a_traj_data_class(dir) and self._is_it_a_full_data_class(dir):
            tmp = dataClass(base_dir=dir)
         elif self._is_it_a_full_data_class(dir):
            tmp = dataClassFullSims(base_dir=dir)
         else:
            tmp = dataClassTraj(base_dir=dir)
         tmp_stat_writer_param_dict = tmp.stat_writer_param_dict()
         if self._is_it_a_traj_data_class(dir):
            self._PARAM_DICT_TRAJ_LIST.append(tmp_stat_writer_param_dict)
         if self._is_it_a_full_data_class(dir):
            self._PARAM_DICT_FULL_LIST.append(tmp_stat_writer_param_dict)
         self._PARAM_DICT_LIST.append(tmp_stat_writer_param_dict)

         dclasskey = self.turn_dict_into_dict_key(tmp_stat_writer_param_dict)
         self._DATA_CLASS_DICT[dclasskey] = tmp

   def _is_it_a_traj_data_class(self,direct):
      """Check if the data class in directory 'direct'
      contains trajectory stats"""
      traj_folder = os.path.join(direct, "trajectories")
      if os.path.exists(traj_folder):
         return True
      else:
         return False

   def _is_it_a_full_data_class(self,direct):
      """Check if the data class in directory 'direct'
      contains full simulations stats"""
      full_folder = os.path.join(direct, "var_0_emp_vD")
      if os.path.exists(full_folder):
         return True
      else:
         return False

   def make_folder_name_for_data_class(self, frozenparamdict):
      tmp = self._DATA_CLASS_DICT[frozenparamdict]
      if tmp.stat_writer_param_dict() in self._PARAM_DICT_FULL_LIST:
         if tmp.param_dict['algorithm'] == 'approx':
            folder = "approx"
         else:
            folder = "full"
            if tmp.param_dict['fitness'] == 'parents':
               folder = os.path.join(folder,"parents")
            else:
               folder = os.path.join(folder, "offspring")
      else:
         folder = "pure_lande_traj"
      return folder

   @staticmethod
   def copy_directory_tree_summary_files_only(src, dest):
      print(dest)
      try:
         shutil.copytree(src, dest, ignore=shutil.ignore_patterns('*.py','*_have_read' '*.sh', '*_aRun','*_aRun_lande'))
      except OSError as e:
         # If the error was caused because the source wasn't a directory
         print(e.errno)
         if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
         else:
            print('Directory not copied. Error: %s' % e)


