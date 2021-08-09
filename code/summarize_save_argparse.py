"""
summarize_save_argparse.py summarizes the statistics recorded over many runs of the population or
trajectory simulations. It saves these summaries in files called 'summary', which contain means, standard errors and
standard deviations of the statistic over many runs. It also deletes the results of individual runs once these
summaries have been generated. You may not always want to do this if you have performed very computationally
expensive runs.
"""
import os
import argparse
from summarize_save_class import SumAndSave

# Change my_base_directory_local to be the in which you place the code folder
# so that it is ".../yourfolder"
my_base_directory_local = '/Users/Alexandra/pcloud_synced/PycharmProjects/Polished_code_python3/'

read_directory = os.path.join(my_base_directory_local,'results_local_test')
save_output_dir =  os.path.join(my_base_directory_local,'results_local_test_output')

parser = argparse.ArgumentParser(description='Summarize simulation results from populations sims. '
                     'Delete files that have been read. Save some results as text files')

parser.add_argument('-rD', '--read_directory', type=str, default = read_directory,help='The directory in which '
         'the results folders are stored and are saved. Results folders have the form "...._pD')
parser.add_argument('-sD', '--save_output_dir', type=str, default = save_output_dir,help='The directory in which '
         'you want to save output. E.g. Text files with some statistics from the runs')
args = parser.parse_args()


# parent directory is the directory were you results directories are stored
# save_output_dir is the directory you want new output to be saved
my_sum_and_save= SumAndSave(parent_dir=args.read_directory,save_output_dir=args.save_output_dir)

# summarize all of the statistics saved during the full runs if any
my_sum_and_save.summarize_bstat_data_all()
# summarize all of the statistics saved during the traj runs if any
my_sum_and_save.summarize_traj_data_all()

# Delete the runs you have already read and incorporated into the sumary
# If you have performed lots of time-consuming runs, you may not want to do this
my_sum_and_save.delete_full_sim_stats_that_have_been_read_all()
my_sum_and_save.delete_traj_stats_that_have_been_read_all()

# Write some all the regular stats you saved into text files
# The files have the format
# Line 1= space seperated times at which stats were collected
# Line 2 = space seperated average over runs of stat at times in line 1
# Line 2 = space seperated 1.96*standard error in stat over runs
# at times in line 1
my_sum_and_save.write_bstats_to_text_file_all()

# write the final traj stats to a text file
my_sum_and_save.write_final_tstats_to_text_file_all()
my_sum_and_save.write_tstats_at_lande_time_to_text_file_all()

tstat_list = ['dx_per_seg_var','x_per_seg_var', 'd2ax_per_mut_input']

# Write some all the histogram stats you saved into text files
# The text files have the format
# Line 1= space seperated boundaries of histogram bins
# Line 2 = space seperated average over runs of stat in bins in line 1
# Line 2 = space seperated 1.96*standard error in stat over runs
# in bins in line 1
# The time at which the stats were saved are saved as
# name of text file in the folder
# this will only do something if you ran your simulations with histograms
my_sum_and_save.write_some_hbstats_at_lande_and_end_time_to_text_file_all()
