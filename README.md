# PolygenicAdaptation1D
### This is the code for the polygenic trait simulations from [Hayward and Sella (2019)](https://www.biorxiv.org/content/10.1101/792952v2)


# Overview

The purpose of the code is to simulate a constant size diploid population that is at steady-steady under stabilizing selection, but at time 0 zero is subject to a sudden shift in optimum phenotype. For further details about the scenario and the theory behind the simulations please see [Hayward and Sella (2019)](https://www.biorxiv.org/content/10.1101/792952v2). 


The code folder contains three programs all of which run using Python 3.7. One simulates populations evolving (which we refer to as 'population simulations'). The second simulates allele trajectories, given a particular trajectory of the mean phenotype over time (which we refer to as 'trajectory simulations'). And the third summarizes statistics recorded over many of the trajectory and population simulations.

Please note that both the *full model* and *all allele* simulation in [Hayward and Sella (2019)](https://www.biorxiv.org/content/10.1101/792952v2) are both referred to as *population simulations* here, because they both simulate the dynamics of a population. And the *single allele* simulation (called *only allele* in early versions) in [Hayward and Sella (2019)](https://www.biorxiv.org/content/10.1101/792952v2) are referred to as *trajectory simulations* here, because they only follow trajectories of alleles given a specific phenotypic effect.


These programs can be run using command line, but the juypter notebook **getting\_started\_notebook.ipynb** is highly reccommended to help you get started.

## Population simulations

Use simulate\_populations\_argparse.py to simulate a the full population (either using the computationally expensive *full model* simulation, or the faster approximation (*all allele*) simulation, which just simulates a population by tracking all the alles segregating in the population).
The program can be run on the command line and takes the following parameters:  

-N population size (default=5000) 
 
-Vs squared width of fitness function. If you choose Vs negative it will automaticall be set to 2N (default=-1) 
 
-U mutation rate per haplotype genome per generation (default=0.01)  

-E2Ns expected value of the gamma distribution of scaled steady-state selection coefficients (which are the squared phenotypic effects in units of Vs/(2N)) (default=10.0)  

-V2Ns variance of gamma distribution of scaled steady-state selection coefficients. If you choose Vs negative it will automaticall be set to E2Ns<sup>2</sup>, making the distribution into an exponential distribution with expected value E2Ns (default=-1) 

-D\_s0, --shift\_s0 The shift in units of sigma\_0 (note that (sigma_0)<sup>2</sup> =V<sub>A</sub>(0) in the manuscript), which is the expected standard deviation of the phenotype distribution before the shift in optimum (at steady-state under stabilizing selection) (Default=2)

-hi --histograms Specifying -hi, will cause the simulator to save data for histogram type statistics for segregating alleles, which are binned either by initial MAF (MAF at the time of the shift), or by squared effect size bins. *Recommended*. Default is not to do so. 

-bTN --burn\_time\_N The burntime in units of population size - the burn time before the shift in optimum. (Default= 5 for speed, but recommend 10 for any final conclusions)

-r --runs The number of runs of the population simulations implemented by this single run of simulate\_populations\_argparse.py. Choos runs >1 and run this program fewer times, to make your simulations *less* parallel, if that is convenient for you

-lTN --lag\_time\_N The lag time in units of pop size, i.e. the burn time between runs. If runs>1, then the state of the population right before the shift is copied and saved, and the copied population is later be used for a different run. For this to reasonable, this copied population must have a second burn time, which we call a lag time. For final conclusions, make the lag time the same as the burn time. But to quickly generate some rough simulation results, a shorter lag time can be used. (Default= 1, but recommend 10 for any final conclusions)

-a, --algorithm This must be either 'approx' or 'exact'. 'approx' runs the *all allele* simulation, which does not track individuals, only all of the alleles segregating in the population. 'exact' runs the *full model* simulation, which realizes the full model and tracks individuals in the population too. (Default=approx)

-o --offspring This is only relavent if the 'exact' algorithm is being used. Specifying -o, will make the simulations sample by offspring fitness instead of parents. The default is to use parent fitness which results in faster simulations than using offspring fitness

-sD, --save_directory The directory in which you want the folders with your simulation results to be saved

## Trajectory simulations

simulate\_trajectories\_argparse.py simulates allele trajectories of alleles segregating at the time of the shift, with specific squared effect sizes in units of &delta;<sup>2</sup> =Vs/(2N) (=steady-state scaled selection coefficients); this is referred to as the *single allele* simulation in the paper (or *only allele* (OA) in early versions). Under the recommended default setting, the simulator will average results for a given squared effect size over the corresponding initial MAF distribution. The program can be run on the command line and takes the following parameters: 

-nM --number_mutants The number of alleles (per tuple) that we simulate in this trajectory run. With the simulator defaults, a *tuple* will just correspond to a particular choice of scaled steady-state selection coefficient (=squared phenotypic effect in units of Vs/(2N)). And taking nM=500 for example, means that for each of the 13 scaled selection coefficients, ranging from 0.1 to 100, that the simulator by default uses, it will simulate trajectories of 500 alleles with phenotypic effect aligned to the shift in optimum and 500 alleles with phenotypic effect opposing the shift in optimum. For reasonable results you probably want to do, say, a couple 1000 mutants per run, and also do a number of such runs. (default = 1000)

-oL --only\_Lande This flag causes the simulator to simulate allele trajectories using Lande's approximation for D(t). So in this case we use neither a distance function obtained from population simulations, nor Lande's approximation for D(t) corresponding to parameters already used to generate some population simulations. The is not to do only\_Lande. If you do only\_Lande, you must specify a lande_Folder too, and you should also specify the parameters necessary to generate Lande's approximation for D(t): Namely N, Vs, shift\_s0, sigma\_0\_del (sigma\_0\_del is the steady-state expected standard deviation in the phenotype distribution in units of &delta;=root(Vs/(2N))) . We will sometimes refer to these trajectory simulations as *pure Lande simulations*                             

-lF --lande\_Folder Specify this folder if you want to run trajectory simulations using Lande's approximation for D(t), and your parameter choices for the approximation (N, Vs, shift\_s0, sigma\_0\_del) do NOT correspond to parameters that you previously used for population simulations. 

-sF --simulation\_Folder The simulation folder must be a results folder of the form HASH\_ID\_pD, which contains simulation results for the population simulations. The simulator will read the distance function corresponding to average distance over the population simulations. You should have run a fairly large number of population simulations for this to be a reasonable thing to do. *No population genetics parameters need to be specified in this case*, since the the simulator will use the same ones that were used for the population simulations in simulation\_Folder. The trajectory results output by this program will also be saved in the simulation\_Folder, within the folder 'trajectories'


-l --lande Specifing this, means we are saving the results in a population simulation\_Folder (in contrast to when only\_Lande is specified), but wish to to use Lande's approximation for D(t) with parameters corresponding to the parameters used for the population simulations that generataed the population simulation results in simulation\_Folder. Default is to *not* to do this, but rather to use  D(t) from the populatin simulations in simulation\_Folder. 

-psF --parent\_simulation\_Folder If this is not an empty string, this overrides simulation\_Folder. If there is a parent\_simulation\_Folder, the program will search in the parent\_simulation\_Folder, for a simulation folder (of the form HASH\_ID\_pD) with results from population simulations using the population genetic parameters specified. i.e. you should specify the values of N, U, Vs, E2Ns, V2Ns, shift\_s0 and if a simualtion folder exists in parent\_simulation\_Folder where population simulations were run with those parameters, then the program will locate it and run trajectories using the distance function corresponding to the average over the population simulations in that folder. The trajectory results output by this program will also be saved in that folder, within the folder 'trajectories'


-N population size (default=5000). Only need to specify this if you are have also specified a lande\_Folder or a parent\_simulation\_Folder 
 
-Vs squared width of fitness function. If you choose Vs negative it will automaticall be set to 2N (default=-1) Only need to specify this if you are have also specified a lande\_Folder or a parent\_simulation\_Folder 

-D\_s0, --shift\_s0 The shift in units of sigma\_0 (note that (sigma_0)<sup>2</sup> =V<sub>A</sub>(0) from the manuscript), which is the expected standard deviation of the phenotype distribution before the shift in optimum (at steady-state under stabilizing selection) (Default=2). Only need to specify this if you are have also specified a lande\_Folder or a parent\_simulation\_Folder 

-sigma\_0\_del Steady-state standard deviation of the phenotype distribution, sigma\_0 (=root(V<sub>A</sub>(0)) in units of &delta;= root(Vs/(2N)). Only specify this if you are have also specified a lande\_Folder, and are using the flag --only\_Lande. (Default = 20)

-U Mutation rate per haplotype genome per generation (default=0.01). Only specify if you are using a parent\_simulation\_Folder  

-E2Ns Expected value of the gamma distribution of scaled steady-state selection coefficients (which are the squared phenotypic effects in units of Vs/(2N)) (default=10.0).  Only specify if you are using a parent\_simulation\_Folder  

-V2Ns	Variance of gamma distribution of scaled steady-state selection coefficients. If you choose Vs negative it will automaticall be set to E2Ns<sup>2</sup>, making the distribution into an exponential distribution with expected value E2Ns (default=-1)  Only specify if you are using a parent\_simulation\_Folder 

## Results of Population and Trajectory simulations

The results of both the population and trajectory simulations are saved in folders of the form HASH\_ID\_pD. Here HASH\_ID is a string of letters and numbers that is a hash of the parameter values chosen to generate the simulations. In this way, simulation results that use the same parameters are saved in the same folder. Results of individual simulations are saved in files of the form RUN\_ID\_aRun, with RUN\_ID being a string of numbers uniquely identifying that particular simulation run. Thus if you run the population or trajectory simulations many times for the same parameters (which is advisable), there will be files corresponding to many RUN\_IDs in the same HASH\_ID\_pD folder.

Read the file 'identifiers.txt', inside the HASH\_ID\_pD folder to check what parameter choices simulations in that HASH\_ID\_pD folder correspond to.

There are many folders inside each HASH\_ID\_pD, and each folder contains simulation results for a different statistic recorded during the population simulations. The text file 'read\_me\_about\_stats.txt' will also be saved inside HASH\_ID\_pD. Read this file to understand the details of how the population simulation results are saved in these folders, and what the various statistics are.

Trajectory simulation results are saved in a folder called 'trajectories' inside the HASH\_ID\_pD folder. Inside 'trajectories', there will be 13 (if the simulation defaults are used) folders corresponding to each of the 13 scaled selection coefficeints (=squared phenotypic effects in units Vs/(2N)). We refer to these folders as each being for a different tuple, and they have the form XI\_xi\_0\_S\_a<sup>2</sup>\_t\_time\_tD, with xi, a<sup>2</sup> and time being numbers. With simulation defaults the value of the scaled selection coefficient, a<sup>2</sup>, will be different for each folder. (The value of the scaled With the simulation defaults you will always have xi = -shift\_s0, and time =0.) As an example, a folder called XI\_-2\_0\_S\_0\_1\_t\_0\_tD, contains simulation results for alleles with scaled selection coefficient of 0.1. Inside each tuple folder, there are folders containing simulation results with different statistics recorded during the trajectory simulations, with alleles of effect a<sup>2</sup>. The two text files 'read\_me\_about\_traj\_stats.txt' and 'read\_me\_about\_final\_traj\_stats.txt' and will be saved inside the 'trajectories' folder. Read these files to understand the details of how the trajectory simulation results are saved inside the tuple folders, and what the various statistics are.

Because it is so important and the name could potentially be confusing, we mention one of the allele trajectory statistics here: 'd2ax\_scaled\_per\_mut\_input'.
This is the contribution *per unit mutational input* to the change in mean phenotype from standing variation, divided by the unitless quantity (&Lambda;*&delta;)/V<sub>A</sub>(0). It is most interesting to look at the combined contribution from both aligned and opposing alleles; the scaling should make results from simulations with different shift sizes, and mutation rate, but with the same effect size distribution of incoming mutations, approximately coincide. This is the statistic that in units of &delta; and averaged over initial MAF, we expect the average over many simulations, as a function of phenotypic effect a, to look approximately like v(a) (if &Lambda;>> &delta;) at the end of the rapid phase, and over the long-term we expect it to look approximately like f(a) in Lande cases, and (1+A)\*f(a) in non-Lande cases.


## Summarizing and saving

summarize\_save\_argparse.py summarizes the statistics recorded over many runs of the population or trajectory simulations. It saves these summaries in files called 'summary', which contain means, standard errors and standard deviations of the statistic over many runs. It also deletes the results of individual runs once these summaries have been generated. You may not always want to do the deleting part if you have performed very computationally expensive runs. 

*Note:* Even after you have summarized the results corresponding to a particular set of parameters in a HASH\_ID\_pD directory (and deleted the results of individual runs), you can always perform more runs with the same parameters and rerun summarize\_save\_argparse.py on the folder conting that HASH\_ID\_pD. The resulting 'summary' files will be for *both* the runs previously summarized and the new runs. The reason this works is that summarize\_save\_argparse.py also saves the sums and sum of squares of each statistic, and the number of runs combined so far to produce those sums in a files called 'combined_runs'.


summarize\_save\_argparse.py also saves some of the most important of the results to text files which you can then read using the program of your choice.

The program summarize\_save\_argparse.py can be run on the command line and takes the following parameters:
  
-rD The directory in which all of the results folders that you want to summarize are stored and are saved. (These are the results folders have the form HASH\_ID\_pD). The simulator will read the names of the result folders in this directory, and it will summarize the simulation results in each of them. 

-sD The directory in which you want the resulting '.txt' files with summaries of some of your results to be saved.

# Code files

### Population simulation specific files

simulate\_populations\_class.py contains the class to simulate the full population (both using the full model and the approximation). It uses the population classes in population_class.py.

### Trajectory simulation specific files

simulate\_trajectories\_class.py contains the class to simulate individual allele trajectories, without simulating a whole population. It uses the trajectory class in trajectory_class.py

### Summarizing and saving specific files

summarize\_save\_class.py contains the class to summarize the statistics collected over many simulations; and to save these summaries.

### Other files

read\_data.py contains the three classes to read statistics recorded from both population and trajectory simulations. You may find these classes useful for reading simulation results saved in the HASH\_ID\_pD results files. Use dataClassFullSims if the simulation results in HASH\_ID\_pD were generated from population simulations. Use dataClassTraj if the simulation results in HASH\_ID\_pD were generated from trajectory simulations. And use dataClass if HASH\_ID\_pD contains results generated from both population and trajectory simulations.

record\_stats.py contains the class used to record statistics collected from both population and trajectory simulations

mutation\_class.py has container classes for mutations arising in the population, and is used for both population and trajectory simulations

combined\_theory.py contains a couple of useful classes based on theory such as the expected MAF distribution at steady-state of an allele with particular scaled selection coefficient.

plot\_class.py is messy, but contains a class to quickly generate a number of basic plots of some of the statistics generated by simulations. It uses the plotting functions in plot\_functions.py, and the names in plot\_names.py


