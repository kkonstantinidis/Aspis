# Given a folder with Monte Carlo simulations (using multiple seeds) of the loss it gives the average loss across all the runs.
#
# The input directory/files should have the filename format (prefix will be used to match filenames):
#   ROOT_L0G_FOLDER/folder/loss_alpha_....txt
#   ROOT_L0G_FOLDER/folder/loss_alpha_....txt
#   ROOT_L0G_FOLDER/folder/loss_alpha_....txt
#
# The contents of each file should have the format of a list with loss values:
# loss = [13.24558781090534,0.006079297672876396,8.209186294293134e-05,1.5785077861463464e-06,2.9235328156642854e-08,5.387057311260199e-10]
import os
import math
import numpy as np

# Folder to store logs
ROOT_LOG_FOLDER = "./Final Logs"
# ROOT_LOG_FOLDER = "./Tuning Logs"

# Input folder within current directory
PATH = ROOT_LOG_FOLDER + r"/LR-F-A15/Logs_aspis_plus,n=1000,d=100,K=15,q=6,r=3,randomByz=True,err_mode=constant,err_choice=all,geometric_median,epsilon=1e-10,adversary_const=10_MC"

# Trainings whose terminating loss exceeds this value will be ignored
MAX_LOSS = 0.1

# Read only the files with the training loss.
# As we only care about 1 folder we get only the next() of this generator.
path, dirs, files = next(os.walk(PATH))
files = [file for file in files if file.startswith("loss_alpha_")]

# Computes the mean and st. deviation across a list of lists and outputs two 1-D lists:
#   - List with means.
#   - List with st. deviations.
# Each input list may be of different length so that is taken into account.
# Example (mean):
# [[1,2,3],[9,3],[2,5,6,9,8]] -> [4.0 3.3333333333333335 4.5 9.0 8.0]
# https://stackoverflow.com/a/59281468/1467434
def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

# List of lists of losses for all experiments versus no. of iterations
monteCarloLosses = []

# Fetch all files, each for a different seed
for filename in files:
    fullpath = os.path.join(path, filename)
    fp = open(fullpath, 'r')

    # Each loss file contains exactly 1 line with the loss values
    line = fp.readlines()[0]
    lineFloats = line[line.index('[') + 1: line.index(']')]

    # Read loss from file
    loss = list(map(float, lineFloats.split(',')))

    # Ignore excessive values of final loss
    if any([math.isnan(x) for x in loss]) or loss[-1] > MAX_LOSS:
        continue

    # Store the values
    monteCarloLosses += [loss]

    fp.close()


print("Number of extracted experiments: ", len(monteCarloLosses))

if len(monteCarloLosses) == 0:
    print("No experiments converged to required loss {}".format(MAX_LOSS))
else:
    # Average across Monte Carlo experiments
    avgMonteCarloLoss, stdMonteCarloLoss = tolerant_mean(monteCarloLosses)
    print(avgMonteCarloLoss)
    print(stdMonteCarloLoss)

    # Write to a separate file
    monteCarloFileName = PATH + "/loss_avg_std_Monte_Carlo.txt"
    file_loss = open(monteCarloFileName, 'w')
    file_loss.write("-------------- Generated file --------------\n")
    file_loss.write("No. of extracted experiments: {}\n".format(len(monteCarloLosses)))
    file_loss.write("Max allowed loss for convergence: {}\n\n".format(MAX_LOSS))
    file_loss.write("--------Monte Carlo Average Loss------------\n".format())
    file_loss.write("avg_loss = [" + ",".join(str(x) for x in avgMonteCarloLoss) + "]\n\n")
    file_loss.write("-----Monte Carlo Loss St. Deviation---------\n".format())
    file_loss.write("std_loss = [" + ",".join(str(x) for x in stdMonteCarloLoss) + "]\n")
    file_loss.write("--------------------------------------------\n")
    file_loss.close()