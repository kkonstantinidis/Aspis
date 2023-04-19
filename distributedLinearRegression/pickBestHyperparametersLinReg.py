# Given a folder with tuning results (loss) for learning rate (alpha) and Tikhonov constant (zeta) it gives the best hyperparameter values.
# This is for my work in Aspis+ paper preprint.
# Partially based on pickBestHyperparameters.py
#
# The input directory/files should have the filename format (prefix will be used to match filenames):
#   ROOT_L0G_FOLDER/folder/loss_alpha_0.0001_zeta_0.0001.txt
#   ROOT_L0G_FOLDER/folder/loss_alpha_0.0001_zeta_0.001
#   ROOT_L0G_FOLDER/folder/loss_alpha_0.0001_zeta_0.01
#
# The contents of each file should have the format of a list with loss values:
# loss = [13.24558781090534,0.006079297672876396,8.209186294293134e-05,1.5785077861463464e-06,2.9235328156642854e-08,5.387057311260199e-10]
import os
import matplotlib.pyplot as plt
import math

# Folder to store logs
ROOT_LOG_FOLDER = "./Tuning Logs"
# ROOT_LOG_FOLDER = "./Final Logs"

# Input folder within current directory
PATH = ROOT_LOG_FOLDER + r"/Logs_aspis_plus,n=1000,d=100,K=15,q=4,r=3,randomByz=True,err_mode=constant,err_choice=all,geometric_median,epsilon=1e-10,adversary_const=10"

# How many checkpoints from the end will be used to evaluate tuning (will take their average)
# Note: lastN <= no. of checkpoints. If not, it will use all checkpoints.
lastN = 2

# Trainings whose terminating loss exceeds this value will be ignored
MAX_LOSS = 10**10

# List with tuples of the form:
# (
#   alpha,
#   zeta,
#   loss at last lastN checkpoints,
# )
tuningList = []

# For plotting
lossPlotLines = []

# Read only the files with the training loss.
# As we only care about 1 folder we get only the next() of this generator.
path, dirs, files = next(os.walk(PATH))
files = [file for file in files if file.startswith("loss_")]

fig, ax = plt.subplots()

for filename in files:
    # Fetch alpha and zeta from the filename (strip out file extension first)
    filenameSplit = filename.rsplit('.', 1)[0].split('_')
    alpha, zeta = filenameSplit[2], filenameSplit[4]
    # print(alpha, zeta)

    fullpath = os.path.join(path, filename)
    fp = open(fullpath, 'r')

    # Each loss file contains exactly 1 line with the loss values
    line = fp.readlines()[0]
    lineFloats = line[line.index('[') + 1: line.index(']')]

    # Read loss from file
    loss = list(map(float, lineFloats.split(',')))

    # Last lastN measurements that will be used to choose the best one
    lastN_loss = loss[-lastN:]

    # Average loss of last N checkpoints, as needed
    lastN_loss_avg = sum(lastN_loss)/min(lastN, len(lastN_loss))

    # Ignore excessive values of loss
    if any([math.isnan(x) for x in loss]) or lastN_loss_avg > MAX_LOSS:
        continue

    # Append current training results to plots of loss
    curLossPlotLine, = plt.semilogy(loss)
    lossPlotLines += [curLossPlotLine]

    # Store the values
    tuningList += [(alpha, zeta, lastN_loss_avg)]

    fp.close()

# Add alpha and zeta as legend to each plot
ax.set_xlabel("Checkpoints")
ax.set_ylabel("Loss")
plt.legend(lossPlotLines,
           ["alpha = " + tuningList[i][0] + ", zeta = " + tuningList[i][1] for i in range(len(tuningList))],
           loc='center left', bbox_to_anchor=(1, 0.5))

# Option 1
# plt.show(bbox_inches='tight')

# ...or Option 2
# Exclude the prefix of the path, i.e., the root folder that logs are stored.
plt.savefig("./Tuning Figures/" + PATH[len(ROOT_LOG_FOLDER)+1:], bbox_inches='tight')

tuningListSorted = sorted(tuningList, key=lambda x: x[-1])

print("Number of reported experiments: ", len(tuningListSorted))

if tuningListSorted:
    print("The best parameter sets (alpha, zeta, loss) are: ")
    for x in tuningListSorted:
        print(x)
else:
    print("All tunings of this experiment have excessive loss or no tunings were found in this folder!")

