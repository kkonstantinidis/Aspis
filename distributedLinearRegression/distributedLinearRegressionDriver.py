# Driver code for distributedLinearRegression.py
import matplotlib.pyplot as plt
import os
import sys

from distributedLinearRegression import *

# Maximum number of iterations of linear regression
max_iterations = 1000

# Size of data set
# n = 1000

# Dimensionality
# d = 100

# No. of workers
K = 15

# No. of adversaries
# q = 2

# Adversarial window length for Aspis+
adv_win_length = 50

# Replication (odd)
r = 3

# Random choice of Byzantine workers or not
# randomByzantines = True
randomByzantines = False

# Reversed gradient and constant distortion are supported
# err_mode = REV_GRAD
err_mode = CONSTANT
# err_mode = ALIE

# 2nd level of aggregation
# mode = COORD_MEDIAN
mode = GEOMETRIC_MEDIAN

# Which scheme to use
# approach = BASELINE
# approach = DETOX
# approach = ASPIS
# approach = ASPIS_PLUS

# Learning rate
# alpha = 0.001

# Tolerance of linear regression for termination criterion
epsilon = 0.00001

# Constant for Tikhonov regularization of the loss (ridge regression)
# zeta = 100

# Aspis+: How often to reset the agreements counter for detection at the PS level (in # of iterations)
det_win_length = 15

# Aspis+: randomly permute the file assignment after each iteration (adversarial indices won't change)
# permute_files = True
permute_files = False

# Random seed for the entire algorithm
seed = 428
# seed = None

# Whether to write log to file or not
# logToFile = False
logToFile = True

# Frequency (no. of iterations) of logging stats & storing loss value
pointFreq = 10

# Constant to use in reversed gradient and constant attack as multiplier
adversary_const = 10

# raw_input returns the empty string for "Enter"
if randomByzantines:
    yes = {'yes','y', 'ye', ''}
    no = {'no','n'}
    choice = input("Please confirm whether a random set of Byzantines should be chosen every " + str(adv_win_length) + " iterations:").lower()
    if choice in yes:
       assert 0 == 0
    else:
       assert 0 == 1

# Folder to store logs
ROOT_LOG_FOLDER = "./Tuning Logs"

for approach in [BASELINE]:
    for q in [6]:
        for n in [500]:
            for d in [100]:
                fig, ax = plt.subplots()
                lossLines = []
                figName = "approach={},n={},d={},K={},q={},r={},randomByzantines={},err_mode={},mode={},epsilon={},adversary_const={}" \
                    .format(approach, n, d, K, q, r, randomByzantines, err_mode, mode, epsilon, adversary_const)
                legends = []

                if logToFile:
                    orig_stdout = sys.stdout

                    # Create folder for storing log and loss of current experiment
                    directory = ROOT_LOG_FOLDER + "/Logs_" + figName
                    if not os.path.exists(directory):
                        os.mkdir(directory)

                    # Append to previous logs, if any
                    file_log = open(ROOT_LOG_FOLDER + "/Logs_" + figName + "/log.txt", 'a+')
                    sys.stdout = file_log

                # General setup: alpha: [5, 1, 0.5, 0.1, 0.01, 0.001, 0.0001], zeta: [0.1, 0.01, 0.001, 0.0001, 0]
                for alpha in [0.0001]:
                    for zeta in [0, 1]:
                        try:
                            loss, iterIdx = run(max_iterations, n, d, K, q, adv_win_length, r, randomByzantines, err_mode, mode, approach, alpha,
                                                epsilon, zeta, det_win_length, permute_files, seed, logToFile, pointFreq, adversary_const)
                        except ZeroDivisionError:
                            print("Error! Aborting due to ZeroDivisionError...\n")
                            continue
                        except RuntimeWarning:
                            print("Error! Aborting due to RuntimeWarning...\n")
                            continue

                        # Write the loss for this experiment in a file.
                        # Overwrite previous file, if any.
                        if logToFile:
                            file_loss = open(ROOT_LOG_FOLDER + "/Logs_" + figName + "/loss_" + "alpha_" + str(alpha) + "_zeta_" + str(zeta) + ".txt", 'w')
                            file_loss.write("loss = [" + ",".join(str(x) for x in loss) + "]")
                            file_loss.close()

                        # Add the loss to the plot
                        # curLossLine, = plt.plot(iterIdx, loss)
                        curLossLine, = plt.semilogy(iterIdx, loss)
                        lossLines += [curLossLine]
                        legends += ["alpha={},zeta={}".format(alpha, zeta)]

                ax.set_title(figName)
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Loss")
                plt.legend(lossLines, legends)

                # Option 1
                # plt.show()

                # ...or Option 2
                # plt.savefig("./Figures/" + figName)

                if logToFile:
                    sys.stdout = orig_stdout
                    file_log.close()