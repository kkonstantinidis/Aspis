# Baseline, Aspis, Aspis+, DETOX implementation of mini-batch gradient descent for linear regression
import copy
import itertools
import random
import operator as op
from functools import reduce
import pandas
from collections import defaultdict
import numpy as np
from itertools import chain, product
from geom_median.numpy import compute_geometric_median
from collections import Counter
import time
from typing import Any
from datetime import datetime
import pytz

from distributedLinearRegressionConstants import REV_GRAD, CONSTANT, ALIE, BASELINE, DETOX, ASPIS, ASPIS_PLUS, GEOMETRIC_MEDIAN, COORD_MEDIAN

#######################################################################################################################
# Helper functions
#######################################################################################################################
# Returns n choose r
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

# Generate a random list of Byzantines (1-indexed)
def generateRandomByzantines(K, q):
    return random.sample(range(1,K+1), q)

def shouldDistort(approach, file, A, D):
    if approach in [BASELINE, DETOX]:
        # Distort since worker is adversarial
        return True

    elif approach == ASPIS:
        # Attack ATT-2 on Aspis

        # Group of workers processing the current file
        group = set(file)

        # Set of adversaries in current group
        cur_group_adversaries = set(A) & group

        # Set of non-adversaries in current group
        cur_group_honest = group - cur_group_adversaries

        # If either the group is fully adversarial (cur_group_honest == set()) or its non-adversaries are in the "disagreement set", the adversary chooses to distort.
        return cur_group_honest.issubset(D)

    elif approach == ASPIS_PLUS:
        # Attack ATT-3 on Aspis+
        group = set(file)
        cur_group_adversaries = set(A) & group
        cur_group_honest = group - cur_group_adversaries

        # Distort the current file if its group has adversarial majority
        return len(cur_group_adversaries) > len(cur_group_honest)

def distortGradient(g: Any, err_mode: str, adversary_const: int):
    if err_mode == REV_GRAD:
        # Reversed gradient distortion
        return -adversary_const*g
    elif err_mode == CONSTANT:
        # Constant distortion
        return -adversary_const*np.ones(g.shape)
        # return -100*np.ones(g.shape, dtype=float_type)
    else:
        assert 0 == 1, "This err_mode is not supported"

# Majority voting of gradients
# @param grads
#   List of np.ndarray of gradients to be used as input. All of them should be of same dimension
def _grad_majority_vote(grads):
    _maj_counter = 0
    _maj_grad = None
    for i, grad in enumerate(grads):
        if _maj_counter == 0:
            _maj_grad = grad
            _maj_counter = 1
        elif np.array_equal(grad, _maj_grad):
            _maj_counter += 1
        else:
            _maj_counter -= 1

    return _maj_grad

# Randomly permutes the file assignments by reassigning the values {1,2,...,K} to new values {p(1),p(2),...,p(K)}.
# Then, for each file (represented by its set of workers - ranks), it re-assigns its worker labels to the permuted ones
# p(i), i = ... It makes the same change in hashtable from file indices to workers.
def permuteFiles(K, files, workerFiles):
    permutedRanks = list(range(1, K + 1))
    random.shuffle(permutedRanks)

    # For each file, replace the worker ranks by their permutations
    newFiles = []
    for file in files:
        newFiles += [[permutedRanks[worker-1] for worker in file]]

    # For each worker, map its permutation to the same files since the file assignment has now changed
    newWorkerFiles = defaultdict(list)
    for worker in workerFiles:
        newWorkerFiles[permutedRanks[worker-1]] = workerFiles[worker]

    files = newFiles
    workerFiles = newWorkerFiles
    return files, workerFiles

# Aspis+: Adversarial detection using degree-based argument
# See byzshield_master.py
def _attempt_degree_detection(K, i, det_win_length, agreements, detectedWorkers, files, fileGrads, A, q, pair_groups, r):
    # The +1 is due to the iteration index being 0-indexed for proper counting of agreements
    if i == 0 or i%det_win_length == 0:
        agreements = Counter()
        detectedWorkers = []

    for fileInd, file in enumerate(files):
        for worker1, worker2 in product(file, file):
            if worker1 >= worker2: continue

            worker1_grad = fileGrads[fileInd][file.index(worker1)]
            worker2_grad = fileGrads[fileInd][file.index(worker2)]

            agreement = True

            if not np.array_equal(worker1_grad, worker2_grad):
                agreement = False

                if worker1 not in A and worker2 not in A:
                    assert 0 == 1, "Error! Honest workers disagree. Check for precision issues."

            if agreement: agreements[tuple([worker1, worker2])] += 1

    Gdegree = defaultdict(int)

    # The +1 is due to the iteration index being 0-indexed for proper counting of agreements
    windowIterCtr = (i % det_win_length) + 1

    for worker1 in range(1, K+1):
        for worker2 in range(worker1+1, K+1):
            if agreements[(worker1, worker2)] == windowIterCtr*pair_groups:
                Gdegree[worker1] += 1
                Gdegree[worker2] += 1

    for worker in range(1, K+1):
        if Gdegree[worker] < K-q-1:
            if worker not in detectedWorkers: detectedWorkers += [worker]

    # print("DEBUG_PS_BYZ: BEFORE degree-detectedWorkers: {}".format(detectedWorkers))

    detectedWorkers = detectedWorkers[-q:]

    cleanGradient = None
    for fileInd, file in enumerate(files):
        for worker in file:
            if worker not in detectedWorkers:
                cleanGradient = fileGrads[fileInd][file.index(worker)]
                break

        if cleanGradient is not None: break

    for fileInd, file in enumerate(files):
        if len(set(file) & set(detectedWorkers)) == 0:
            # Un-distorted group
            continue
        elif len(set(file) & set(detectedWorkers)) == r:
            # Fully distorted group
            for worker in file:
                fileGrads[fileInd][file.index(worker)] = copy.deepcopy(cleanGradient)
        else:
            # Partially distorted group
            cleanGroupGradient = None
            for worker in file:
                if worker not in detectedWorkers:
                    cleanGroupGradient = fileGrads[fileInd][file.index(worker)]
                    break

            for worker in file:
                if worker in detectedWorkers:
                    fileGrads[fileInd][file.index(worker)] = copy.deepcopy(cleanGroupGradient)

    # print("DEBUG_PS_BYZ: degree-detectedWorkers: {}".format(detectedWorkers))

    return agreements, detectedWorkers, fileGrads
#######################################################################################################################

#######################################################################################################################
# Full algorithm to run from the driver code in distributedLinearRegressionDriver.py
#######################################################################################################################
def run(max_iterations: int,
        n: int,
        d: int,
        K: int,
        q: int,
        adv_win_length: int,
        r: int,
        randomByzantines: bool,
        err_mode: str,
        mode: str,
        approach: str,
        alpha: float,
        epsilon: float,
        zeta: float,
        det_win_length: float,
        permute_files: bool,
        seed: int,
        logToFile: bool,
        pointFreq: int,
        adversary_const: int):
    #######################################################################################################################
    # Task assignment and choice of Byzantines
    #######################################################################################################################

    assert r%2 and 3 <= r <= K, "Invalid r"
    assert q < K/2, "Too many adversaries"
    # assert q >= (r+1)/2, "Too few adversaries"
    assert K%r == 0, "DETOX requires r|K"

    print("Running distributed linear regression with arguments: {}".format(locals()))

    # Get the current for Los Angeles
    tz = pytz.timezone('America/Los_Angeles')
    datetime_now = datetime.now(tz)
    print("LA current time:", datetime_now.strftime("%m-%d-%Y %H:%M:%S"))

    t0 = time.time()

    # Set random seed for the algorithm (for generating the adversaries & the data)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


    # Global float dtype
    # float_type = np.float32

    # No. of votes needed for majority
    rPrime = (r + 1) // 2

    # Set of all workers (1-indexed)
    W = set(range(1,K+1))

    # List of files. Each element is the set of workers that have it -1 (0-indexed)
    files = None
    A = None
    D = None
    if approach == BASELINE:
        # Baseline uncoded
        files = list([x] for x in range(1,K+1))

        if randomByzantines:
            A = generateRandomByzantines(K, q)
        else:
            # Set of adversaries is wlog {1,...,q}
            A = list(range(1,q+1))

    elif approach == DETOX:
        # DETOX
        files = []
        groupFirstWorker = 1
        for i in range(K//r):
            files += [list(range(groupFirstWorker, groupFirstWorker+r))]
            groupFirstWorker += r

        if randomByzantines:
            A = generateRandomByzantines(K, q)
        else:
            # Based on test_3.py
            # Set of adversaries (worst attack)
            if q % rPrime == 0:
                A =  list(chain.from_iterable([list(range(i, i+rPrime)) for i in range(1, (q//rPrime)*r, r)]))
            else:
                A = list(chain.from_iterable([list(range(i, i+rPrime)) for i in range(1, (q//rPrime)*r, r)]))
                rest = q % rPrime
                A.extend(list(range((q//rPrime)*r+1, (q//rPrime)*r+1 + rest)))

    elif approach == ASPIS:
        # Aspis
        files = list(list(x) for x in itertools.combinations(W, r))

        if randomByzantines:
            A = generateRandomByzantines(K, q)

            # Set of non-adversaries with whom the adversaries will always disagree, WLOG choose them
            # to be the first q honest workers
            H = list(W - set(A))
            D = H[:q]
        else:
            # Set of adversaries is wlog {1,...,q}
            A = list(range(1,q+1))

            # Set of non-adversaries with whom the adversaries will always disagree, WLOG choose them to be {q+1,q+2,...,2q}
            D = set(range(q + 1, q + q + 1))

        assert len(files) == ncr(K, r), "Aspis file allocation failed!"

    elif approach == ASPIS_PLUS:
        # Aspis+

        # Available designs
        if K == 9:
            df = pandas.read_excel('bibd_9_3_1.xlsx', header = None)
            pair_groups = 1
        elif K == 10:
            df = pandas.read_excel('bibd_10_3_2.xlsx', header = None)
            pair_groups = 2
        elif K == 15:
            df = pandas.read_excel('bibd_15_3_1.xlsx', header=None) # STS(15)
            pair_groups = 1
        elif K == 16:
            df = pandas.read_excel('bibd_16_3_2.xlsx', header = None)
            pair_groups = 2
        elif K == 21:
            df = pandas.read_excel('bibd_21_3_1.xlsx', header=None) # STS(21)
            pair_groups = 1
        elif K == 25:
            df = pandas.read_excel('bibd_25_3_1.xlsx', header=None) # STS(25)
            pair_groups = 1
        elif K == 7:
            df = pandas.read_excel('fano_plane.xlsx', header = None)
            pair_groups = 1
        else:
            assert 0 == 1, "Error! No known design for Aspis+ for this choice of K."

        # pair_groups = int(input("Enter lambda of the design (common files between any pair of workers): "))

        # As the files use 0-indexing, make the workers 1-indexed
        df += 1

        files = []
        maxWorkerInd = float('-inf')
        for _, row in df.iterrows():
            maxWorkerInd = max(maxWorkerInd, max(row))
            files += [list(row)]

        assert maxWorkerInd == K, "Invalid Aspis+ design for this choice of K"
        assert randomByzantines, "Only random choice of Byzantines is supported in Aspis+!"

        A = generateRandomByzantines(K, q)

    else:
        assert 0 == 1, "Invalid approach!"

    # Convert A to a set
    A = set(A)

    assert len(A) == q, "Byzantine set has not been populated correctly!"

    # Assign worker to files
    # Key: worker index (1-indexed)
    # Value: set of file indices (0-indexed) that the worker is assigned to
    workerFiles = defaultdict(list)
    for fileInd, file in enumerate(files):
        for worker in file:
            workerFiles[worker] += [fileInd]

    #######################################################################################################################
    # Linear regression using distributed gradient descent
    #######################################################################################################################

    # Data points: n x (d+1)
    X = np.random.normal(0, 1, size = (n,d))

    # Actual model
    w = np.random.normal(0, 1, size = (d,1))

    # Data point labels: n x 1
    y = np.matmul(X, w)

    # No. of files
    f = len(files)

    assert f <= n, "The number of files cannot exceed the number of samples!"

    # Partition the n samples of the dataset into f files row-wise
    # See https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
    Xsplit = np.array_split(X, f, axis = 0)

    # Partition the n labels of the dataset into f files row-wise
    ysplit = np.array_split(y, f, axis = 0)

    # Initialize random model parameters to iteratively optimize
    wt = np.random.normal(0,1, size = (d,1))

    # Loss at each iteration
    loss = []

    # Iteration indices to keep stats for
    iterIdx = []

    # Aspis+
    detectedWorkers = []
    agreements = Counter()

    # Distributed full gradient descent for linear regression
    i = 0
    while 1:
        # File gradients
        # Key: File index (0-indexed), Value: list of gradients of the workers assigned to the file
        fileGrads = defaultdict(list)

        for worker in range(1, K+1):
            for fileInd in workerFiles[worker]:
                # We need to maintain ordering of the gradients for each file as Aspis+ detection relies on that.
                # Hence, initialize list to store r gradients.
                if fileInd not in fileGrads:
                    if approach in [DETOX, ASPIS, ASPIS_PLUS]:
                        fileGrads[fileInd] = [None]*r
                    elif approach == BASELINE:
                        fileGrads[fileInd] = [None]
                    else:
                        assert 0 == 1, "Invalid approach!"

                # Linear regression closed-form gradient (Tikhonov regularization)
                grad = np.matmul(np.matmul(np.transpose(Xsplit[fileInd]), Xsplit[fileInd]), wt) - np.matmul(np.transpose(Xsplit[fileInd]), ysplit[fileInd]) + np.multiply(zeta, wt)
                # grad = np.matmul(np.matmul(np.transpose(Xsplit[fileInd]), Xsplit[fileInd]), wt) - np.matmul(np.transpose(Xsplit[fileInd]), ysplit[fileInd])
                if worker in A:
                    if shouldDistort(approach, files[fileInd], A, D):
                        # Distort gradient
                        grad = distortGradient(grad, err_mode, adversary_const)

                fileGrads[fileInd][files[fileInd].index(worker)] = copy.deepcopy(grad)

        # Majority vote
        # Key: File index (0-indexed), Value: majority vote gradient for the file of dimension a x n
        # where a depends on whether the split is exact or not but roughly it is (d+1)/f
        fileMajGrad = {}

        # Aspis+ detection
        if approach == ASPIS_PLUS:
            agreements, detectedWorkers, fileGrads =  _attempt_degree_detection(K, i, det_win_length, agreements, detectedWorkers, files, fileGrads, A, q, pair_groups, r)

        ###################################################################################################################
        # 1st level of aggregation
        if approach == BASELINE:
            # No aggregation
            for fileInd in range(f):
                # Only one gradient for each file
                assert len(fileGrads[fileInd]) == 1, "Error! Baseline scheme cannot have redundancy!"
                fileMajGrad[fileInd] = fileGrads[fileInd][0]

        elif approach == DETOX or approach == ASPIS or approach == ASPIS_PLUS:
            for fileInd in range(f):
                fileMajGrad[fileInd] = _grad_majority_vote(fileGrads[fileInd])

        else:
            assert 0 == 1, "Invalid approach!"

        # Assert gradient dimensions
        for fileInd in range(f):
            assert fileMajGrad[fileInd].shape[0] == d, "Invalid gradient dimensions!"
            assert fileMajGrad[fileInd].shape[1] == 1, "Invalid gradient dimensions!"
        ###################################################################################################################

        ###################################################################################################################
        # 2nd level of aggregation
        finalGrad = None
        if mode == COORD_MEDIAN:
            finalGrad = np.median(np.array([fileMajGrad[fileInd] for fileInd in range(f)]), axis = 0)
        elif mode == GEOMETRIC_MEDIAN:
            finalGrad = compute_geometric_median([fileMajGrad[fileInd] for fileInd in range(f)]).median
        else:
            assert 0 == 1, "Invalid mode!"

        # Assert gradient dimensions
        assert finalGrad.shape[0] == d, "Invalid gradient dimensions!"
        assert finalGrad.shape[1] == 1, "Invalid gradient dimensions!"
        ###################################################################################################################

        ###################################################################################################################
        # Update model
        wt = wt - alpha*finalGrad
        ###################################################################################################################

        # Compute loss and print stats
        if i % pointFreq == 0:
            loss += [(1/n * 0.5 * (np.matmul(np.transpose(y - np.matmul(X, wt)), y - np.matmul(X, wt)) + zeta*np.linalg.norm(wt))).item()]
            # loss += [(1/n * 0.5 * (np.matmul(np.transpose(y - np.matmul(X, wt)), y - np.matmul(X, wt)))).item()]

            iterIdx += [i]

            print("Iteration:", i, ", Loss:", loss[-1], ", |w|:", np.linalg.norm(wt))

        i += 1

        # Special handling for Aspis+
        if approach == ASPIS_PLUS:
            if permute_files:
                files, workerFiles = permuteFiles(K, files, workerFiles)

        # All approaches support random choice of Byzantines at windows of fixed length
        if randomByzantines and i%adv_win_length == 0:
            A = set(generateRandomByzantines(K, q))
            print("Iteration: {}, New set of Byzantines: {}".format(i, A))

        # Done
        if np.linalg.norm(finalGrad) < epsilon or i == max_iterations: break
        # if len(loss) > 1 and abs(loss[-1] - loss[-2]) < epsilon or i == MAX_ITERATIONS: break

    t1 = time.time()
    print("End! Total time: ", t1 - t0, "sec\n")

    return loss, iterIdx