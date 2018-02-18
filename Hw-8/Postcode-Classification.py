"""
CS 156a: Homework #8
Anthony Bartolotta
Problems 2,3,4,5,6,7,8,9,10
"""
import numpy as np
import sklearn.svm as svm
import scipy.stats as stats

def readInData(filepath):
    F = open(filepath,"r")
    rawLines = F.readlines()
    X_store = []
    y_store = []
    for line in rawLines:
        entries = line.split()
        X_store.append(map(float, entries[1:]))
        y_store.append(int(float(entries[0])))
    X_vec = np.array(X_store)
    y_vec = np.array(y_store)
    return [X_vec, y_vec]

def processData_NversusALL(n, y_vec):
    # Find digits with value "n"
    ind = (y_vec == n)
    y_class = np.zeros((len(y_vec),))
    y_class[ind] = 1
    y_class[~ind] = -1
    return y_class

def processData_NversusM(n, m, X_vec, y_vec):
    # Make local copies of data
    X_temp = X_vec.copy()
    y_temp = y_vec.copy()
    # Find digits with values "n" and "m"
    n_ind = (y_temp == n)
    m_ind = (y_temp == m)
    # Process and trim data
    y_temp[n_ind] = 1
    y_temp[m_ind] = -1
    y_trimmed = y_temp[n_ind + m_ind]
    x_trimmed = X_temp[n_ind + m_ind, :]
    return [x_trimmed, y_trimmed]

def sampleError(X_vec, y_vec, clf):
    # Classify sample points
    c_vec = clf.predict(X_vec)
    # Find fraction of mismatch
    fMismatch = float(sum(c_vec != y_vec)) / len(c_vec)
    return fMismatch

def sub_problem_2_3(X_vec, y_vec, digits, c, Q, Highest_Bool):
    # Define polynomial SVM
    clf = svm.SVC(kernel='poly', degree=Q, gamma=1, coef0=1, C=c)
    l = len(digits)
    results = np.zeros((l,3))
    # Loop over list of provided digits
    for j in range(l):
        digit = digits[j]
        y_proc = processData_NversusALL(digit, y_vec)
        clf.fit(X_vec, y_proc)
        # Store the digit, number of support vectors, and in-sample error
        n_support = sum(clf.n_support_)
        E_in = sampleError(X_vec, y_proc, clf)
        results[j,:] = [digit, E_in, n_support]
    # If Highest_Bool=True, find digit with highest in-sample error
    if Highest_Bool:
        ind = np.argmax(results[:,1])
    # If Highest_Bool=False, find digit with lowest in-sample error
    else:
        ind = np.argmin(results[:,1])
    return results[ind,:]

def problem_2_3_4(X_vec, y_vec):
    # Set values for c and Q
    c = .01
    Q = 2
    # Separate digits into even and odd
    digits_odd = [1,3,5,7,9]
    digits_even = [0,2,4,6,8]
    # Find results from problems 2 and 3
    prob2_results = sub_problem_2_3(X_vec, y_vec, digits_even, c, Q, True)
    prob3_results = sub_problem_2_3(X_vec, y_vec, digits_odd, c, Q, False)
    # Print results for problem 2
    print("Problem 2:")
    [digit_2, E_in_2, n_support_2] = prob2_results
    print("Digit with highest E_in was: " + repr(digit_2))
    print("E_in = " + repr(E_in_2))
    print("Number of support vectors = " + repr(n_support_2) + "\n")
    # Print results for problem 3
    print("Problem 3:")
    [digit_3, E_in_3, n_support_3] = prob3_results
    print("Digit with lowest E_in was: " + repr(digit_3))
    print("E_in = " + repr(E_in_3))
    print("Number of support vectors = " + repr(n_support_3) + "\n")
    # Print results for problem 4
    print("Problem 4:")
    n_diff = abs(n_support_2 - n_support_3)
    print("Difference in number of support vectors = " + repr(n_diff) + "\n")
    return

def problem_5_6(X_train, y_train, X_test, y_test):
    # Consider 1 vs 5 classifier
    digit_1 = 1
    digit_2 = 5
    [X_train_proc, y_train_proc] = processData_NversusM(digit_1, digit_2, \
                                                            X_train, y_train)
    [X_test_proc, y_test_proc] = processData_NversusM(digit_1, digit_2, \
                                                            X_test, y_test)
    # Values of C and Q to be tested
    Q_v = [2, 5]
    C_v = [.001, .01, .1, 1]
    # Create list of parameter pairs to iterate through
    Q_list, C_list = map( lambda x : x.flatten(), np.meshgrid(Q_v, C_v) )
    n = len(Q_list)
    results = np.zeros((n, 5))
    for ind in range(n):
        # Read off values of c and Q for this trial
        Q = Q_list[ind]
        c = C_list[ind]
        # Train the SVM
        clf = svm.SVC(kernel='poly', degree=Q, gamma=1, coef0=1, C=c)
        clf.fit(X_train_proc, y_train_proc)
        # Calculate and store E_in, E_out, N_support
        E_in = sampleError(X_train_proc, y_train_proc, clf)
        E_out = sampleError(X_test_proc, y_test_proc, clf)
        n_support = sum(clf.n_support_)
        results[ind,:] = [c, Q, E_in, E_out, n_support]
    print("Problems 5 and 6:")
    print("Columns arranged as C, Q, E_in, E_out, N_SV: ")
    print(repr(results) + "\n")
    return

def problem_7_8(X_train, y_train):
    # Consider 1 vs 5 classifier
    digit_1 = 1
    digit_2 = 5
    [X_train_proc, y_train_proc] = processData_NversusM(digit_1, digit_2, \
                                                            X_train, y_train)
    # Values of C to be considered
    C_v = [.0001, .001, .01, .1, .1]
    # Vector of SVMs to be considered
    clf_v = [svm.SVC(kernel='poly', degree=2, gamma=1, coef0=1, C=c) \
                                                                 for c in C_v]
    # K-fold cross validation to be performed
    K = 10
    # Number of cross-validation trials to be performed
    N_trials = 100
    multitrial_results = np.zeros((N_trials, 2))
    for j_trial in range(N_trials):
        # Randomly permute data
        perm_ind = np.random.permutation(len(y_train_proc))
        X_permed = X_train_proc[perm_ind]
        y_permed = y_train_proc[perm_ind]
        # Partition data into K sets of equal or near equal size
        X_split = np.array(np.array_split(X_permed, K))
        y_split = np.array(np.array_split(y_permed, K))
        # Define results of each cross-validation step for each SVM
        cv_results = np.zeros((K, len(C_v)))
        for j_cv in range(K):
            # Define training and validation sets
            X_val = X_split[j_cv]
            y_val = y_split[j_cv]
            X_t = np.concatenate(X_split[np.arange(K) != j_cv])
            y_t = np.concatenate(y_split[np.arange(K) != j_cv])
            # Train all SVMs
            map(lambda clf : clf.fit(X_t, y_t), clf_v)
            # Find validation error for each SVM and store
            E_val_v = [sampleError(X_val, y_val, clf) for clf in clf_v]
            cv_results[j_cv, :] = E_val_v
        # Average cross-validation resutls for each validation set
        cv_ave = np.mean(cv_results, axis=0)
        # Find best performing SVM and save its CV error and c parameter
        ind_min = np.argmin(cv_ave)
        multitrial_results[j_trial, :] = [C_v[ind_min], cv_ave[ind_min]]
    # Interpret results
    c_best = stats.mode(multitrial_results[:,0])[0][0]
    trial_ind = (multitrial_results[:,0] == c_best)
    ave_cv_error = np.mean(multitrial_results[trial_ind,1])
    print("Problems 7 and 8:")
    print("In cross-validation, c = " + repr(c_best) + \
                                              " was chosen most frequently")
    print("Average cross-validation error = " + repr(ave_cv_error) + "\n")
    return
    
def problem_9_10(X_train, y_train, X_test, y_test):
    # Consider 1 vs 5 classifier
    digit_1 = 1
    digit_2 = 5
    [X_train_proc, y_train_proc] = processData_NversusM(digit_1, digit_2, \
                                                            X_train, y_train)
    [X_test_proc, y_test_proc] = processData_NversusM(digit_1, digit_2, \
                                                            X_test, y_test)
    # Values of C to be tested
    C_v = [10**(-2), 10**(0), 10**(2), 10**(4), 10**(6)]
    results = np.zeros((len(C_v), 3))
    for ind in range(len(C_v)):
        # Read off value of c for this trial
        c = C_v[ind]
        # Train the SVM
        clf = svm.SVC(kernel='rbf', gamma=1, C=c)
        clf.fit(X_train_proc, y_train_proc)
        # Calculate and store E_in, E_out
        E_in = sampleError(X_train_proc, y_train_proc, clf)
        E_out = sampleError(X_test_proc, y_test_proc, clf)
        results[ind,:] = [c, E_in, E_out]
    print("Problems 9 and 10:")
    print("Columns arranged as C, E_in, E_out: ")
    print(repr(results) + "\n")
    return


def main(training_data_path, test_data_path):
    [X_train, y_train] = readInData(training_data_path)
    [X_test, y_test] = readInData(test_data_path)
    problem_2_3_4(X_train, y_train)
    problem_5_6(X_train, y_train, X_test, y_test)
    problem_7_8(X_train, y_train)
    problem_9_10(X_train, y_train, X_test, y_test)
    return

# The training and test data files originally used with this script
# have not been included in this GitHub repository. The original
# dataset consisted of three tab delimited columns derived from the
# MNIST dataset. The first column was the digit's classification 
# (0 throught 9, stored as float). The second and third columns were
# high level features extracted from each sample (floats).
training_data_path = "features.train"
test_data_path = "features.test"
main(training_data_path, test_data_path)
