"""
CS 156a: Final Exam
Anthony Bartolotta
Problems 7,8,9,10
"""
import numpy as np


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

def transformInputSpace(X_vec, NonlinearBool):
    # Perform transform of raw data
    if NonlinearBool:
    # If NonlinearBool==True, perform a nonlinear transform of the input
    # space
        c1 = np.ones((len(X_vec[:,0]),1))
        c4 = X_vec[:,0]*X_vec[:,1]
        c5 = X_vec[:,0]**2
        c6 = X_vec[:,1]**2
        tVec = np.column_stack([c1, X_vec, c4, c5, c6])
    else:
    # If NonlinearBool==False, just add the constant dimension to the
    # input space
        c1 = np.ones((len(X_vec[:,0]),1))
        tVec = np.column_stack([c1, X_vec])
    return tVec

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

def pseudoInverse(X, lamb):
    xPrime = np.dot(np.transpose(X), X)
    L = lamb * np.identity(xPrime.shape[0])
    xPseudo = np.dot(np.linalg.inv(xPrime + L), np.transpose(X))
    return xPseudo

def regulatedLinearRegression(X_vec, y_vec, lamb):
    # Calculate pseudo-inverse including regulator
    xPseudo = pseudoInverse(X_vec, lamb)
    # Perform linear regression using pseudo-inverse
    wVec = np.dot(xPseudo, y_vec)
    return wVec

def sampleError(X_vec, y_vec, wVec):
    # Classify sample points
    wClass = np.sign(np.dot(X_vec, wVec))
    # Find fraction of mismatch
    fMismatch = float(sum(y_vec != wClass)) / len(y_vec)
    return fMismatch

def problems_7_8_9(inSamplePath, outSamplePath):
    # Load in-sample and out-of-sample data from files
    [X_in_raw, y_in] = readInData(inSamplePath)
    [X_out_raw, y_out] = readInData(outSamplePath)
    # Prepare data in both standard and nonlinearly transformed ways
    X_in_lin_pre = transformInputSpace(X_in_raw, False)
    X_out_lin_pre = transformInputSpace(X_out_raw, False)
    X_in_nonlin_pre = transformInputSpace(X_in_raw, True)
    X_out_nonlin_pre = transformInputSpace(X_out_raw, True)
    # Prepare data for n vs. all digit classification for all possible n
    in_data = [ [X_in_lin_pre, X_in_nonlin_pre, \
                         processData_NversusALL(n, y_in)] for n in range(10)]
    out_data = [ [X_out_lin_pre, X_out_nonlin_pre, \
                         processData_NversusALL(n, y_out)] for n in range(10)]
    # For each possible classification task, evaluate performance of original
    # and nonlinearly transformed data both in-sample and out-of-sample. Use
    # linear regression with regularization parameter Lambda = 1.
    results = []
    for n in range(10):
        # Load in-sample data and perform linear regression
        [X_in_l, X_in_nl, y_in] = in_data[n]
        w_l = regulatedLinearRegression(X_in_l, y_in, 1)
        w_nl = regulatedLinearRegression(X_in_nl, y_in, 1)
        # Load out-of-sample data. Score E_in and E_out for original and
        # nonlinearly transformed data
        [X_out_l, X_out_nl, y_out] = out_data[n]
        E_in_l = sampleError(X_in_l, y_in, w_l)
        E_out_l = sampleError(X_out_l, y_out, w_l)
        E_in_nl = sampleError(X_in_nl, y_in, w_nl)
        E_out_nl = sampleError(X_out_nl, y_out, w_nl)
        # Store results as [digit, Bool_transform, E_in, E_out]
        results.append([n, False, E_in_l, E_out_l])
        results.append([n, True, E_in_nl, E_out_nl])
    return results

def problem_10(inSamplePath, outSamplePath):
    # Load in-sample and out-of-sample data from files
    [X_in_raw, y_in_raw] = readInData(inSamplePath)
    [X_out_raw, y_out_raw] = readInData(outSamplePath)
    # Prepare nonlinearly transformed data
    X_in_pre = transformInputSpace(X_in_raw, True)
    X_out_pre = transformInputSpace(X_out_raw, True)
    # Consider 1 vs 5 classification task
    [X_in, y_in] = processData_NversusM(1, 5, X_in_pre, y_in_raw)
    [X_out, y_out] = processData_NversusM(1, 5, X_out_pre, y_out_raw)
    # Compare performance with Lambda = .01 and Lambda = 1.
    lambda_values = [.01, 1]
    results = []
    for lamb in lambda_values:
        # Perform linear regression with regularization
        w_vec = regulatedLinearRegression(X_in, y_in, lamb)
        # Find E_in and E_out
        E_in = sampleError(X_in, y_in, w_vec)
        E_out = sampleError(X_out, y_out, w_vec)
        # Store results as [lambda, E_in, E_out]
        results.append([lamb, E_in, E_out])
    return results

def main(inSamplePath, outSamplePath):
    r_7_8_9 = problems_7_8_9(inSamplePath, outSamplePath)
    print("Problems 7, 8, and 9: ")
    print("n vs. all classification with and without a nonlinear transform.")
    print("Columns arranged as n, (transformed?), E_in, E_out :")
    print(repr(np.array(r_7_8_9)) + "\n")
    r_10 = problem_10(inSamplePath, outSamplePath)
    print("Problem 10: ")
    print("1 vs 5 classification with nonlinear transform.")
    print("Columns arranged as lambda, E_in, E_out :")
    print(repr(np.array(r_10)) + "\n")
    return

# The training and test data files originally used with this script
# have not been included in this GitHub repository. The original
# dataset consisted of three tab delimited columns derived from the
# MNIST dataset. The first column was the digit's classification 
# (0 throught 9, stored as float). The second and third columns were
# high level features extracted from each sample (floats).
inSamplePath = "features.train"
outSamplePath = "features.test"
main(inSamplePath, outSamplePath)
