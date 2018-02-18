"""
CS 156a: Final Exam
Anthony Bartolotta
Problems 13,14,15,16,17,18
"""

import numpy as np
import sklearn.svm as svm


def pseudoInverse(X):
    # Calculate pseudo-inverse
    tempM = np.linalg.inv(np.dot(np.transpose(X), X))
    xPseudo = np.dot(tempM, np.transpose(X))
    return xPseudo

def generateData(nSamples):
    rerun = True
    while rerun:
        # Generate points in [-1,1]x[-1,1]
        X_vec = np.random.uniform(-1,1,[nSamples,2])
        # Classify each point
        y_vec = np.sign(X_vec[:,1]-X_vec[:,0]+.25*np.sin(np.pi*X_vec[:,0]))
        # If all points have same classification, generate new points
        rerun = all( [ y_vec[i]==y_vec[0] for i in range(len(y_vec)) ] )
    return [X_vec, y_vec]

def sampleError(y_true, y_fit):
    # Find fraction of mismatch
    fMismatch = float(np.count_nonzero(y_true != y_fit)) / len(y_true)
    return fMismatch

def findNearestCenters(X_vec, centers):
    # Calculate distances of sample points from centers
    distances = np.array([[ np.linalg.norm(X_vec[i,:] - centers[j]) \
                    for j in range(len(centers))] for i in range(len(X_vec))])
    # Find closest center for each point
    closestCenter = np.argmin(distances, axis=1)
    return closestCenter

def initializeLloyds(X_vec, K):
    # Reinitialize algorithm until non-empty starting clusters are produced
    reinit = True
    while reinit:
        # Choose K points at random uniformly from the space as initial
        # centers
        centers = [np.random.uniform(-1,1,[1,2]) for j in range(K)]
        # Find the closest center for each sample point
        closestCenters = findNearestCenters(X_vec, centers)
        # Group sample points by their nearest center
        groups = [ X_vec[closestCenters==j, :] for j in range(K)]
        # Check that all groups are non-empty. If some are empty, repeat.
        if all([len(g)!=0 for g in groups]):
            reinit = False
    return [centers, groups]

def iterationLloyds(centers, groups):
    # Perform one iteration of Lloyd's algorithm
    # Define new centers
    newCenters = [np.average(g, axis=0) for g in groups]
    # Return all sample points to a single group
    X_vec = np.vstack(groups)
    # Find the closest center for each sample point
    closestCenters = findNearestCenters(X_vec, newCenters)
    # Group sample points by their nearest center
    newGroups = [ X_vec[closestCenters==j, :] for j in range(len(newCenters))]
    return [newCenters, newGroups]

def lloydsAlgorithm(X_vec, K):
    # Initialize boolean for if the iteration process should continue
    iterate = True
    # Initialize the algorithm
    [centers, groups] =  initializeLloyds(X_vec, K)
    oldCenters = centers
    # Iterate
    while iterate:
        # Perform one iteration of Lloyd's algorithm
        [centers, groups] = iterationLloyds(oldCenters, groups)
        # Check that groups are non empty
        if any([len(g)==0 for g in groups]):
            # If a cluster has become empty, the algorithm has failed and
            # needs to be reinitialized
            [centers, groups] =  initializeLloyds(X_vec, K)
        # Check if algorithm has converged
        if all([np.linalg.norm(centers[i] - oldCenters[i]) <= 10**(-10) \
                                            for i in range(K)]):
            # If algorithm has converged, terminate.
            iterate = False
        else:
            # If algorithm hasn't converged, continue iterating
            oldCenters = centers
    return centers

def trainRBF(X_vec, y_vec, gamma, K):
    # Use Lloyd's algorithm to perform clustering
    centers = lloydsAlgorithm(X_vec, K)
    # Use linear regression to find appropriate weights for radial
    # basis functions
    phi = np.array( [ [ \
                np.exp( -gamma*np.linalg.norm(X_vec[i,:]-centers[j])**2 ) \
                for j in range(len(centers)) ] for i in range(len(X_vec)) ] )
    # Augment matrix to account for constant bias term
    phi_aug = np.hstack([np.ones((len(X_vec),1)), phi])
    invPhi_aug = pseudoInverse(phi_aug)
    w_vec = np.dot(invPhi_aug, y_vec)
    return [w_vec, centers]

def classifyRBF(X_out, w_vec, centers, gamma):
    # Classify points
    phi = np.array( [ [ \
                np.exp( -gamma*np.linalg.norm(X_out[i,:]-centers[j])**2 ) \
                for j in range(len(centers))] for i in range(len(X_out))])
    phi_aug = np.hstack([np.ones((len(X_out),1)), phi])
    y_rbf = np.sign(np.dot(phi_aug, w_vec))
    return y_rbf

def evaluateRBF(X_train, y_train, X_test, y_test, gamma, K):
    # Fit data using RBF with K clusters
    [w_vec, centers] = trainRBF(X_train, y_train, gamma, K)
    # Classify in-sample points and find in-sample error
    y_rbf_in = classifyRBF(X_train, w_vec, centers, gamma)
    E_in = sampleError(y_train, y_rbf_in)
    # Classify out-of-sample points and find out-of-sample error
    y_rbf_out = classifyRBF(X_test, w_vec, centers, gamma)
    E_out = sampleError(y_test, y_rbf_out)
    return [E_in, E_out]

def evaluateSVM(X_train, y_train, X_test, y_test, g):
    # Train the SVM
    clf = svm.SVC(kernel='rbf', gamma=g, C=10**6)
    clf.fit(X_train, y_train)
    # Classify in-sample points and find in-sample error
    y_in = clf.predict(X_train)
    E_in = sampleError(y_train, y_in)
    # Classify out-of-sample points and find out-of-sample error
    y_out = clf.predict(X_test)
    E_out = sampleError(y_test, y_out)
    return [E_in, E_out]

def trial(n_in, n_out):
    # Generate training data
    [X_train, y_train] = generateData(n_in)
    # Generate testing data
    [X_test, y_test] = generateData(n_out)
    # Evaluate performance of hard-margin RBF-kernel SVM with gamma = 1.5
    [E_in_1, E_out_1] = evaluateSVM(X_train, y_train, X_test, y_test, 1.5)
    # Evaluate performance of regular RBF with K = 9, gamma = 1.5
    [E_in_2, E_out_2] = evaluateRBF(X_train, y_train, X_test, y_test, 1.5, 9)
    # Evaluate performance of regular RBF with K = 9, gamma = 2
    [E_in_3, E_out_3] = evaluateRBF(X_train, y_train, X_test, y_test, 2.0, 9)
    # Evaluate performance of regular RBF with K = 12, gamma = 1.5
    [E_in_4, E_out_4] = evaluateRBF(X_train, y_train, X_test, y_test, 1.5, 12)
    # Compile results
    trialResults = [E_in_1, E_out_1, E_in_2, E_out_2, E_in_3, E_out_3, \
                                                            E_in_4, E_out_4]
    return trialResults

def main():
    # Parameters for trials
    nTrials = 1000
    n_in = 10**2
    n_out = 10**3
    # Collect results
    trialResults = np.array( [trial(n_in, n_out) for j in range(nTrials)] )
    # Fraction of trials data can't be separated by hard-margin SVM
    E_in_svm = trialResults[:,0]
    badTrials = ( E_in_svm > 1.0/(2.0*n_in) )
    f_failed = float(sum(badTrials)) / nTrials
    print("Fraction of trials data was inseparable = "+repr(f_failed)+"\n")
    # Find out-of-sample errors for trials with separable data
    E_out_svm = trialResults[~badTrials, 1]
    E_out_9_15 = trialResults[~badTrials, 3]
    E_out_12_15 = trialResults[~badTrials, 7]
    # Fraction of trials kernel SVM beat K=9, gamma=1.5 RBF
    f_better_9 = float(sum(E_out_9_15 > E_out_svm)) / sum(~badTrials)
    print("Fraction of trials kernel SVM beat K=9, gamma=1.5 RBF = " + \
                                                      repr(f_better_9)+"\n")
    # Fraction of trials kernel SVM beat K=12, gamma=1.5 RBF
    f_better_12 = float(sum(E_out_12_15 > E_out_svm)) / sum(~badTrials)
    print("Fraction of trials kernel SVM beat K=12, gamma=1.5 RBF = " + \
                                                      repr(f_better_12)+"\n")
    # 
    E_in_9 = trialResults[:, 2]
    E_out_9 = trialResults[:, 3]
    E_in_12 = trialResults[:, 6]
    E_out_12 = trialResults[:, 7]
    delta_E_in = (E_in_12 - E_in_9)
    delta_E_out = (E_out_12 - E_out_9)
    f_dec_E_in = float(sum(delta_E_in < 0)) / nTrials
    f_dec_E_out = float(sum(delta_E_out < 0)) / nTrials
    print("When going from K=9 to K=12 RBF with gamma=1.5: ")
    print("Fraction of trials E_in decreased = " + repr(f_dec_E_in))
    print("Fraction of trials E_out decreased = " + repr(f_dec_E_out) + "\n")
    #
    E_in_20 = trialResults[:, 4]
    E_out_20 = trialResults[:, 5]
    delta_E_in = (E_in_20 - E_in_9)
    delta_E_out = (E_out_20 - E_out_9)
    f_dec_E_in = float(sum(delta_E_in < 0)) / nTrials
    f_dec_E_out = float(sum(delta_E_out < 0)) / nTrials
    print("When going from gamma=1.5 to gamma=2.0 RBF with K=9: ")
    print("Fraction of trials E_in decreased = " + repr(f_dec_E_in))
    print("Fraction of trials E_out decreased = " + repr(f_dec_E_out) + "\n")
    #
    goodTrials = ( E_in_9 < 1.0/(2.0*n_in) )
    f_good = float(sum(goodTrials)) / nTrials
    print("Fraction of trials with E_in=0 for K=9 gamma=1.5 RBF : " \
                                                              + repr(f_good))
    return

main()
