"""
CS 156a: Homework #7
Anthony Bartolotta
Problems 8,9,10
"""
import numpy as np
import quadprog as qp

def generateClassifyingLine():
    # Generate two random points in [-1,1]x[-1,1]
    [p1,p2] = np.random.uniform(-1,1,[2,2])
    # Find normal vector to line passing between points
    # This defines the positive direction.
    vec12 = p1 - p2
    rotation = np.array([[0,1],[-1,0]])
    normalVec = np.dot(rotation, vec12)
    nVec = normalVec / np.linalg.norm(normalVec)
    # Define 1+2 dimensional classification vector
    cVec = np.concatenate(([-np.dot(nVec, p1)], nVec))
    return cVec

def generateData(nSamples, cVec):
    rerun = True
    while rerun:
        # Generate points in [-1,1]x[-1,1]
        points = np.random.uniform(-1,1,[nSamples,2])
        # Zeroth component is always 1
        ones = np.ones([nSamples,1])
        vecs = np.concatenate((ones, points),1)
        # Classify each point
        yVec = np.sign(np.dot(vecs, cVec))
        # If all points have same classification, generate new points
        rerun = all( [ yVec[i]==yVec[0] for i in range(len(yVec)) ] )
    data = [yVec, vecs]
    return data

def sampleError(data, wVec):
    [trueClass, sampleVecs] = data
    # Classify sample points
    wClass = np.sign(np.dot(sampleVecs, wVec))
    # Find fraction of mismatch
    fMismatch = float(sum(trueClass != wClass)) / len(trueClass)
    return fMismatch

def perceptronLearning(data, wVec):
    [trueClassVec, sampleVecs] = data
    # Classify all data points
    percClassVec = np.sign(np.dot(sampleVecs, wVec))
    # See if any points are misclassified
    misclassVec = (percClassVec != trueClassVec)
    misclassified = any(misclassVec)
    # Randomly select a misclassified point if it exists
    if misclassified:
        misclassPoints = sampleVecs[misclassVec]
        misclassClasses = trueClassVec[misclassVec]
        rInt = np.random.randint(0, len(misclassPoints))
        rPoint = misclassPoints[rInt]
        rClass = misclassClasses[rInt]
        # Update perceptron vector
        wVec = wVec + rClass*rPoint
    return [misclassified, wVec]

def perceptron(data):
    # Initialize weight vector
    wVec = np.array([0,0,0])
    # Begin Learning iterations
    misclassified = True
    while misclassified:
        [misclassified, wVec] = perceptronLearning(data, wVec)
    return wVec

def SVM(data):
    # Unpack data
    [yVec, xVecs] = data
    n = len(yVec)
    # Define quadratic cost matrix
    M = np.identity(3)
    M[0,0] = 10**(-10)
    # Define linear cost vector
    q = np.zeros((3,))
    # Inequality constraints
    G = np.array([yVec[i]*xVecs[i,:] for i in range(n)]).transpose()
    h = np.ones((n,))
    neq = 0
    # Solve quadratic program
    wVec = qp.solve_qp(M, q, G, h, neq)[0]
    return wVec

def numSupportVectors(data, wVec):
    # Set a rounding error for the margin
    roundingError = 10**(-5)
    # Unpack data
    [yVec, xVecs] = data
    n = len(yVec)
    # Inequality constraint
    G = np.array([yVec[i]*xVecs[i,:] for i in range(n)])
    # Find support vectors
    isSupport = (np.dot(G,wVec) <= 1 + roundingError)
    nSupport = sum(isSupport)
    return nSupport

def singleTrial(nSamples, nOutSample):
    # Generate line for classification
    cVec = generateClassifyingLine()
    # Generate in-sample and out-of-sample data points
    inData = generateData(nSamples, cVec)
    outData = generateData(nOutSample, cVec)
    # The PLA to find the target function. Find out-of-sample error
    plaVec = perceptron(inData)
    eOutPLA = sampleError(outData, plaVec)
    # Use SVM to find the target function. Find out-of-sample error and
    # number of support vectors.
    svmVec = SVM(inData)
    nSupport = numSupportVectors(inData, svmVec)
    eOutSVM = sampleError(outData, svmVec)
    # Return results from trial
    return [eOutPLA, eOutSVM, nSupport]

def trailsAverage(nTrials, nSamples, nOutSample):
    results = np.zeros((nTrials, 3))
    for j in range(nTrials):
        results[j,:] = singleTrial(nSamples, nOutSample)
    averagedResults = np.mean(results, axis=0)
    avg_E_pla, avg_E_svm, n_SV = averagedResults
    f_SVM_better = float(sum(results[:,1] <= results[:,0])) / nTrials
    print("Average PLA Out-of-Sample Error = " + repr(avg_E_pla))
    print("Average SVM Out-of-Sample Error = " + repr(avg_E_svm))
    print("Fraction of trials SVM beat PLA = " + repr(f_SVM_better))
    print("Average number of support vectors = " + repr(n_SV))
    return

def main():
    nTrials = 1000
    nOutSample = 10**4
    print("Number of data points = 10")
    nSamples = 10
    trailsAverage(nTrials, nSamples, nOutSample)
    print("Number of data points = 100")
    nSamples = 100    
    trailsAverage(nTrials, nSamples, nOutSample)
    return

main()
