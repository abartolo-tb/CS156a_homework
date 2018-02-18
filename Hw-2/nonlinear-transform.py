"""
CS 156a: Homework #2
Anthony Bartolotta
Problems 8,9,10
"""
import numpy as np

def pseudoInverse(X):
    tempM = np.linalg.inv(np.dot(np.transpose(X), X))
    xPseudo = np.dot(tempM, np.transpose(X))
    return xPseudo

def noisyClassification(points):
    # Set true classification vector
    cVec = np.array([-.6, 0, 0, 0, 1, 1])
    # First find true classification of points
    classifications = np.sign(np.dot(points, cVec))
    # Randomly select 10% of the points and flip their classification
    nPoints = len(classifications)
    nFlip = nPoints / 10
    randInts = np.random.randint(low=0, high=nPoints, size=[nFlip,1])
    classifications[randInts] = -classifications[randInts]
    return classifications

def generateData(nSamples):
    # Generate points in [-1,1]x[-1,1]
    X1 = np.random.uniform(-1,1,[nSamples,1])
    X2 = np.random.uniform(-1,1,[nSamples,1])
    # Calculate remaining components of transformed data
    ones = np.ones([nSamples,1])
    X1X2 = X1 * X2
    squareX1 = X1**2
    squareX2 = X2**2
    vecs = np.concatenate((ones, X1, X2, X1X2, squareX1, squareX2), axis=1)
    # Classify each point
    classifications = noisyClassification(vecs)
    data = [classifications, vecs]
    return data

def inSampleError(data, wVec):
    [trueClass, inSampleVecs] = data
    # Classify in sample points
    wClass = np.sign(np.dot(inSampleVecs, wVec))
    # Find fraction of mismatch
    fMismatch = float(sum(trueClass != wClass)) / len(trueClass)
    return fMismatch

def outSampleError(wVec, nOutSample):
    # Generate new out-of-sample points
    [classifications, vecs] = generateData(nOutSample)
    # Classify these points using the learned function
    learnClass = np.sign(np.dot(vecs, wVec))
    # Return fraction of mismatch
    fMismatch = float(sum(classifications != learnClass)) / nOutSample
    return fMismatch

def compareHypothesis(hVec, wVec, nTestPoints):
    [cVec, testVecs] = generateData(nTestPoints)
    hClassifications = np.sign(np.dot(testVecs, hVec))
    wClassifications = np.sign(np.dot(testVecs, wVec))
    fMismatch = float(sum(hClassifications != wClassifications))/ nTestPoints
    return fMismatch

def simpleLinearRegression(data):
    [trueClassVec, inSampleVecs] = data
    # Data is already transformed, strip off transformed components
    inSampleVecs = inSampleVecs[:, 0:3]
    # Calculate pseudo-inverse
    xPseudo = pseudoInverse(inSampleVecs)
    # Perform linear regression
    wVec = np.dot(xPseudo, trueClassVec)
    # Weight vector is shorter than the transformed data. It must
    # be padded with zeros to match the original length.
    wVec = np.concatenate((wVec, np.zeros([3])), axis=0)
    return wVec

def transformedLinearRegression(data):
    [trueClassVec, inSampleVecs] = data
    # Calculate pseudo-inverse
    xPseudo = pseudoInverse(inSampleVecs)
    # Perform linear regression
    wVec = np.dot(xPseudo, trueClassVec)
    return wVec

def singleTrial(nSamples, nOutSample):
    # Generate in-sample data points
    data = generateData(nSamples)
    [trueClassVec, inSampleVecs] = data
    # Use non-transformed linear regression to approximate target function
    wSimple = simpleLinearRegression(data)
    # Calculate in-sample error
    eInSimple = inSampleError(data, wSimple)
    # Use transformed linear regression to approximate target function
    wTrans = transformedLinearRegression(data)
    # Calculate out-of-sample error
    eOutTrans = outSampleError(wTrans, nOutSample)
    # Define five possible hypotheses
    gA = np.array([-1, -.05, .08, .13, 1.5, 1.5])
    gB = np.array([-1, -.05, .08, .13, 1.5, 15])
    gC = np.array([-1, -.05, .08, .13, 15, 1.5])
    gD = np.array([-1, -1.5, .08, .13, .05, .05])
    gE = np.array([-1, -.05, .08, 1.5, .15, .15])
    # Compare hypotheses to transformed linear regression weight vector
    nTestPoints = 10**4
    fA = compareHypothesis(gA, wTrans, nTestPoints)
    fB = compareHypothesis(gB, wTrans, nTestPoints)
    fC = compareHypothesis(gC, wTrans, nTestPoints)
    fD = compareHypothesis(gD, wTrans, nTestPoints)
    fE = compareHypothesis(gE, wTrans, nTestPoints)
    return [eInSimple, eOutTrans, fA, fB, fC, fD, fE]

def main(nTrials, nSamples, nOutSample):
    # Initialize arrays
    eInSimpArray = np.zeros([nTrials])
    eOutTransArray = np.zeros([nTrials])
    fAArray = np.zeros([nTrials])
    fBArray = np.zeros([nTrials])
    fCArray = np.zeros([nTrials])
    fDArray = np.zeros([nTrials])
    fEArray = np.zeros([nTrials])
    # Run trials
    for j in range(nTrials):
        [eInSimple, eOutTrans, fA, fB, fC, fD, fE] = \
                                            singleTrial(nSamples, nOutSample)
        eInSimpArray[j] = eInSimple
        eOutTransArray[j] = eOutTrans
        fAArray[j] = fA
        fBArray[j] = fB
        fCArray[j] = fC
        fDArray[j] = fD
        fEArray[j] = fE
    # Return averages
    avg_E_In_Simple = np.mean(eInSimpArray)
    avg_E_Out_Trans = np.mean(eOutTransArray)
    avg_f_A = np.mean(fAArray)
    avg_f_B = np.mean(fBArray)
    avg_f_C = np.mean(fCArray)
    avg_f_D = np.mean(fDArray)
    avg_f_E = np.mean(fEArray)
    print("Average In-Sample Error for simple linear regression = " + \
                                                  repr(avg_E_In_Simple))
    print("Average Out-of-Sample Error for transformed linear regression = " \
                                                  + repr(avg_E_Out_Trans))
    print("Average mismatch probability for hypothesis A = " + repr(avg_f_A))
    print("Average mismatch probability for hypothesis B = " + repr(avg_f_B))
    print("Average mismatch probability for hypothesis C = " + repr(avg_f_C))
    print("Average mismatch probability for hypothesis D = " + repr(avg_f_D))
    print("Average mismatch probability for hypothesis E = " + repr(avg_f_E))
    return

main(1000, 1000, 1000)
