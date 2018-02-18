"""
CS 156a: Homework #5
Anthony Bartolotta
Problems 8,9
"""
import numpy as np

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
    cVec = np.concatenate(([-np.vdot(normalVec, p1)], nVec))
    return cVec

def generateData(nSamples, cVec):
    # Generate points in [-1,1]x[-1,1]
    points = np.random.uniform(-1,1,[nSamples,2])
    # Zeroth component is always 1
    ones = np.ones([nSamples,1])
    vecs = np.concatenate((ones, points),1)
    # Classify each point
    classifications = np.sign(np.dot(vecs, cVec))
    data = [classifications, vecs]
    return data

def outSampleError(cVec, wVec, nOutSample):
    # Generate new out-of-sample points
    [trueClass, testVecs] = generateData(nOutSample, cVec)
    # Return the out-of-sample error
    Eout = np.sum(np.log(1+np.exp(-trueClass*np.dot(testVecs, wVec)))) \
                    / len(trueClass)
    return Eout

def CalcGrad(wVec, y, xVec):
    # Calculate gradient of the cross entropy error for fixed sample point
    numer = np.exp(-y*np.dot(xVec, wVec))*(-y*xVec)
    denom = 1 + np.exp(-y*np.dot(xVec, wVec))
    grad = numer / denom
    return grad

def logisticRegressionEpoch(data, wVec, nu):
    [trueClassVec, sampleVecs] = data
    # Find permutation of sample points for SGD
    perm = np.random.permutation(len(trueClassVec))
    # Perform SGD for entire epoch
    for ind in perm:
        # Take randomly selected point
        y = trueClassVec[ind]
        xVec = sampleVecs[ind,:]
        # Calculate gradient and update
        grad = CalcGrad(wVec, y, xVec)
        wVec = wVec - nu * grad
    return wVec

def logisticRegression(data, nu):
    # Initialize and set weight vector to all zeros
    nSteps = 0
    cont = True
    wVecOld = np.zeros([3])
    wVecNew = wVecOld
    while cont:
        # Continue SGD epochs until change in weight vector becomes small
        wVecNew = logisticRegressionEpoch(data, wVecOld, nu)
        nSteps += 1
        dW = np.linalg.norm(wVecNew - wVecOld)
        if dW < .01:
            cont = False
        else:
            wVecOld = wVecNew
    return [nSteps, wVecNew]

def singleTrial(nSamples, nOutSample, nu):
    cVec = generateClassifyingLine()
    data = generateData(nSamples, cVec)
    [nSteps, wVec] = logisticRegression(data, nu)
    Eout = outSampleError(cVec, wVec, nOutSample)
    return [nSteps, Eout]

def problem_8_and_9(nTrials, nSamples, nOutSample, nu):
    nStepsArray = np.zeros([nTrials])
    eOutArray = np.zeros([nTrials])
    for j in range(nTrials):
        [nSteps, Eout] = singleTrial(nSamples, nOutSample, nu)
        nStepsArray[j] = nSteps
        eOutArray[j] = Eout
    avg_Steps = np.mean(nStepsArray)
    avg_E_Out = np.mean(eOutArray)
    print("Average Number of Steps = " + repr(avg_Steps))
    print("Average Out-of-Sample Error = " + repr(avg_E_Out))
    return

problem_8_and_9(100, 100, 10**5, .01)
