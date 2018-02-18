"""
CS 156a: Homework #2
Anthony Bartolotta
Problems 5,6,7
"""
import numpy as np

def pseudoInverse(X):
    tempM = np.linalg.inv(np.dot(np.transpose(X), X))
    xPseudo = np.dot(tempM, np.transpose(X))
    return xPseudo

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

def inSampleError(data, wVec):
    [trueClass, inSampleVecs] = data
    # Classify in sample points
    wClass = np.sign(np.dot(inSampleVecs, wVec))
    # Find fraction of mismatch
    fMismatch = float(sum(trueClass != wClass)) / len(trueClass)
    return fMismatch

def outSampleError(cVec, wVec, nOutSample):
    # Generate new out-of-sample points
    testPoints = np.random.uniform(-1,1,[nOutSample,2])
    ones = np.ones([nOutSample,1])
    testVecs = np.concatenate((ones, testPoints),1)
    # Classify these points using the target function
    trueClass = np.sign(np.dot(testVecs, cVec))
    # Classify these points using the learned function
    learnClass = np.sign(np.dot(testVecs, wVec))
    # Return fraction of mismatch
    fMismatch = float(sum(trueClass != learnClass)) / nOutSample
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

def linearRegression(data):
    [trueClassVec, inSampleVecs] = data
    # Calculate pseudo-inverse
    xPseudo = pseudoInverse(inSampleVecs)
    # Perform linear regression
    wVec = np.dot(xPseudo, trueClassVec)
    return wVec

def perceptron(data):
    # Initialize weight vector
    wVec = np.array([0,0,0])
    # Begin Learning iterations
    misclassified = True
    steps = 0
    while misclassified:
        [misclassified, wVec] = perceptronLearning(data, wVec)
        steps +=1
    return [steps, wVec]

def perceptronWithLinearStart(data):
    # Use linear regression to initialize the weight vector
    wVec = linearRegression(data)
    # Begin Learning iterations
    misclassified = True
    steps = 0
    while misclassified:
        [misclassified, wVec] = perceptronLearning(data, wVec)
        steps +=1
    return [steps, wVec]

def linRegTrial(nSamples, nOutSample):
    # Generate line for classification
    cVec = generateClassifyingLine()
    # Generate random data points and classify
    data = generateData(nSamples, cVec)
    [trueClassVec, inSampleVecs] = data
    # Use linear regression to approximate target function
    wVec = linearRegression(data)
    # Calculate in-sample error
    eInSample = inSampleError(data, wVec)
    # Calculate out-of-sample error
    eOutSample = outSampleError(cVec, wVec, nOutSample)
    return [eInSample, eOutSample]

def linRegWithPercTrial(nSamples):
    # Generate line for classification
    cVec = generateClassifyingLine()
    # Generate random data points and classify
    data = generateData(nSamples, cVec)
    # Use linear regression to approximate target function
    [steps, wVec] = perceptronWithLinearStart(data)
    return steps

def problem_5_and_6(nTrials, nSamples, nOutSample):
    eInArray = np.zeros([nTrials])
    eOutArray = np.zeros([nTrials])
    for j in range(nTrials):
        [eInSample, eOutSample] = linRegTrial(nSamples, nOutSample)
        eInArray[j] = eInSample
        eOutArray[j] = eOutSample
    avg_E_In = np.mean(eInArray)
    avg_E_Out = np.mean(eOutArray)
    print("Average In-Sample Error = " + repr(avg_E_In))
    print("Average Out-of-Sample Error = " + repr(avg_E_Out))
    return

def problem_7(nTrials, nSamples):
    stepArray = np.zeros([nTrials])
    for j in range(nTrials):
        nSteps = linRegWithPercTrial(nSamples)
        stepArray[j] = nSteps
    avg_Steps = np.mean(stepArray)
    print("Average Steps = " + repr(avg_Steps))
    return

problem_5_and_6(1000, 100, 1000)
problem_7(1000, 10)
