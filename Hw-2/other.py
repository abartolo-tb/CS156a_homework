"""
CS 156a: Homework #2
Anthony Bartolotta
Other
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

def individualTrial(nSamples, nOutSample):
    # Generate line for classification
    cVec = generateClassifyingLine()
    
    # Generate random data points and classify
    data = generateData(nSamples, cVec)
    [trueClassVec, inSampleVecs] = data
    
    # Use linear regression to approximate target function
    wVecLin = linearRegression(data)
    # Calculate linear regression in-sample error
    eInLin = inSampleError(data, wVecLin)
    # Calculate linear regression out-of-sample error
    eOutLin = outSampleError(cVec, wVecLin, nOutSample)
    
    # Use PLA to approximate target function
    [stepsPLA, wVecPLA] = perceptron(data)
    # Calculate PLA in-sample error
    eInPLA = inSampleError(data, wVecPLA)
    # Calculate PLA out-of-sample error
    eOutPLA = outSampleError(cVec, wVecPLA, nOutSample)

    # Use PLA w/ Lin. Reg. to approximate target function
    [stepsLinPLA, wVecLinPLA] = perceptron(data)
    # Calculate PLA w/ Lin. Reg. in-sample error
    eInLinPLA = inSampleError(data, wVecLinPLA)
    # Calculate PLA w/ Lin. Reg. out-of-sample error
    eOutLinPLA = outSampleError(cVec, wVecLinPLA, nOutSample)
    
    return [eInLin, eOutLin, stepsPLA, eInPLA, eOutPLA,
                                    stepsLinPLA, eInLinPLA, eOutLinPLA]

def main(nTrials, nSamples, nOutSample):
    eInLinArray = np.zeros([nTrials])
    eOutLinArray = np.zeros([nTrials])
    stepsPLAArray = np.zeros([nTrials])
    eInPLAArray = np.zeros([nTrials])
    eOutPLAArray = np.zeros([nTrials])
    stepsLinPLAArray = np.zeros([nTrials])
    eInLinPLAArray = np.zeros([nTrials])
    eOutLinPLAArray = np.zeros([nTrials])
    for j in range(nTrials):
        returnData = individualTrial(nSamples, nOutSample)
        eInLinArray[j] = returnData[0]
        eOutLinArray[j] = returnData[1]
        stepsPLAArray[j] = returnData[2]
        eInPLAArray[j] = returnData[3]
        eOutPLAArray[j] = returnData[4]
        stepsLinPLAArray[j] = returnData[5]
        eInLinPLAArray[j] = returnData[6]
        eOutLinPLAArray[j] = returnData[7]
    avg_eInLin = np.mean(eInLinArray)
    avg_eOutLin = np.mean(eOutLinArray)
    avg_stepsPLA = np.mean(stepsPLAArray)
    avg_eInPLA = np.mean(eInPLAArray)
    avg_eOutPLA = np.mean(eOutPLAArray)
    avg_stepsLinPLA = np.mean(stepsLinPLAArray)
    avg_eInLinPLA = np.mean(eInLinPLAArray)
    avg_eOutLinPLA = np.mean(eOutLinPLAArray)
    print("Number of Learning Samples = " + repr(nSamples))
    print("For Linear Regression, average E_in = " + repr(avg_eInLin))
    print("For Linear Regression, average E_out = " + repr(avg_eOutLin))
    print("For PLA, average steps = " + repr(avg_stepsPLA))
    print("For PLA, average E_in = " + repr(avg_eInPLA))
    print("For PLA, average E_out = " + repr(avg_eOutPLA))
    print("For Lin Reg w PLA, average steps = " + repr(avg_stepsLinPLA))
    print("For Lin Reg w PLA, average E_in = " + repr(avg_eInLinPLA))
    print("For Lin Reg w PLA, average E_out = " + repr(avg_eOutLinPLA))
    return

main(1000, 10, 10000)
main(1000, 100, 10000)
