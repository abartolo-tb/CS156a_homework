"""
CS 156a: Homework #7
Anthony Bartolotta
Problems 1,2,3,4,5
"""
import numpy as np

def pseudoInverse(X):
    xPrime = np.dot(np.transpose(X), X)
    xPseudo = np.dot(np.linalg.inv(xPrime), np.transpose(X))
    return xPseudo

def readInData(filepath):
    F = open(filepath,"r")
    rawLines = F.readlines()
    xVec = []
    yVec = []
    cVec = []
    for line in rawLines:
        entries = line.split()
        xVec.append(float(entries[0]))
        yVec.append(float(entries[1]))
        cVec.append(float(entries[2]))
    rawData = [xVec, yVec, cVec]
    return rawData

def nonlinearTransform(rawData):
    # Perform nonlinear transform of raw data
    [xVec, yVec, classVec] = np.array(rawData)
    c1 = np.ones(np.shape(xVec))
    c2 = xVec
    c3 = yVec
    c4 = xVec**2
    c5 = yVec**2
    c6 = xVec*yVec
    c7 = np.abs(xVec - yVec)
    c8 = np.abs(xVec + yVec)
    tVec = np.transpose(np.array([c1, c2, c3, c4, c5, c6, c7, c8]))
    tData = [classVec, tVec]
    return tData

def sampleError(data, wVec):
    [trueClass, sampleVecs] = data
    # Classify sample points
    wClass = np.sign(np.dot(sampleVecs, wVec))
    # Find fraction of mismatch
    fMismatch = float(sum(trueClass != wClass)) / len(trueClass)
    return fMismatch

def limitedLinearRegression(data, k):
    assert (k<=8)
    # Import data and truncate transformed data vectors after kth entry
    [trueClassVec, sampleVecs] = data
    sampleVecs = sampleVecs[:, :k+1]
    # Calculate pseudo-inverse
    xPseudo = pseudoInverse(sampleVecs)
    # Perform linear regression using pseudo-inverse
    wVec = np.dot(xPseudo, trueClassVec)
    # Zero-pad weight vector so it can be used on full nonlinear space
    padLen = 8 - len(wVec)
    wVec = np.pad(wVec, (0,padLen), 'constant', constant_values=0)
    return wVec

def singleTrial(tData, vData, oData, k):
    # Use linear regression to approximate target function
    wVec = limitedLinearRegression(tData, k)
    # Calculate in-sample, validation, and out-of-sample error
    eTraining = sampleError(tData, wVec)
    eValidation = sampleError(vData, wVec)
    eOut = sampleError(oData, wVec)
    return [eTraining, eValidation, eOut]

def main(inSamplePath, outSamplePath):
    # Set size of initial training data set
    tSize = 25
    # Load in-sample and out-of-sample data from files
    inSampleRaw = readInData(inSamplePath)
    outSampleRaw = readInData(outSamplePath)
    # Separate training and validation data
    trainingRaw = [inSampleRaw[j][:tSize] for j in range(len(inSampleRaw))]
    validationRaw = [inSampleRaw[j][tSize:] for j in range(len(inSampleRaw))]
    # Perform non-linear transformation of data
    tData = nonlinearTransform(trainingRaw)
    vData = nonlinearTransform(validationRaw)
    oData = nonlinearTransform(outSampleRaw)
    # First consider case where training set is larger than validation
    print("For Training = 25, Validation = 10: \n")
    for k in [2, 3, 4, 5, 6, 7]:
            [eT, eV, eOut] = singleTrial(tData, vData, oData, k)
            print("Nonlinear transformation k = " + repr(k) + " :")
            print("Training Error = " + repr(eT))
            print("Validation Error = " + repr(eV))
            print("Out-of-Sample Error = " + repr(eOut) + "\n")
    # Switch training and validation sets
    tData, vData = vData, tData
    print("For Training = 10, Validation = 25: \n")
    for k in [2, 3, 4, 5, 6, 7]:
            [eT, eV, eOut] = singleTrial(tData, vData, oData, k)
            print("Nonlinear transformation k = " + repr(k) + " :")
            print("Training Error = " + repr(eT))
            print("Validation Error = " + repr(eV))
            print("Out-of-Sample Error = " + repr(eOut) + "\n")
    return

# The training and test data files originally used with this script
# have not been included in this GitHub repository. The original
# dataset consisted of three tab delimited columns. These columns
# were the x coordinate (float), y coordinate (float), and
# data point classification (+/-1, float). All points were within the # unit square [-1,1]x[-1,1].
inSamplePath = "in.data"
outSamplePath = "out.data"
main(inSamplePath, outSamplePath)
