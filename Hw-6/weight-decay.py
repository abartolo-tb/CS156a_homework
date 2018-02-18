"""
CS 156a: Homework #6
Anthony Bartolotta
Problems 2,3,4,5,6
"""
import numpy as np

def pseudoInverse(X, lamb):
    xPrime = np.dot(np.transpose(X), X)
    L = lamb * np.identity(xPrime.shape[0])
    xPseudo = np.dot(np.linalg.inv(xPrime + L), np.transpose(X))
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

def weightDecayLinearRegression(data, k, decayBool):
    [trueClassVec, sampleVecs] = data
    # Calculate pseudo-inverse
    if decayBool:
        # Calculate pseudo-inverse including weight decay
        xPseudo = pseudoInverse(sampleVecs, 10**k)
    else:
        # Calculate pseudo-inverse without weight decay
        xPseudo = pseudoInverse(sampleVecs, 0)
    # Perform linear regression using pseudo-inverse
    wVec = np.dot(xPseudo, trueClassVec)
    return wVec

def singleTrial(inSamplePath, outSamplePath, k, decayBool):
    # Load in-sample and out-of-sample data from files
    inSampleRaw = readInData(inSamplePath)
    outSampleRaw = readInData(outSamplePath)
    # Perform non-linear transformation of data
    inData = nonlinearTransform(inSampleRaw)
    outData = nonlinearTransform(outSampleRaw)
    # Use linear regression to approximate target function
    wVec = weightDecayLinearRegression(inData, k, decayBool)
    # Calculate in-sample and out-of-sample error
    eInSample = sampleError(inData, wVec)
    eOutSample = sampleError(outData, wVec)
    return [eInSample, eOutSample]

def main(inSamplePath, outSamplePath):
    [eIn, eOut] = singleTrial(inSamplePath, outSamplePath, 0, False)
    print("For linear regression without weight decay: ")
    print("In-Sample Error = " + repr(eIn))
    print("Out-of-Sample Error = " + repr(eOut) + "\n")
    for k in [-3, -2, -1, 0, 1, 2, 3]:
            [eIn, eOut] = singleTrial(inSamplePath, outSamplePath, k, True)
            print("Weight decay with k = " + repr(k) + " :")
            print("In-Sample Error = " + repr(eIn))
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
