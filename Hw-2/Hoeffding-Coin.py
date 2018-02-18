"""
CS 156a: Homework #2
Anthony Bartolotta
Problems 1,2
"""

import numpy as np

def generateCoinFlips(nCoins, nFlips):
    # Flip (nCoins) coins (nFlips) times. Count fraction of heads. 
    flips = (np.random.uniform(low=-1, high=1, size=[nCoins,nFlips]) > 0)
    fHeads = np.sum(flips,axis=1) / float(nFlips)
    # Count fraction of heads for first, random, and minimum coin
    nu1 = fHeads[0]
    nuRand = fHeads[np.random.randint(low=0,high=nCoins)]
    nuMin = min(fHeads)
    return np.array([nu1, nuRand, nuMin])

def collectStatistics(nCoins, nFlips, nTrials):
    # Flip (nCoins) coins (nFlips) times for (nTrials) trials.
    nuStatistics = np.array([generateCoinFlips(nCoins,nFlips) 
                                                for j in range(nTrials)])
    return nuStatistics

def reduceData(nuStatistics):
    # Find absolute deviation from mean
    nuDeviation = np.abs(nuStatistics-.5)
    # Find distribution of deviations for each coin
    devNu1 = np.unique(nuDeviation[:,0], return_counts=True)
    devNuRand = np.unique(nuDeviation[:,1], return_counts=True)
    devNuMin = np.unique(nuDeviation[:,2], return_counts=True)
    return [devNu1, devNuRand, devNuMin]

def satisfiesHoeffding(devNu, nFlips, nTrials):
    # Check if statistics of a coin satisfies the Hoeffding
    # inequality
    deviation = devNu[0]
    fTrials = devNu[1] / float(nTrials)
    pLargeDeviation = np.array([np.sum(fTrials[j:]) 
                                            for j in range(len(fTrials))])
    hoeffdingLimit = 2*np.exp(-2*nFlips*(deviation**2))
    isHoeffding = all( (hoeffdingLimit - pLargeDeviation) >= 0)
    return isHoeffding

def main(nCoins, nFlips, nTrials):
    # Collect statistics
    nuStatistics = collectStatistics(nCoins, nFlips, nTrials)
    # Return mean of nu_min
    meanNuMin = np.mean(nuStatistics[:,2])
    print("The mean of nu_{min} = " + repr(meanNuMin))
    # Find distribution of deviations about the mean
    [devNu1, devNuRand, devNuMin] = reduceData(nuStatistics)
    # Check which set of coins selected satisfies Hoeffding
    hoeffding1 = satisfiesHoeffding(devNu1, nFlips, nTrials)
    hoeffdingRand = satisfiesHoeffding(devNuRand, nFlips, nTrials)
    hoeffdingMin = satisfiesHoeffding(devNuMin, nFlips, nTrials)
    print("The first coin satisfies Hoeffding? = "+ repr(hoeffding1))
    print("A random coin satisfies Hoeffding? = "+ repr(hoeffdingRand))
    print("The minimum coin satisfies Hoeffding? = "+ repr(hoeffdingMin))
    return

main(1000, 10, 100000)
