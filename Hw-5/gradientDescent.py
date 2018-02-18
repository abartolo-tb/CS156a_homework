"""
CS 156a: Homework #5
Anthony Bartolotta
Problems 5,6,7
"""
import numpy as np

def error(coords):
    # Error function
    [u,v] = coords
    err = (u*np.exp(v) - 2*v*np.exp(-u))**2
    return err

def dEdu(coords):
    # Partial derivative of error function with respect to u
    [u,v] = coords
    dE = 2*(u*np.exp(v) - 2*v*np.exp(-u) )*(np.exp(v) + 2*v*np.exp(-u))
    return dE

def dEdv(coords):
    # Partial derivative of error function with respect to v
    [u,v] = coords
    dE = 2*(u*np.exp(v) - 2*v*np.exp(-u) )*(u*np.exp(v) - 2*np.exp(-u))
    return dE

def gradE(coords):
    # Return gradient of error function at given coordinates
    grad = np.array([dEdu(coords), dEdv(coords)])
    return grad

def gradDescFullStep(coords,nu):
    # Perform a single step of gradient descent
    newCoords = coords - nu*gradE(coords)
    return newCoords

def coordDescFullStep(coords,nu):
    # Perform a full iteration of coordinate descent
    newCoords1 = np.array([coords[0] - nu*dEdu(coords), coords[1]])
    newCoords2 = np.array([newCoords1[0], newCoords1[1]- nu*dEdv(newCoords1)])
    return newCoords2

def main_5_6(coords, nu):
    targetE = 10**(-14)
    currentE = error(coords)
    nSteps = 0
    while (currentE > targetE):
        coords = gradDescFullStep(coords, nu)
        currentE = error(coords)
        nSteps += 1
    print("Number of steps: " + repr(nSteps))
    print("Final coordinates: " + repr(coords))
    return

def main_7(coords, nu):
    nSteps = 0
    while (nSteps < 15):
        coords = coordDescFullStep(coords, nu)
        nSteps += 1
    finalError = error(coords)
    print("Final error: " + repr(finalError))
    return

coords = np.array([1.0, 1.0])
nu = 0.1
main_5_6(coords, nu)
main_7(coords, nu)
