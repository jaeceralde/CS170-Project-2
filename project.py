import random
import numpy as np

def getRand():
    return random.random()

class node:
    accuracy = 0 # keeps track of accuracy score
    remainingFeatures = np.array(0) # keeps track of unvisited states(?)
    featuresSubset = {} # keeps track of visited states(?)