import random
import numpy as np

def getRand():
    return random.random()

class node:
    #accuracy = 0 # keeps track of accuracy score
    #remainingFeatures = np.array(0) # keeps track of unvisited states(?)
    #featuresSubset = {} # keeps track of visited states(?)

    # below i created a constructor for the node class
    def __init__(self, accuracy = 0, remainingFeatures = None, featuresSubset = None):
        self.accuracy = accuracy 
        self.remainingFeatures = remainingFeatures if remainingFeatures is not None else np.array(0) 
        self.featuresSubset = featuresSubset if featuresSubset is not None else {}

def forward_selection(num_features):
    queue = [] 
    best_subset = []
    best_accuracy = 0

    # adding a node that represents the initial state (aka no features selected yet) to the queue
    queue.append(node(remainingFeatures = np.arange(num_features)))

    # starting the bfs loop 
    while queue:
        curr_node = queue.pop() #removing and retrieving the first node from the queue

        # iterate over the remaining features of the current node
        for feature in curr_node.remainingFeatures:
            if feature not in curr_node.featuresSubset: # checks if the feature has been selected in the current subset
                new_features = curr_node.featuresSubset.copy() # create a copy of the current subset of features
                new_features[feature] = True # add the current feature to the new subset of features
                accuracy = getRand() * 100 # generate a random accuracy score (for now i think??)

                # update the best accuracy and best subset if the current subset performs better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy 
                    best_subset = new_features
                
                # generate the remaining features by excluding the current feature
                remaining_features = np.setdiff1d(curr_node.remainingFeatures, [feature])

                # add a new node representing the updated subset of features to the queue
                queue.append(node(remainingFeatures = remaining_features, featuresSubset = new_features, accuracy = accuracy))

    return best_subset, best_accuracy 
