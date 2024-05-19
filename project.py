import random
import numpy as np

def getRand():
    return random.random()

class node:
    # constructor for the node class
    def __init__(self, accuracy = 0, remainingFeatures = None, featuresSubset = None):
        self.accuracy = accuracy 
        self.remainingFeatures = remainingFeatures if remainingFeatures is not None else np.array(0) 
        self.featuresSubset = featuresSubset if featuresSubset is not None else {}

def forward_selection(num_features):
    queue = [] 
    best_subset = []
    best_accuracy = 0
    visited = set()

    # adding a node that represents the initial state (aka no features selected yet) to the queue
    queue.append(node(remainingFeatures = np.arange(num_features)))
    
    print('Beginning search.\n')
    cur_size = 1

    # starting the bfs loop 
    while queue:
        curr_node = queue.pop() #removing and retrieving the first node from the queue

        # iterate over the remaining features of the current node
        for feature in curr_node.remainingFeatures:
            if feature not in visited: # checks if the feature has been selected in the current subset
                new_features = curr_node.featuresSubset.copy() # create a copy of the current subset of features
                new_features[feature] = True # add the current feature to the new subset of features

                # checking if it was visited
                features_tuple = tuple(sorted(new_features.keys()))
                if features_tuple in visited:
                    continue
                visited.add(features_tuple)

                # checking if the features have changed size so we can show whats the best accuracy so far
                if len(features_tuple) > cur_size:
                    best_subset_sofar = sorted(best_subset.keys())
                    combined_best_subset = ','.join(map(str, best_subset_sofar))
                    print(f'\nFeature set {{{combined_best_subset}}} was best, accuracy ' + 'is {:.2f}%\n'.format(best_accuracy))
                    cur_size = len(features_tuple)

                accuracy = getRand() * 100 # generate a random accuracy score (for now)
                
                sorted_keys = sorted(new_features.keys())
                sorted_features = ','.join(map(str, sorted_keys))
                
                print(f'\tUsing feature(s) {{{sorted_features}}}' + ' accuracy is {:.2f}%'.format(accuracy))

                # update the best accuracy and best subset if the current subset performs better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy 
                    best_subset = new_features
                
                # generate the remaining features by excluding the current feature
                remaining_features = np.setdiff1d(curr_node.remainingFeatures, [feature])

                # add a new node representing the updated subset of features to the queue
                queue.append(node(remainingFeatures = remaining_features, featuresSubset = new_features, accuracy = accuracy))
                
                # if found then end
                if len(features_tuple) == num_features:
                    return best_subset, best_accuracy

    return best_subset, best_accuracy 

# same thing as the forward function but backwards 
def backward_selection(num_features):
    queue = [] 
    best_subset = []
    best_accuracy = 0

    # initialize the queue with all features selected
    queue.append(node(featuresSubset={i: True for i in range(num_features)}))

    while queue:
        curr_node = queue.pop()

        # calculate the accuracy (placeholder for now b/c i think its supposed to calculate it based on the current feature subset tho)
        accuracy = getRand() * 100
        
        sorted_keys = sorted(curr_node.featuresSubset.keys())
        sorted_features = ','.join(map(str, sorted_keys))
        
        print(f'Feature set {{{sorted_features}}}' + ' accuracy is {:.2f}%'.format(accuracy))

        # update the best accuracy and best subset if the current subset performs better
        if accuracy > best_accuracy:
            best_accuracy = accuracy 
            best_subset = curr_node.featuresSubset

        # iterate over each feature and remove it from the subset
        for feature in curr_node.featuresSubset:
            new_features = curr_node.featuresSubset.copy()
            del new_features[feature]

            # add the new subset to the queue
            queue.append(node(featuresSubset=new_features))

    return best_subset, best_accuracy
