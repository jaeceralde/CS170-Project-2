import random
import numpy as np
from collections import deque

# custom print for the lists so they are sorted and easier to read
def custom_print_list(lst):
    sorted_keys = sorted(lst)
    return ','.join(map(str, sorted_keys))


def getRand():
    return random.random()


class node:
    # constructor for the node class
    def __init__(self, accuracy = 0, remainingFeatures = None, featuresSubset = None):
        self.accuracy = accuracy 
        self.remainingFeatures = remainingFeatures if remainingFeatures is not None else np.array(0) 
        self.featuresSubset = featuresSubset if featuresSubset is not None else []


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
                new_features.append(feature) # add the current feature to the new subset of features

                # checking if it was visited
                features_tuple = tuple(new_features)
                if features_tuple in visited:
                    continue
                visited.add(features_tuple)

                # checking if the features have changed size so we can show whats the best accuracy so far
                if len(features_tuple) > cur_size:
                    print(f'\nFeature set {{{custom_print_list(best_subset)}}} was best, accuracy ' + 'is {:.2f}%\n'.format(best_accuracy))
                    cur_size = len(features_tuple)

                accuracy = getRand() * 100 # generate a random accuracy score (for now)

                print(f'\tUsing feature(s) {{{custom_print_list(new_features)}}}' + ' accuracy is {:.2f}%'.format(accuracy))

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
                    if accuracy < best_accuracy:
                        print('\n(Warning, accuracy has decreased!)')
                    return best_subset, best_accuracy

    return best_subset, best_accuracy 


# same thing as the forward function but backwards 
def backward_selection(num_features):
    queue = deque()
    best_subset = []
    best_accuracy = 0
    cur_size = num_features
    visited = set()

    # initialize the queue with all features selected
    queue.append(node(featuresSubset=[i for i in range(num_features)]))

    while queue:
        curr_node = queue.popleft()
        
        if len(curr_node.featuresSubset) == 0:
            if accuracy < best_accuracy:
                print('\n(Warning, accuracy has decreased!)')
            break

        # checking if it was visited
        features_tuple = tuple(curr_node.featuresSubset)
        if features_tuple in visited:
            continue
        visited.add(features_tuple)

        # calculate the accuracy (placeholder for now)
        accuracy = getRand() * 100
        
        if cur_size > len(curr_node.featuresSubset):
            print(f'\nFeature set {{{custom_print_list(best_subset)}}} was best, accuracy ' + 'is {:.2f}%\n'.format(best_accuracy))
            cur_size = len(curr_node.featuresSubset)

        print(f'\tUsing feature(s) {{{custom_print_list(curr_node.featuresSubset)}}}' + ' accuracy is {:.2f}%'.format(accuracy))

        # update the best accuracy and best subset if the current subset performs better
        if accuracy > best_accuracy:
            best_accuracy = accuracy 
            best_subset = curr_node.featuresSubset

        # iterate over each feature and remove it from the subset
        for feature in curr_node.featuresSubset:
            new_features = [f for f in curr_node.featuresSubset if f != feature]

            # add the new subset to the queue
            queue.append(node(featuresSubset = new_features))

    return best_subset, best_accuracy