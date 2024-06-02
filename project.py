import numpy as np
from collections import deque
from collections import Counter


def load(filename):
    data = np.loadtxt(filename)

    # seperate labels and features 
    labels = data[:, 0]
    features = data[:, 1:]

    # normalize the features 
    features = (features - np.mean(features, axis = 0)) / np.std(features, axis = 0)

    # combine the labels and normalized features 
    data = np.column_stack((labels, features))
    
    return data



def most_common(list):
    commonclass = Counter(list)
    return commonclass.most_common(1)[0]  # returns the most common class & its instance count



def default(commonclass, setsize):
    return commonclass / setsize



# custom print for the lists so they are sorted and easier to read
def custom_print_list(lst):
    sorted_keys = sorted(lst)
    return ','.join(map(str, sorted_keys))



class node:
    # constructor for the node class
    def __init__(self, accuracy = 0, remainingFeatures = None, featuresSubset = None):
        self.accuracy = accuracy 
        self.remainingFeatures = remainingFeatures if remainingFeatures is not None else np.array(0) 
        self.featuresSubset = featuresSubset if featuresSubset is not None else []



class Classifier:
    def __init__(self):
        self.train_data = None  # list of the training data
        self.selected_features = None   # the features that are inputted
        
    def train(self, data, selected_features):
        self.train_data = data
        self.selected_features = selected_features
        
    def euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((p1[self.selected_features] - p2[self.selected_features]) ** 2))  # calculate euclidean distance 
    
    # here we go through train_data and find which is the closest point
    def test(self, instance):
        # find the nearest neighbor 
        # compute the distance between a given test instance and all training instances in the dataset
        distance = [self.euclidean_distance(instance, train_instance) for train_instance in self.train_data]
        nn_index = np.argmin(distance)  # get the index from the smallest distance
        return self.train_data[nn_index, 0] # return the class label of the nearest neighbor



class Validator:
    def leave_one_out_validation(self, classifier, data, selected_features):
        correct_pred = 0
        total = len(data)
        
        # go through all the data and run train and test
        # count how many are right then return correct_pred / total
        for i in range(total):
            # remove the test instance from the original dataset
            train_data = np.concatenate((data[:i], data[i+1:]), axis=0)
            # select the ith instance from the original dataset to be used as the test instance 
            test_instance = data[i]

            # train the classifier
            classifier.train(train_data, selected_features)
            # test the classifier
            predictor = classifier.test(test_instance)
            
            # if the predictor is the actual label
            if predictor == test_instance[0]:
                correct_pred += 1

        return correct_pred / total
    


def forward_selection(num_features, data):
    best_subset = [] # holder for the best feature set
    best_accuracy = 0.0

    validator = Validator()
    classifier = Classifier()

    num_features_loop = num_features+1

    print('\nBeginning search.\n')

    for i in range(1, num_features_loop):
        best_feature_acc = 0
        curr_feature = 0
        
        for feature in range(1, num_features_loop):
            # check if the features in the subset already
            if feature not in best_subset:
                # if its not in the subset then add it
                current_subset = best_subset + [feature]
                # test accuracy
                accuracy = validator.leave_one_out_validation(classifier, data, current_subset) * 100
                print(f'\tUsing feature(s) {{{custom_print_list(current_subset)}}} accuracy is {accuracy:.2f}%')

                # if its the best accuracy, save it
                if accuracy > best_feature_acc:
                    best_feature_acc = accuracy
                    curr_feature = feature
        
        best_subset.append(curr_feature) # add the best feature to the end
        best_accuracy = best_feature_acc
        print(f'\nFeature set {{{custom_print_list(best_subset)}}} was best, accuracy is {best_accuracy:.2f}%\n')

    return best_subset, best_accuracy



# same thing as the forward function but backwards 
def backward_selection(num_features, data, defaultrate):
    queue = deque()
    best_subset = []
    best_accuracy = 0.0
    cur_size = num_features
    visited = set()
    
    validator = Validator()
    classifier = Classifier()

    # initialize the queue with all features selected
    queue.append(node(featuresSubset=[i for i in range(1, num_features+1)]))
    
    print('Beginning search.\n')

    while queue:
        curr_node = queue.popleft()
        
        if len(curr_node.featuresSubset) == 0:
            print('\nRunning nearest neighbor with no features (default rate), using \"leaving-one-out\" evaluation, I get an accuracy of {:.2f}%'.format(defaultrate * 100))
            
            if defaultrate < best_accuracy:
                print('\n(Warning, accuracy has decreased!)')
            break

        # checking if it was visited
        features_tuple = tuple(curr_node.featuresSubset)
        if features_tuple in visited:
            continue
        visited.add(features_tuple)

        # evaluate accuracy using leave-one-out cross-validation
        accuracy = validator.leave_one_out_validation(classifier, data, curr_node.featuresSubset) * 100
        
        
        if cur_size > len(curr_node.featuresSubset):
            print(f'\nFeature set {{{custom_print_list(best_subset)}}} was best, accuracy ' + 'is {:.2f}%\n'.format(best_accuracy))
            cur_size = len(curr_node.featuresSubset)

        print(f'\tUsing feature(s) {{{custom_print_list(curr_node.featuresSubset)}}}' + ' accuracy is {:.2f}%'.format(accuracy))

        # update the best accuracy and best subset if the current subset performs better
        if accuracy > best_accuracy:
            best_accuracy = accuracy 
            best_subset = curr_node.featuresSubset
        elif accuracy == best_accuracy and len(curr_node.featuresSubset) < len(best_subset):
            best_accuracy = accuracy 
            best_subset = curr_node.featuresSubset

        # iterate over each feature and remove it from the subset
        for feature in curr_node.featuresSubset:
            new_features = [f for f in curr_node.featuresSubset if f != feature]

            # add the new subset to the queue
            queue.append(node(featuresSubset = new_features))

    return best_subset, best_accuracy