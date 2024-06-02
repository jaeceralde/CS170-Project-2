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
    total_best_subset = []
    total_best_accuracy = 0.0

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

        if total_best_accuracy < best_accuracy:
            total_best_accuracy = best_accuracy
            total_best_subset = best_subset.copy()

    if total_best_accuracy > best_accuracy:
        print('(Warning, accuracy has decreased!)')

    return total_best_subset, total_best_accuracy



# same thing as the forward function but backwards 
def backward_selection(num_features, data, defaultrate):
    best_subset = list(range(1, num_features + 1))  # initialize the starting feature subset with all features
    best_accuracy = defaultrate * 100  # initialize the best accuracy with the default rate
    # initialize the overall best subset and accuracy with the starting subset and accuracy
    overall_best_subset = best_subset.copy() 
    overall_best_accuracy = best_accuracy

    # create instances of Validator and Classifier
    validator = Validator()
    classifier = Classifier()

    print('Beginning search.\n')

    # initial evaluation with all features
    initial_accuracy = validator.leave_one_out_validation(classifier, data, best_subset) * 100

    # check if initial accuracy is lower than default rate
    if initial_accuracy < best_accuracy:
        print(f'\n(Warning, accuracy has decreased!)') # if so, print warning that accuracy has decreased
    else:
        # update overall best accuracy and subset if initial accuracy is better
        overall_best_accuracy = initial_accuracy
        overall_best_subset = best_subset.copy()

    #set best accuracy to initial accuracy
    best_accuracy = initial_accuracy
    # print initial evaluation 
    print(f'\tUsing feature(s) {{{custom_print_list(best_subset)}}} accuracy is {initial_accuracy:.2f}%')
    print(f'\nFeature set {{{custom_print_list(best_subset)}}} was best, accuracy is {best_accuracy:.2f}%\n')

    # iterate through each feature to remove it and evaluate accuracy
    for i in range(num_features, 0, -1):
        current_best_accuracy = 0.0
        feature_to_remove = -1

        # iterate through each feature in the current subset
        for feature in best_subset:
            subset = [f for f in best_subset if f != feature] # remove the current feature and create a new subset
            accuracy = validator.leave_one_out_validation(classifier, data, subset) * 100 # calculate accuracy 
            print(f'\tUsing feature(s) {{{custom_print_list(subset)}}} accuracy is {accuracy:.2f}%') #print accuracy

            # update current best accuracy and feature to remove if accuracy is better
            if accuracy > current_best_accuracy:
                current_best_accuracy = accuracy
                feature_to_remove = feature

        # remove the feature that has the best accuracy
        if feature_to_remove != -1:
            best_subset.remove(feature_to_remove)
            print(f'\nFeature set {{{custom_print_list(best_subset)}}} was best, accuracy is {current_best_accuracy:.2f}%\n')

            # update overall best accuracy and subset if current best accuracy is better
            if current_best_accuracy > overall_best_accuracy:
                overall_best_accuracy = current_best_accuracy
                overall_best_subset = best_subset.copy()
            elif current_best_accuracy < overall_best_accuracy: # print warning if current best accuracy is worse than overall best accuracy
                print(f'\n(Warning, accuracy has decreased!)')

    return overall_best_subset, overall_best_accuracy