import numpy as np


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
            train_data = np.delete(data, i, axis = 0)
            # select the ith instance from the original dataset to be used as the test instance 
            test_instance = data[i]

            print(f'The test instance\'s actual label is:', test_instance[0])   # for trace

            # train the classifier
            classifier.train(train_data, selected_features)
            # test the classifier
            predictor = classifier.test(test_instance)

            print(f'The predictor says the test instance\'s label is:', predictor)  # for trace

            # if the predictor is the actual label
            if predictor == test_instance[0]:
                correct_pred += 1
                print('\tRIGHT PREDICTION\n')   # for trace
            else:
                print('\tWRONG PREDICTION\n')   # for trace

        return correct_pred / total
        
        
def load(filename):
    data = np.loadtxt(filename)

    # seperate labels and features 
    labels = data[:, 0]
    features = data[:, 1:]

    # normalize the features 
    features = (features - np.mean(features, axis = 0) / np.std(features, axis = 0))

    # combine the labels and normalized features 
    data = np.column_stack((labels, features))
    
    return data