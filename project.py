import numpy as np


class Classifier:
    def __init__(self):
        self.train_data = None # list of the training data
        self.selected_features = None # the features that are inputted
        
    def train(self, data, selected_features):
        self.train_data = data
        self.selected_features = selected_features
        
    def euclidean_distance(self, p1, p2):
        
    
    def test(self, instance):
        # here we go through train_data and find which is the closest point
        
    

class Validator:
    def leave_one_out_validation(self, classifier, data, selected_features):
        correct_pred = 0
        total = len(data)
        # go through all the data and run train and test and count how many
        # are right then return correct_pred / total
        
        
def load(filename):
    data = np.loadtxt(filename)

    # figure out what to do with data here
    
    return data