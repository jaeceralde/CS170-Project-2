import project
from project import *

# load datasets
small_data = load('small-test-dataset.txt')
large_data = load('large-test-dataset.txt')

# initialize classifier and validator 
classifier = Classifier()
validator = Validator()


# calculated the accuracy of the nearest neighbor classifier on the small dataset
small_features = [3, 5, 7]
print('Using feature(s): ' + ''.join(str(small_features)) + '\n')  # print out small_features for trace
small_acc = validator.leave_one_out_validation(classifier, small_data, small_features)
print(f'Accuracy on small dataset: {small_acc:.2f}\n')  # newline for trace format

# calculated the accuracy of the nearest neighbor classifier on the large dataset
large_features = [1, 15, 27]
print('Using feature(s): ' + ''.join(str(large_features)) + '\n')  # print out large_features for trace
large_acc = validator.leave_one_out_validation(classifier, large_data, large_features)
print(f'Accuracy on large dataset: {large_acc:.2f}')