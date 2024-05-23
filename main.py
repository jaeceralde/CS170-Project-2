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
small_acc = validator.leave_one_out_validation(classifier, small_data, small_features)
print(f'Accuracy on small dataset: {small_acc:.2f}')

# calculated the accuracy of the nearest neighbor classifier on the large dataset
large_features = [1, 15, 27]
large_acc = validator.leave_one_out_validation(classifier, large_data, large_features)
print(f'Accuracy on large dataset: {large_acc:.2f}')