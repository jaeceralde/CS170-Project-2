import project
import time
from project import *

print('Welcome to Feature Selection Algorithm.')
filename = input('Type in the name of the file to test: ')

dataset = load(filename)
labels = dataset[:, 0]
features = dataset[:, 1:]

data_dim = dataset.shape
numFeatures = data_dim[1] - 1   # offset to not count the labels column
numInstances = data_dim[0]

print('\nType the number of the algorithm you want to run.\n')
print('1. Forward Selection')
print('2. Backward Selection\n')

numAlgo = int(input())

print(f'\nThis data set has ' + str(numFeatures) + ' features with ' + str(numInstances) + ' instances.\n')

mostCommonClass = most_common(labels)    # assuming the label is the class
defaultrate = default(mostCommonClass[1], numInstances)



if (numAlgo == 1):  # forward selection function
    print('Running nearest neighbor with no features (default rate), using \"leaving-one-out\" evaluation, I get an accuracy of {:.2f}%'.format(defaultrate * 100))
    best_subset, best_acc = forward_selection(numFeatures, dataset)
    print('\nFinished search!')
    print(f'\nThe best feature subset is {{{custom_print_list(best_subset)}}}' + ' which has an accuracy of {:.2f}%'.format(best_acc))

elif (numAlgo == 2):    # backward selection function
    best_subset, best_acc = backward_selection(numFeatures, dataset, defaultrate)
    print('\nFinished search!')
    print(f'\nThe best feature subset is {{{custom_print_list(best_subset)}}}' + ' which has an accuracy of {:.2f}%'.format(best_acc))