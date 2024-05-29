import project
from project import *

print('Welcome to Feature Selection Algorithm.')
filename = input('Type in the name of the file to test: ')

dataset = load(filename)

# print(f'This data set has ' + numFeatures + ' with ' + numInstances + '.\n')  #FIXME

print('\nType the number of the algorithm you want to run.\n')
print('1. Forward Selection')
print('2. Backward Selection\n')

numAlgo = int(input())

# initialize classifier and validator 
classifier = Classifier()
validator = Validator()

# FIXME
# mostCommonClass = most_common(dataset.labels)

if (numAlgo == 1):  # forward selection function
    # FIXME
    # print('Running nearest neighbor with no features (default rate), using \"leaving-one-out\" evaluation, I get an accuracy of ' + default(dataS.most_common(1), size(dataset.features)) + '.\n')
    
    best_subset, best_acc = forward_selection(numFeatures)

    print('\nFinished search!')
    print(f'\nThe best feature subset is {{{custom_print_list(best_subset)}}}' + ' which has an accuracy of {:.2f}%'.format(best_acc))

elif (numAlgo == 2):    # backward selection function
    best_subset, best_acc = backward_selection(numFeatures)
    
    # FIXME
    # print('Running nearest neighbor with no features (default rate), using \"leaving-one-out\" evaluation, I get an accuracy of ' + default(x,y) + '.\n')
    
    print('\nFinished search!')
    print(f'\nThe best feature subset is {{{custom_print_list(best_subset)}}}' + ' which has an accuracy of {:.2f}%'.format(best_acc))