import project
from project import *


rand = getRand() * 100  # convert the random number into a percentage
rand = "{:.2f}".format(rand)    # converts the percent into a string & limits the decimal to 2

print('Welcome to Feature Selection Algorithm.\n')

numFeatures = int(input('Please enter total number of features: '))

print('\nType the number of the algorithm you want to run.\n')
print('1. Forward Selection')
print('2. Backward Selection\n')

numAlgo = int(input())

print('\nUsing no features and \"random\" evaluation, I get an accuracy of ' + rand + '%\n')

if (numAlgo == 1):
    # testing the forward selection function :,) 
    best_subset, best_acc = forward_selection(numFeatures)

    print(f'\nThe best feature subset is {{{custom_print_list(best_subset)}}}' + ' which has an accuracy of {:.2f}%'.format(best_acc))
elif (numAlgo == 2):
    best_subset, best_acc = backward_selection(numFeatures)
    
    print(f'\nThe best feature subset is {{{custom_print_list(best_subset)}}}' + ' which has an accuracy of {:.2f}%'.format(best_acc))