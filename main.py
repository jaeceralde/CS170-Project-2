import project
from project import * # you have to import functions so this is importing all of them


# you dont need a main function in python
# to run just type python main.py

rand = getRand() * 100  # convert the random number into a percentage
# print(rand)
rand = "{:.2f}".format(rand)    # converts the percent into a string & limits the decimal to 2

print('Welcome to Feature Selection Algorithm.\n')

numFeatures = int(input('Please enter total number of features: '))

print('\nType the number of the algorithm you want to run.\n')
print('1. Forward Selection')
print('2. Backward Selection\n')

numAlgo = int(input())

print('\nUsing no features and \"random\" evaluation, I get an accuracy of ' + rand + '%\n')

# testing the forward selection function :,) 
best_subset, best_acc = forward_selection(numFeatures)
print('\nThe highest accuracy is ' + str(best_acc) + ' and the best subset is ' + str(best_subset) + '%\n')