import project
from project import *


small_data = load('small-test-dataset.txt')
large_data = load('large-test-dataset.txt')



small_feat = [3, 5, 7]
small_acc = 0
print(f'Accuracy on small dataset: {small_acc:.2f}')

large_feat = [1, 15, 27]
large_acc = 0
print(f'Accuracy on large dataset: {large_acc:.2f}')