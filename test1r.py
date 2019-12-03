import numpy as np
from sklearn.svm import SVC
import pickle


with open('clf.pickle', 'rb') as f:
    clf = pickle.load(f)
print('predict [-0.8, -1]:', clf.predict([[-0.8, -1]]))
score = clf.score([[-1, -0.5], [-1.5, -1], [1, 1.5], [2.5, 1]], [1, 1, 2, 2])
print("The score is : %f" % score)