import numpy as np
import sklearn as sl
from math import log10

NUM_OF_FEATURES = 48

def load_data(num_of_features):
    with open('spambase.data', 'r') as f:
        line = f.readline()
        X, y = [], []
        while line:
            array = [float(d) for d in line.replace('\n', '').split(',')]
            X.append(np.array(array[:num_of_features]))
            y.append(array[-1])
            line = f.readline()
    return X, y

def split_dataset(X, y):
    class_spam = [X[i] for i in range(len(X)) if y[i] == 1]
    class_ham = [X[i] for i in range(len(X)) if y[i] == 0]

    return class_spam, class_ham

def calc_likelihood():
    likelihood_spam = np.mean(class_spam, axis=0) / 100
    likelihood_ham = np.mean(class_ham, axis=0) / 100

    return likelihood_spam, likelihood_ham
    
def calc_prior_distribution():
    prior_spam = len(class_spam) / ( len(class_spam) + len(class_ham) )
    prior_ham = len(class_ham) / ( len(class_spam) + len(class_ham) )

    return prior_spam, prior_ham

def predict(x):
    assert(len(x) == NUM_OF_FEATURES)
    x = [int(i > 0) for i in x]
    prob_is_spam = log10(prior_spam)
    prob_is_ham = log10(prior_ham)
    for i, v in enumerate(x):
        if v:
            prob_is_spam += log10(likelihood_spam[i])
            prob_is_ham += log10(likelihood_ham[i])
        else:
            prob_is_spam += log10(1 - likelihood_spam[i])
            prob_is_ham += log10(1 - likelihood_ham[i])
    print(prob_is_spam, prob_is_ham)
    if prob_is_spam > prob_is_ham:
        return 1
    return 0

X, y = load_data(48)

class_spam, class_ham = split_dataset(X, y)
likelihood_spam, likelihood_ham = calc_likelihood()
prior_spam, prior_ham = calc_prior_distribution()



