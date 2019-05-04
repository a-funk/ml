# Starter code for CS 165B HW2 Spring 2019

def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition. 
    You are permitted to use the numpy library but you must write 
    your own code for the linear classifier. 

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values 

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED
        
        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """
    
    heading = training_input[0]
    print(heading)

    numA = heading[1]
    numB = heading[2]
    numC = heading[3]

    dim  = heading[0]           #Number of columns of data (num features)
    n = numA+numB+numC          #Number of entries (rows of initial data)
    num_per_class = np.array([numA, numB, numC])

    centA = calc_centroid(dim, num_per_class, training_input, "A")
    print(centA)    
    centB = calc_centroid(dim, num_per_class, training_input, "B")
    print(centB)
    centC = calc_centroid(dim, num_per_class, training_input, "C")
    print(centC)
    #pass


"""
dim is dimension of data
n is the numX 
data is the training input
clust is class A,B,C
"""
def calc_centroid(dim, num_per_class, data, clust):
    centroid = np.zeros(dim)
    n = num_per_class
    if(clust == "A"):
        for i in range(1,n[0]+1):  #Ranges from data[1:numA+1] 
            for j in range (0, dim):
                centroid[j] = centroid[j]+data[i][j]
        for i in range(0,dim):
            centroid[i] = centroid[i]/n[0]

    elif(clust == "B"):
        for i in range(n[0]+1,n[0]+n[1]+1):
            for j in range (0, dim):
                centroid[j] = centroid[j]+data[i][j]
        for i in range(0,dim):
            centroid[i] = centroid[i]/n[1]

    elif(clust == "C"):
        for i in range(n[0]+n[1]+1,n[0]+n[1]+n[2]+1):
            for j in range (0, dim):
                centroid[j] = centroid[j]+data[i][j]
        for i in range(0,dim):
            centroid[i] = centroid[i]/n[2]
    

    return centroid



#######
# The following functions are provided for you to test your classifier.
######
def parse_file(filename):
    """
    This function is provided to you as an example of the preprocessing we do
    prior to calling run_train_test
    """
    with open(filename, "r") as f:
        data = [[float(y) for y in x.strip().split(" ")] for x in f]
        data[0] = [int(x) for x in data[0]]

        return data

if __name__ == "__main__":
    import os
    import time
    import math
    import numpy as np
    import scipy
    from scipy import sparse
    from scipy import linalg
    import scipy.sparse.linalg as spla
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import axes3d
    """
    You can use this to test your code.
    python hw2.py [training file path] [testing file path]
    """
    import sys
    try:
        training_input = parse_file(sys.argv[1])
    except:
        pass
    try:
        testing_input = parse_file(sys.argv[2])
    except:
        pass

    try:
        run_train_test(training_input, testing_input)
    except:
        pass

