# Starter code for CS 165B HW3
import os
import time
import math
import numpy as np
import scipy
from scipy import sparse
from scipy import linalg
import scipy.sparse.linalg as spla
import pprint
def run_train_test(training_file, testing_file):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition. 

    Inputs:
        training_file: file object returned by open('training.txt', 'r')
        testing_file: file object returned by open('test1/2/3.txt', 'r')

    Output:
        Dictionary of result values 

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED
        
        Example:
            return {
    			"gini":{
    				'True positives':0, 
    				'True negatives':0, 
    				'False positives':0, 
    				'False negatives':0, 
    				'Error rate':0.00
    				},
    			"entropy":{
    				'True positives':0, 
    				'True negatives':0, 
    				'False positives':0, 
    				'False negatives':0, 
    				'Error rate':0.00}
    				}
    
    """
    #data = [[str(y) for y in x.strip().split(" ")] for x in testing_file]
    #data[0] = [str(x) for x in data[0]]
    #print(data)

    training_data = parse_data(training_file)
    test_data = parse_data(testing_file)
    header = parse_header(training_file)
    pass


def parse_data(filename):
    """
    This function parses the data for the decision tree algorithm
    """
    header = filename.readline()
    header = ([[str(y) for y in header.strip().split(" ")][1:]])
    data = filename.readlines()
    data = ([[int(y) for y in x.strip().split(" ")] for x in data])
    print(header)
    print(data)
    return(data)


#######
# The following functions are provided for you to test your classifier.
#######

if __name__ == "__main__":
    """
    You can use this to test your code.
    python hw3.py [training file path] [testing file path]
    """
    import sys

    training_file = open(sys.argv[1], "r")
    testing_file = open(sys.argv[2], "r")

    run_train_test(training_file, testing_file)

    training_file.close()
    testing_file.close()

