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
    print("Training data: ", heading)

    numA = heading[1]
    numB = heading[2]
    numC = heading[3]

    dim  = heading[0]           #Number of columns of data (num features)
    n = numA+numB+numC          #Number of entries (rows of initial data)
    num_per_class = np.array([numA, numB, numC])

    centA = calc_centroid(dim, num_per_class, training_input, "A")
    centB = calc_centroid(dim, num_per_class, training_input, "B")
    centC = calc_centroid(dim, num_per_class, training_input, "C")
    print("Centroids calculated...")
    vectAB = calc_vect(centA, centB, dim)
    midAB = calc_mid(centA, centB, dim)
   
    vectBC = calc_vect(centB, centC, dim)
    midBC = calc_mid(centB, centC, dim)

    vectAC = calc_vect(centA, centC, dim)
    midAC = calc_mid(centA, centC, dim)
    print("Midpoints and vectors calculated...")

    #Here is high level info from the testing data
    heading = testing_input[0]
    actualA = heading[1]
    actualB = heading[2]
    actualC = heading[3]
    actual_per_class = np.array([actualA, actualB, actualC])
    dim = heading[0]
    print("Testing Data: ", heading )

    predA = np.array([])
    predB = np.array([])
    predC = np.array([])

    predA,predB,predC = dec_boundary(testing_input, vectAC, vectAB, vectBC, 
                                    midAC, midAB, midBC, actual_per_class, dim)
    
    print("Predicted A: ",predA,'\n',
         "Predicted B: ",predB,'\n',
         "Predicted C: ",predC,'\n')

    output_results = calc_rates(predA, predB, predC, actual_per_class)
    pp.pprint(output_results)
    return(output_results)


"""
Here are the calculations of the relevants attributes
True Positive Rate
False Positive Rate
Error Rate
Accuracy
Precision

"""
def calc_rates(predA, predB, predC, actual_per_class):
    #Getting the TP and FP counts, TPR and FPR
    total = actual_per_class[0]+actual_per_class[1]+actual_per_class[2] # Total examples
    tprA,fprA,accA,precA = ratesA(predA, predB, predC, actual_per_class, total)
    tprB,fprB,accB,precB = ratesB(predA, predB, predC, actual_per_class, total)
    tprC,fprC,accC,precC = ratesC(predA, predB, predC, actual_per_class, total)
    
    tpr = (tprA+tprB+tprC)/3.0  # Average true positive rate
    fpr = (fprA+fprB+fprC)/3.0  # Average false positive rate
    accuracy = (accA+accB+accC)/3.0
    error_rate = 1-accuracy
    precision = (precA+precB+precC)/3.0

    return {
                "True Postive Rate: ": tpr,
                "False Positive Rate: ": fpr,
                "Error Rate: ": error_rate,
                "Accuracy: ": accuracy,
                "Precision: ": precision
            }



def ratesA(predA, predB, predC, actual_per_class, total):
    #TP and FP and TN count for A
    true_pos =  predA[0]
    false_pos = predB[0]+predC[0]
    true_neg = predB[1]+predB[2]+predC[1]+predC[2]
    #TPR and FPR count for A
    tpr = true_pos/actual_per_class[0]
    fpr = false_pos/(actual_per_class[1]+actual_per_class[2])
    accuracy = (true_pos+true_neg)/total
    precision = true_pos/(true_pos+false_pos)
    return tpr, fpr, accuracy, precision

def ratesB(predA, predB, predC, actual_per_class, total):
    #TP and FP and TN count for B
    true_pos =  predB[1]
    false_pos = predA[1]+predC[1]
    true_neg = predA[0]+predA[2]+predC[0]+predC[2]

    #TPR and FPR count for B
    tpr = true_pos/actual_per_class[1]
    fpr = false_pos/(actual_per_class[0]+actual_per_class[2])
    accuracy = (true_pos+true_neg)/total
    precision = true_pos/(true_pos+false_pos)
    return tpr, fpr, accuracy, precision

def ratesC(predA, predB, predC, actual_per_class, total):
    #TP and FP and TN count for C
    true_pos =  predC[2]
    false_pos = predA[2]+predB[2]
    true_neg = predA[0]+predA[1]+predB[0]+predB[1]
    #TPR and FPR count for C
    tpr = true_pos/actual_per_class[2]
    fpr = false_pos/(actual_per_class[0]+actual_per_class[1])
    accuracy = (true_pos+true_neg)/total
    precision = true_pos/(true_pos+false_pos)
    return tpr, fpr, accuracy, precision

"""
#Decision boindary is a plane orthonal to the vector 
#between two centroids and passing through the midpoint
#of
Defining the Decision boundary and counting number of test 
predicted to be within the boundary
a(x-x0)+b(y-y0)+c(z-z0)=0 : plane eqn
a,b,c are vector elements
x0,y0,z0 are midpoint coords 
x,y,z are test data values
 the line segment which connects the two centroids
"""
def dec_boundary(testing_input, vectAC, vectAB, vectBC, midAC, midAB, midBC, actual_per_class, dim):
    predA = np.zeros(3)
    predB = np.zeros(3)
    predC = np.zeros(3)
    n = actual_per_class
    for i in range(1,n[0]+1):  #Ranges from data[1:numA+1] 1 through end of A's
        if(ab_bound(vectAB, midAB, dim, testing_input[i])<=0):
            if(ac_bound(vectAC, midAC, dim, testing_input[i])<=0):
                predA[0] = predA[0]+1
            else:
                predA[2] = predA[2]+1
        elif(bc_bound(vectBC, midBC, dim, testing_input[i])<=0):
            predA[1] = predA[1]+1
        else:
            predA[2] = predA[2]+1

    for i in range(n[0]+1,n[0]+n[1]+1): #Ranges from data[NumA+1:NumA+NumB+1]
        if(ab_bound(vectAB, midAB, dim, testing_input[i])<=0):
            if(ac_bound(vectAC, midAC, dim, testing_input[i])<=0):
                predB[0] = predB[0]+1
            else:
                predB[2] = predB[2]+1
        elif(bc_bound(vectBC, midBC, dim, testing_input[i])<=0):
            predB[1] = predB[1]+1
        else:
            predB[2] = predB[2]+1

    for i in range(n[0]+n[1]+1,n[0]+n[1]+n[2]+1): #Ranges from data[NumA+NumB+1:NumA+NumB+NumC+1]       
        if(ab_bound(vectAB, midAB, dim, testing_input[i])<=0):
            if(ac_bound(vectAC, midAC, dim, testing_input[i])<=0):
                predC[0] = predC[0]+1
            else:
                predC[2] = predC[2]+1
        elif(bc_bound(vectBC, midBC, dim, testing_input[i])<=0):
            predC[1] = predC[1]+1
        else:
            predC[2] = predC[2]+1

    return predA,predB,predC

def ab_bound(vectAB, midAB, dim, data):
    val = 0
    for i in range(0,dim):
        val=val+vectAB[i]*(data[i]-midAB[i])
    return val

def bc_bound(vectBC, midBC, dim, data):
    val = 0
    for i in range(0,dim):
        val=val+vectBC[i]*(data[i]-midBC[i])
    return val

def ac_bound(vectAC, midAC, dim, data):
    val = 0
    for i in range(0,dim):
        val=val+vectAC[i]*(data[i]-midAC[i])
    return val


"""
Calculate the vector direction to establish boundary
"""
def calc_vect(centX, centY, dim):
    vect = np.zeros(dim)
    for i in range(0, dim):
        vect[i] = centY[i]-centX[i]
    return vect

"""
Calculate the midpoint between two centroids
"""
def calc_mid(centX, centY, dim):
    midpoint = np.zeros(dim)
    for i in range(0, dim):
        midpoint[i] = (centX[i]+centY[i])/2
    return midpoint    
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
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
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

