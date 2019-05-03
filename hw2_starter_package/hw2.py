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

    numA = heading[1]
    numB = heading[2]
    numC = heading[3]

    dim  = heading[0]           #Number of columns of data (num features)
    n = numA+numB+numC          #Number of entries (rows of initial data)
    print(heading)

    
    pass


"""
dim is dimension of data
n is the numX 
data is the training input
clust is class A,B,C
"""
def calc_centroid(dim, n, data, clust):
    if(clust == 'A'):
        for i in range(1,n):



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

