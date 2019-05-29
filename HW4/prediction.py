# Starter code for CS 165B HW4

"""
Implement the testing procedure here. 

Inputs:
    Unzip the hw4_test.zip and place the folder named "hw4_test" in the same directory of your "prediction.py" file, your "prediction.py" need to give the following required output.

Outputs:
    A file named "prediction.txt":
         The prediction file must have 10000 lines because the testing dataset has 10000 testing images.
         Each line is an integer prediction label (0 - 9) for the corresponding testing image.
        
    Notes: 
        1. The teaching staff will run your "prediction.py" to obtain your "prediction.txt" after the competition ends.
        2. The output "prediction.txt" must be the same as the final version you submitted to the CodaLab, 
        otherwise you will be given 0 score for your hw4.


**!!!!!!!!!!Important Notes!!!!!!!!!!**
    To open the folder "hw4_test" or load other related files, 
    please use open('./necessary.file') instead of open('some/randomly/local/directory/necessary.file').

    For instance, in the student Jupyter's local computer, he stores the source code like:
    - /Jupyter/Desktop/cs165B/hw4/prediction.py
    - /Jupyter/Desktop/cs165B/hw4/hw4_test
    If he/she use os.chdir('/Jupyter/Desktop/cs165B/hw4/hw4_test'), this will cause an IO error 
    when the teaching staff run his code under other system environments.
    Instead, he should use os.chdir('./hw4_test').


    If you use your local directory, your code will report an IO error when the teaching staff run your code,
    which will cause 0 score for your hw4.
"""
import os
import time
import math
import numpy as np
import scipy
from scipy import sparse
from scipy import linalg
from sklearn import tree
from sklearn.datasets import load_iris
import scipy.sparse.linalg as spla
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from PIL import Image


def parse_training_data():
	file_num = 0
	training_data = np.zeros(shape=10,dtype=object)
	for f in range(0,10):
		folder = f
		DIR = './hw4_train/'+str(folder)+'/'
		# Get num files in folder hw4_train/file$f
		num_files = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
		training_data[f] = parse_img(num_files, DIR, f)
	return training_data

def parse_img(num_files, DIR, f):
	current_clothes = np.zeros(shape=(int(num_files),784),dtype=object)
	for i in range(0,num_files):
	        img = Image.open(DIR+str(f)+'_'+str(i)+'.png')
	        img_arr = np.array(img.getdata(), np.uint8)
	        current_clothes[i,:] = img_arr
	return current_clothes
#inputs training data, outputs training data as single plus label column vector
def parse_for_tree(training_data):
	tree_training_data = 

def run_train_test():


if __name__ == "__main__":
    """
    You can use this to test your code.
    python hw3.py [training file path] [testing file path]
    """
    import sys

    parse_training_data()

    run_train_test(training_file, testing_file)

    training_file.close()
    testing_file.close()
