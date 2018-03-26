# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 23:21:52 2016

@author: JBeau7153453
"""

import pandas as pd
import json
import numpy as np

def intermediate_node_num_mistakes(labels_in_node):
    #labels_in_node is 'safe_loans' column 
    # Corner case: If labels_in_node is empty, return 0
    if labels_in_node.shape[0] == 0:
        return 0    
    # Count the number of 1's (safe loans)
    count_safe_loans = sum(labels_in_node[labels_in_node == 1])
    # Count the number of -1's (risky loans)
    count_risky_loans = -sum(labels_in_node[labels_in_node == -1])             
    # Return the number of mistakes that the majority classifier makes.
    if count_safe_loans > count_risky_loans:
        Nbr_mistakes = count_risky_loans
    else:
        Nbr_mistakes = count_safe_loans
    return Nbr_mistakes


#Function to pick best feature to split on
#The function best_splitting_feature takes 3 arguments:
#The data
#$The features to consider for splits (a list of strings of column names to consider for splits)
#The name of the target/label column (string)
#The function will loop through the list of possible features, and consider splitting on each of them. It will calculate 
#the classification error of each split and return the feature that had the smallest classification error when split on.
#best_splitting_feature(data, remaining_features)
def best_splitting_feature(data, features_list, target):
    best_feature = None
    best_error = 10
    # Keep track of the best feature 
   # Keep track of the best error so far 
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.
    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(data.shape[0])
    for feature in features_list:
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature]==0][target]
        # The right split will have all data points where the feature value is 1
        right_split = data[data[feature]==1][target]
        
        # Calculate the number of misclassified examples in the left split.

        # Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
        # YOUR CODE HERE
        left_split_mistakes = intermediate_node_num_mistakes(left_split)
        right_split_mistakes = intermediate_node_num_mistakes(right_split)
        # Calculate the number of misclassified examples in the right split.           
        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        ## YOUR CODE HERE
        error =(left_split_mistakes + right_split_mistakes)/num_data_points
        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        ## YOUR CODE HERE
        if error < best_error: 
            best_error = error
            best_feature = feature            
    return best_feature #Return the best feature we found


#10. First, we will write a function that creates a leaf node given a set of target values. Your code should be analogous to  
def create_leaf(target_values):    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True,  
            'prediction': None }   ## YOUR CODE HERE 
   
    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])    

    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = +1     ## YOUR CODE HERE
    else:
        leaf['prediction'] = -1          ## YOUR CODE HERE        

    # Return the leaf node
    return leaf 

#We have provided a function that learns the decision tree recursively and implements 3 stopping conditions:
#Stopping condition 1: All data points in a node are from the same class.
#Stopping condition 2: No more features to split on.
#Additional stopping condition: In addition to the above two stopping conditions covered in lecture, in this 
#assignment we will also consider a stopping condition based on the max_depth of the tree. By not letting the tree grow too deep, 
#we will save computational effort in the learning process.  
  
  
  
#11. Now, we will provide a Python skeleton of the learning algorithm. Note that this code is not complete; it needs to be completed by you if you are using Python. Otherwise, your code should be analogous to
def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
    if  intermediate_node_num_mistakes(target_values) == 0:  #If no mistakes (majority class = all class - no count for minority)
        print "Stopping condition 1 reached."     
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if len(remaining_features) == 0:   ## YOUR CODE HERE
        print "Stopping condition 2 reached."    
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)    
    
    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:  ## YOUR CODE HERE
        print "Reached maximum depth. Stopping for now."
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)

    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    splitting_feature = best_splitting_feature(data, remaining_features, target)
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]      ## YOUR CODE HERE
    remaining_features = np.setdiff1d(remaining_features, splitting_feature)
    print "Split on feature %s. (%s, %s)" % (\
    splitting_feature, len(left_split), len(right_split))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == data.shape[1]:
        print "Creating leaf node."
        return create_leaf(left_split[target])
    if len(right_split) == data.shape[1]:
        print "Creating leaf node."
        return create_leaf(right_split[target])
 
        
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth)        
    ## YOUR CODE HERE
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth)

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}


#Build the tree!
#12. Train a tree model on the train_data. Limit the depth to 6 (max_depth = 6) to make sure the algorithm doesn't 
#run for too long. Call this tree my_decision_tree. Warning: The tree may take 1-2 minutes to learn.
#Making predictions with a decision tree

#13. As discussed in the lecture, we can make predictions from the decision tree with a simple recursive function. 
#Write a function called classify, which takes in a learned tree and a test point x to classify. Include an option annotate that describes the prediction path when set to True. Your code should be analogous to

def classify(tree, x, annotate = True):
       # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction']
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
             print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)

def evaluate_classification_error(tree, data, target):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(my_decision_tree, x, annotate=True), axis=1) #needs axis to apply to each row, per default columns
    # Once you've made the predictions, calculate the classification error and return it
    data['prediction'] = prediction
    data['mistakes'] = data.apply(lambda x : 0 if x['prediction'] == x[target] else 1, axis=1)
    Nbr_errors = sum(data['mistakes'])
    
    return (Nbr_errors*1.0/data.shape[0])
    
#Follow these steps to implement best_splitting_feature:
#Step 1: Loop over each feature in the feature list
#Step 2: Within the loop, split the data into two groups: one group where all of the data has feature value 0 or False (we will call this the left split), and one group where all of the data has feature value 1 or True (we will call this the right split). Make sure the left split corresponds with 0 and the right split corresponds with 1 to ensure your implementation fits with our implementation of the tree building process.
#Step 3: Calculate the number of misclassified examples in both groups of data and use the above formula to compute theclassification error.
#Step 4: If the computed error is smaller than the best error found so far, store this feature and its error.
#Note: Remember that since we are only dealing with binary features, we do not have to consider thresholds for real-valued features. This makes the implementation of this function much easier.
#Your code should be analogous to        
        
        
# from PIL import Image
# im = Image.open("dt.png")
# im.show()

#The goal of this notebook is to implement your own binary decision tree classifier. You will:
#Use SFrames to do some feature engineering.
#Transform categorical variables into binary variables.
#Write a function to compute the number of misclassified examples in an intermediate node.
#Write a function to find the best feature to split on.
#Build a binary decision tree from scratch.
#Make predictions using the decision tree.
#Evaluate the accuracy of the decision tree.
#Visualize the decision at the root node.
#Important Note: In this assignment, we will focus on building decision trees where the data contain only binary (0 or 1) features. This allows us to avoid dealing with:
#Multiple intermediate nodes in a split
#The thresholding issues of real-valued features.

dataFile = r'E:/DataScientist/myNotebook/ML_classification (Uni.Washington)/assig3b/lending-club-data.csv'
#products = pd.read_table(dataFile, sep=' ', header=None, names=colNames) # user_id gender  age  occupation    zip
#1. Load in the LendingClub dataset with the software of your choice.
#2. Like the previous assignment, reassign the labels to have +1 for a safe loan, and -1 for a risky (bad) loan. You should have code analogous to
loans = pd.read_csv(dataFile, header=0, low_memory=False)
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop('bad_loans', 1)
#3. Unlike the previous assignment, we will only be considering these four features:
features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'
#Extract these feature columns from the dataset, and discard the rest of the feature columns.
#If you are NOT using SFrame, download the list of indices for the training and test sets:
loans_target = loans[target]
loans = loans[features]
#loans = loans[loans['emp_length']!= 'n/a']

for colName in features:
    if loans[colName].dtype == object :
        # Create a set of dummy variables from the sex variable
        dummies = pd.get_dummies(loans[colName])
        #update dummies cols name to include initial name reference
        for dummiesName in dummies.columns.values:
            newName = colName + '_' + dummiesName
            dummies.rename(columns={dummiesName: newName}, inplace=True)
        # Join the dummy variables to the main dataframe
        loans = loans.join(dummies)
        loans = loans.drop(colName, 1)       
features = loans.columns.values
print 'How many features?:', len(features)
loans[target] = loans_target      
trainIdxFile = r'E:/DataScientist/myNotebook/ML_classification (Uni.Washington)/assig3b/module-5-assignment-2-train-idx.json'
testIdxFile = r'E:/DataScientist/myNotebook/ML_classification (Uni.Washington)/assig3b/module-5-assignment-2-test-idx.json'
train_idx = json.load(open(trainIdxFile)) #open the json file as a string and parse it with json.load ==> a list
test_idx = json.load(open(testIdxFile))
train_data = loans.iloc[train_idx]
test_data = loans.iloc[test_idx]
print train_data.iloc[0]
#features = c = np.setdiff1d(,b)
#Apply one-hot encoding to loans. Your tool may have a function for one-hot encoding.
#Load the JSON files into the lists train_idx and test_idx.
#Perform train/validation split using train_idx and test_idx. In Pandas, for instance:
#appl
#Perform train/validation split using train_idx and test_idx.
#claass balance already performed/
#Note. Some elements in loans are included neither in train_data nor test_data. This is to perform sampling to achieve class balance.
#Now proceed to the section "Decision tree implementation", skipping three sections below.
#Subsample dataset to make sure classes are balanced
#4. Just as we did in the previous assignment, we will undersample the larger class (safe loans) in order to balance out our dataset. This means we are throwing away many data points. You should have code analogous to

#Note: There are many approaches for dealing with imbalanced data, including some where we modify the learning algorithm. These approaches are beyond the scope 
#of this course, but some of them are reviewed in this paper. For this assignment, we use the simplest possible approach, where we subsample the overly represented class 
#to get a more balanced dataset. In general, and especially when the data is highly imbalanced, we recommend using more advanced methods.
#Transform categorical data into binary features
#In this assignment, we will implement binary decision trees (decision trees for binary features, a specific case of categorical variables taking on two values, e.g., true/false). 
#Since all of our features are currently categorical features, we want to turn them into binary features.
#http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=5128907&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel5%2F69%2F5173046%2F05128907.pdf%3Farnumber%3D5128907

#or instance, the home_ownership feature represents the home ownership status of the loanee, which is either own, mortgage or rent. For example, if a data point has the feature
#
 #  {'home_ownership': 'RENT'}
#we want to turn this into three features:
# { 
#   'home_ownership = OWN'      : 0, 
#   'home_ownership = MORTGAGE' : 0, 
#   'home_ownership = RENT'     : 1
# }
#5. This technique of turning categorical variables into binary variables is called one-hot encoding. Using the software of your choice, perform one-hot encoding on the four features 
#described above. You should now have 25 binary features.
#6. Split the data into a train test split with 80% of the data in the training set and 20% of the data in the test set. Call 
#these datasets train_data and test_data, respectively. Using SFrame, this would look like
print 'Data split train/test %.2f pct| %.2f pct' % (100.0*train_data.shape[0]/(train_data.shape[0]+test_data.shape[0]), 100.0*test_data.shape[0]/(train_data.shape[0]+test_data.shape[0]))

#Decision tree implementation
#In this section, we will implement binary decision trees from scratch. There are several steps involved in building 
#a decision tree. For that reason, we have split the entire assignment into several sections.
#Function to count number of mistakes while predicting majority class
#Recall from the lecture that prediction at an intermediate node works by predicting the majority class for all data points 
#that belong to this node. Now, we will write a function that calculates the number of misclassified examples when predicting the majority class. This will be used 
#to help determine which feature is the best to split on at a given node of the tree.
#Note: Keep in mind that in order to compute the number of mistakes for a majority classifier, we only need the label (y values) of the data points in the node.
#Steps to follow:
#Step 1: Calculate the number of safe loans and risky loans.
#Step 2: Since we are assuming majority class prediction, all the data points that are not in the majority class are considered mistakes.
#Step 3: Return the number of mistakes.
#7. Now, let us write the function intermediate_node_num_mistakes which computes the number of 
#misclassified examples of an intermediate node given the set of labels (y values) of the data points contained in the node. Your code should be analogous to


#Building the tree

#With the above functions implemented correctly, we are now ready to build our decision tree. Each node in the decision tree is represented as a dictionary which contains the following keys and possible values:

#{ 
#   'is_leaf'            : True/False.
#   'prediction'         : Prediction at the leaf node.
#   'left'               : (dictionary corresponding to the left tree).
#   'right'              : (dictionary corresponding to the right tree).
#   'splitting_feature'  : The feature that this node splits on.}

#Build the tree!
#Train a tree model on the train_data. Limit the depth to 6 (max_depth = 6) to make sure the algorithm doesn't run for too long. 
#Call this tree my_decision_tree. Warning: The tree may take 1-2 minutes to learn.

#14. Now, let's consider the first example of the test set and see what my_decision_tree model predicts for this data point.
print '************ train data ************'
my_decision_tree = decision_tree_create(train_data, features, target, current_depth = 0, max_depth = 6)
#print 'Predicted class: %s ' % classify(my_decision_tree, test_data.iloc[0])
#print 'Predicted class:',classify(my_decision_tree, test_data)
#15. Let's add some annotations to our prediction to see what the prediction path was that lead to this predicted class:
#print 'start test_data[0]'
print '*-------------------------*'
print test_data.iloc[0]
print '************ test data 0 ************'
classRes = classify(my_decision_tree, test_data.iloc[0], annotate=True)
print 'Predicted class: %s ' % classRes
print '*************************'
print '************ fulll test data  ************'
error_Test = evaluate_classification_error(my_decision_tree, test_data, 'safe_loans')
print error_Test

#Printing out a decision stump
# As discussed in the lecture, we can print out a single decision stump (printing out the entire tree is left as an exercise to the curious reader). 
#Here we provide Python code to visualize a decision stump. If you are using different software, make sure your code is analogous to:

def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    #split_feature, split_value = split_name.split('.')
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))

#19. Using this function, we can print out the root of our decision tree:

print print_stump(my_decision_tree)
print 'Quiz Question: What is the feature that is used for the split at the root node?'
print '\n term_36month'


#Exploring the intermediate left subtree
#The tree is a recursive dictionary, so we do have access to all the nodes! We can use
#my_decision_tree['left'] to go left
#my_decision_tree['right'] to go right
#20. We can print out the left subtree by running the code
print_stump(my_decision_tree['left'], my_decision_tree['splitting_feature'])
#We can similarly print out the left subtree of the left subtree of the root by running the code

print_stump(my_decision_tree['left']['left']['left'], my_decision_tree['left']['left']['splitting_feature'])
#Quiz question: What is the path of the first 3 feature splits considered along the left-most branch of my_decision_tree?

#Quiz question: What is the path of the first 3 feature splits considered along the right-most branch of my_decision_tree?
