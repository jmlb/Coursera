# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 08:14:04 2016

Code source: Jean-Marc Beaujour (jmlbeaujour@gmail.com)
"""

import pandas as pd
import json
import numpy as np
from sklearn import tree
import subprocess
# from PIL import Image
# im = Image.open("dt.png")
# im.show()
dataFile = r'E:/DataScientist/myNotebook/ML_classification (Uni.Washington)/assig3a/lending-club-data.csv'
#products = pd.read_table(dataFile, sep=' ', header=None, names=colNames) # user_id gender  age  occupation    zip
loans = pd.read_csv(dataFile, header=0, low_memory=False) 

trainIdxFile = r'E:/DataScientist/myNotebook/ML_classification (Uni.Washington)/assig3a/module-5-assignment-1-train-idx.json'
validationIdxFile = r'E:/DataScientist/myNotebook/ML_classification (Uni.Washington)/assig3a/module-5-assignment-1-validation-idx.json'
train_idx = json.load(open(trainIdxFile)) #open the json file as a string and parse it with json.load ==> a list
validation_idx = json.load(open(validationIdxFile))

#print loans.columns.values #print the names of the columns
#Exploring some features
# column = 'bad_loans' is the name of the target column, where 1=risky(bad), 0 =means safe
#We will reassign the target such that +1 as a safe loan -1 bad loan
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: 1 if x==0 else -1)
#remove the 'bad_loans' column
loans = loans.drop('bad_loans', 1) #df.drop('column_name', axis=1, inplace=True)
print 'ok'
#Let's explore the distribution of the column safe_loans. This gives a sense of how many and risky loans are present in the dataset
#Print out the percentage of safe loan and risky loan
nbr_safe_loans = sum(loans['safe_loans']==1)
nbr_risky_loans =abs(sum(loans['safe_loans']==-1))
print 'Percentage of safe loan is', 100.0*(nbr_safe_loans)/(nbr_safe_loans+nbr_risky_loans), '| risky loan', 100.0*(nbr_risky_loans)/(nbr_safe_loans+nbr_risky_loans) 
#Extract the features of interest
features = ['grade',                    # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',         # has borrower had a delinquincy
            'last_major_derog_none',    # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]
target = 'safe_loans'
#Extract the feature columns and target columns only
loans = loans[features + [target]]
#Apply one-hot encoding to loans : Hot encoding
#This is because some columns values are letters: C, C4 etc, and sci-kit-learn's decision tree implementattion requires only numerical values for its data matrix. This means you will have to turn categorical variables into binary features via 
#one-hot encoding
for colName in features:
    if loans[colName].dtype == object :
        # Create a set of dummy variables from the sex variable
        dummies = pd.get_dummies(loans[colName])
        # Join the dummy variables to the main dataframe
        loans = loans.join(dummies)
        loans = loans.drop(colName, 1)

#Perform train/validation split using train_idx and validation_idx
train_data = loans.iloc[train_idx]
validation_data = loans.iloc[validation_idx]
print train_data.shape

#Check that the split is 80-20
train_exples = 1.0*train_data.shape[0]
validation_exples = 1.0*validation_data.shape[0]
print 'Split of training/validation examples (percent):', 100*train_exples/(train_exples+validation_exples) , '|' , 100*validation_exples/(train_exples+validation_exples)
#check class balance: i.e 50% safe loans, 50% bad loans in trainign sets
#some examples were left out of the initial data to achieve class balance. In this case, class balance was obtained by kicking out some examples to get to 50% 
Nbr_of_risky = -1.0*sum(train_data[train_data[target]==-1][target])
Nbr_of_safe = sum(train_data[train_data[target]==1][target])
print 'Percentage of Risky Loans in dataset:',Nbr_of_risky/(Nbr_of_risky + Nbr_of_safe)*100,'|Safe Loans:', Nbr_of_safe/(Nbr_of_risky + Nbr_of_safe)*100
#train_data_matrix = np.array(train_data)
#As we explored above, our data is disproportionally full of safe loans. 
#Let's create two datasets: one with just the safe loans (safe_loans_raw) 
#and one with just the risky loans (risky_loans_raw).
#One way to combat class imbalance is to undersample the larger class until the class distribution is approximately half and half. 
#Here, we will undersample the larger class (safe loans) in order to balance out our dataset. 
#This means we are throwing away many data points. We used seed=1 so everyone gets the same results.

# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
# Append the risky_loans with the downsampled version of safe_loans
#You can verify now that loans_data is comprised of approximately 50% safe loans and 50% risky loans.
#Note: There are many approaches for dealing with imbalanced data, 
#including some where we modify the learning algorithm. These approaches are beyond the scope of this course, 
#but some of them are reviewed in this paper. For this assignment, we use the simplest possible approach, 
#where we subsample the overly represented class to get a more balanced dataset. 
#In general, and especially when the data is highly imbalanced, we recommend using more advanced methods.


#use the built-in scikit-learn decision tree learner (sklearn) to create a prediction model on the training data
#Convert the training dat and validation data into numpy array
train_data_target = np.asarray(train_data[target])
train_data_matrix = np.asarray(train_data.drop(target,1))
validation_data_target = np.asarray(validation_data[target])
validation_data_matrix = np.asarray(validation_data.drop(target,1))

decision_tree_model = tree.DecisionTreeClassifier(max_depth=6)
small_model = tree.DecisionTreeClassifier(max_depth=2)
all_featuresName = (train_data.drop(target,1)).columns
decision_tree_model = decision_tree_model.fit(train_data_matrix, train_data_target)
small_model = small_model.fit(train_data_matrix, train_data_target)

#Vizualization of the learned model
#GraphViz + sci-kit learn
dotfile = open(r'E:/DataScientist/myNotebook/ML_classification (Uni.Washington)/assig3a/dtree2.dot', 'w')
f = tree.export_graphviz(small_model, out_file = dotfile, feature_names = all_featuresName, class_names =['risky', 'safe'], filled=True, rounded=True, special_characters=True) 
dotfile.close()
pngFile =r'E:/DataScientist/myNotebook/ML_classification (Uni.Washington)/assig3a/dtree2.png'
dFile = r'E:/DataScientist/myNotebook/ML_classification (Uni.Washington)/assig3a/dtree2.dot'
dotExe = r'C:\Program Files (x86)\Graphviz2.38\bin\dot.exe'
subprocess.call([dotExe, '-Tpng', dFile, '-o', pngFile])
# ipython notebookImage(graph.create_png())  

#Making predictions
#let's consider 2 positive and 2 negative examples from the validation set and see what the model predicts. We will do the following:
#Predict whether or not a loan is safe
#Predict the probability that a loan is safe

#Let's take 2 positive examples and 2 negative examples
sample_validation_safe_loans = validation_data[validation_data[target]==1].iloc[0:2]
#loc works on labels in the index.
#iloc works on the positions in the index (so it only takes integers).
#ix usually tries to behave like loc but falls back to behaving like iloc if the label is not in the index.
sample_validation_risky_loans = validation_data[validation_data[target]==-1].iloc[0:2]
sample_validation_data = sample_validation_safe_loans.append(sample_validation_risky_loans)
sample_validation_data_noLabel = sample_validation_data.drop(target,1)
#Now, we will use our model to predict whether or not a loan is likely to default.
#For each row in the sample_validation_data, use the decision_tree_model to predict 
#whether or not the loan is classified as a safe loan. 
#(Hint: if you are using scikit-learn, you can use the .predict() method)
prediction_smallValidationData = decision_tree_model.predict(sample_validation_data_noLabel)
result_smallValidation = (prediction_smallValidationData == np.array(sample_validation_data[target]))

print 'What percentage of the predictions on sample_validation_data did decision_tree_model get correct?', 100.0*sum((result_smallValidation==1).astype(int))/result_smallValidation.shape[0]

#Explore probability predictions
#For each row in the sample_validation_data, what is the probability (according decision_tree_model) of a loan being classified as safe? 
#(Hint: if you are using scikit-learn, you can use the .predict_proba() method)
proba_smallValidationData = decision_tree_model.predict_proba(sample_validation_data_noLabel) 
# thsi a 2D array [-1 1] probability
#DecisionTreeClassifier is capable of both binary (where the labels are [-1, 1]) classification and multiclass (where the labels are [0, ..., K-1]) classification.
print 'what is the probability (according to decision_tree_model) of a loan being classified as safe?', proba_smallValidationData[:,1]
print 'Quiz Question: Which loan has the highest probability of being classified as a safe loan?', np.argmax(proba_smallValidationData[:,1])+1
#+ 1 since np.array indexing starts at 0

#Checkpoint: Can you verify that for all the predictions with probability >= 0.5, the model predicted the label +1?
expected_target_smallValidation = np.vectorize(lambda x: 1 if x>=0.5 else -1)(proba_smallValidationData[:,1])
comparison = (prediction_smallValidationData == expected_target_smallValidation)
print 'Can you verify that for all the predictions with probability >= 0.5, the model predicted the label +1', comparison

#Tricky predictions!
#14. Now, we will explore something pretty interesting. For each row in the sample_validation_data, 
#what is the probability (according to small_model) of a loan being classified as safe?
proba_smallModel_smallValidationData = small_model.predict_proba(sample_validation_data_noLabel)

print 'what is the probability (according to small_model) of a loan being classified as safe?', proba_smallModel_smallValidationData[:,1]
print 'Quiz Question: Notice that the probability preditions are the exact same for the 2nd and 3rd loans. Why would this happen? the features up to depth 2 are the same'
same_percent_samples = sample_validation_data_noLabel
print same_percent_samples

#Visualize the prediction on a tree
#Note that you should be able to look at the small tree (of depth 2), 
#traverse it yourself, and visualize the prediction being made. 
#Consider the following point in the sample_validation_data:

validation_point = sample_validation_data.iloc[1]
print validation_point
print 'Quiz Question:\n' 
print 'Based on the visualized tree, what prediction would you make for this data point (according to small_model)? Risky loan'

#Now, verify your prediction by examining the prediction made using small_model.
#Evaluating accuracy of the decision tree model
#Recall that the accuracy is defined as follows: accuracy = correctly classified examples/total examples
#Evaluate the accuracy of small_model and decision_tree_model on the training data. 
#(Hint: if you are using scikit-learn, you can use the .score() method)
#score Returns the mean accuracy on the given test data and labels.
score_train_decisionTreeModel = decision_tree_model.score(train_data_matrix, train_data_target)
score_train_smallModel = small_model.score(train_data_matrix, train_data_target)
print 'Evaluate the accuracy on the training data of decision_tree_model', score_train_decisionTreeModel,'| and small model ',  score_train_smallModel
#Checkpoint: You should see that the small_model performs worse than the decision_tree_model on the training data.
score_validation_decisionTreeModel = decision_tree_model.score(validation_data_matrix, validation_data_target)
score_validation_smallModel = small_model.score(validation_data_matrix, validation_data_target)
print '\n Evaluate the accuracy on the validation data of decision_tree_model', score_validation_decisionTreeModel,'| and small model ',  score_validation_smallModel

print '**** Quiz Question:****'
print '\nWhat is the accuracy of decision_tree_model on the validation set (rounded to the nearest .01): %.2f?' % score_validation_decisionTreeModel

#Evaluating accuracy of a complex decision tree model
#Here, we will train a large decision tree with max_depth=10. 
#This will allow the learned tree to become very deep, and result in a very complex model. 
#Recall that in lecture, we prefer simpler models with similar predictive power. 
#This will be an example of a more complicated model which has similar predictive power, i.e. something we don't want.
big_model = tree.DecisionTreeClassifier(max_depth=10)
big_model = big_model.fit(train_data_matrix, train_data_target)
big_model_train_score = big_model.score(train_data_matrix, train_data_target)
big_model_validation_score = big_model.score(validation_data_matrix, validation_data_target)
#Evaluate the accuracy of big_model on the training set and validation set.
print '\n Evaluate the accuracy of big_model on: the training set %.2f|validation set: %.2f' % (big_model_train_score, big_model_validation_score)
#Checkpoint: We should see that big_model has even better performance on the training set than decision_tree_model did on the training set.
print '**** Quiz Question:****'
print '\n How does the performance of big_model on the validation set compare to decision_tree_model on the validation set?'
print '\n answer: slightly worst. This is a sign of overfitting, as accuracy increases on training set with big model'


#Quantifying the cost of mistakes
#Every mistake the model makes costs money. In this section, 
#we will try and quantify the cost each mistake made by the model. Assume the following:
#False negatives: Loans that were actually safe but were predicted to be risky. 
#This results in an oppurtunity cost of loosing a loan that would have otherwise been accepted.
#False positives: Loans that were actually risky but were predicted to be safe. 
#These are much more expensive because it results in a risky loan being given.
#Correct predictions: All correct predictions don't typically incur any cost.
#Let's write code that can compute the cost of mistakes made by the model. Complete the following 4 steps:

#First, let us compute the predictions made by the model.
validation_data_predict = decision_tree_model.predict(validation_data_matrix)
print 'size', validation_data_matrix.shape[0]
#Second, compute the number of false positives: i.e loans predicted 1 (safe) but target (-1) risky
validation_predict_target = pd.concat([pd.DataFrame({'predict': validation_data_predict}), pd.DataFrame({'target': validation_data_target})], axis=1)
false_positive = validation_predict_target[validation_predict_target['predict']==1]
count_false_positive = false_positive[false_positive['target']==-1].shape[0]
print 'The number of False positive is:', count_false_positive  
#Third, compute the number of false negatives.
false_negative = validation_predict_target[validation_predict_target['predict']==-1]
count_false_negative = false_negative[false_negative['target']==1].shape[0]
print 'The number of False negative is:', count_false_negative
#Finally, compute the cost of mistakes made by the model by adding up the costs of true positives and false positves.
#Quiz Question: Let's assume that each mistake costs us money: a false negative costs $10,000, while a false positive positive costs $20,000. What is the total cost of mistakes made by decision_tree_model on validation_data?
cost_error = 10000*count_false_negative + 20000*count_false_positive
print '*********** Quiz Question ************'
print '\nThe total cost of mistakes made by decision_tree_model on validation_data?: %.2f' % cost_error
print '\nThe cost of mistakes per application made by decision_tree_model on validation_data?: %.2f' % (cost_error/validation_data_matrix.shape[0])