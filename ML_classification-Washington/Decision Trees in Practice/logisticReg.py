# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:02:15 2016

@author: JBeau7153453
"""

import pandas as pd
import string
import json
import numpy as np
def remove_punctuation(text):
    return text.translate(None, string.punctuation)
                
def dataFrame2Array(dataframe, features, label):
    dataframe['constant'] = 1 #add the bias term
    features = ['constant']+['contains_'+x for x in features] #add 'contains_' to all features to match column name #features is a list of string
    feature_matrix = np.array(dataframe[features]) #extract set of columns from dataframe and convert to matrix
    label_array = np.array(dataframe[label]) #create an array with the label (actual value)
    label_array = np.reshape(label_array, (np.shape(label_array)[0],1))
    return (feature_matrix, label_array)

def predict_probability(feature_matrix, coefficients):
    score = np.dot(feature_matrix, coefficients)
    predictions = 1/(1 + np.exp(-score))
    return predictions

def feature_derivative(errors, feature):
    derivative = np.dot(np.transpose(feature), errors)
    return derivative
    
def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment == +1)    
    scores = np.dot(feature_matrix, coefficients)
    lp = np.sum((indicator - 1)*scores -np.log(1. + np.exp(-scores)))
    return lp

def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients)
    for itr in xrange(max_iter):
        predictions = predict_probability(feature_matrix, coefficients)
        indicator = (sentiment == +1)
        errors = indicator - predictions
        for j in xrange(len(coefficients)): #loop over each coefficient
            derivative = feature_derivative(errors, feature_matrix[:,j])
            coefficients[j] = coefficients[j] + step_size * derivative
        
        if itr<=15 or (itr <= 100 and itr%10 == 0) or (itr <= 1000 and itr%100 == 0) or (itr <= 10000 and itr%100 == 0) or itr%10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' %(int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients
  
#Load review dataset in dataFrame. The subset has 4 cols: name of product=name, review='text', rating = int between 0 and 5 and sentiment 1(positive sentiment) or -1(negative sentiment)
dataFile = r'E:/DataScientist/myNotebook/ML_classification (Uni.Washington)/assig2/amazon_baby_subset.csv'
file_important_words = r'E:/DataScientist/myNotebook/ML_classification (Uni.Washington)/assig2/important_words.json'
important_words = json.load(open(file_important_words)) #open the json file as a string and parse it with json.load ==> a list
nbr_features = len(important_words)
print 'nbr of features:', nbr_features
colNames = ['name', 'review', 'rating', 'sentiment']
#products = pd.read_table(dataFile, sep=' ', header=None, names=colNames) # user_id gender  age  occupation    zip
products = pd.read_csv(dataFile, header=0, names=colNames) #[shape=(183531,3)].\n"
#load important_word json file
print products['review'][0]
#Name of the first 10 products
print 'List of the first 10 products:', products['name'][1:11]  #must use iloc to return element at index id (products.iloc[1])

print 'Generate review_clean column'
#for the empty review, fill (n/a)
products = products.fillna({'review':''})
#Apply text cleaning to the data: create a new column with review without punctuations
products['review_clean'] = products['review'].apply(remove_punctuation)

print 'Generate columns w/ count in review of important_words'
#For each word in important_words (193 words), we compute a count for the number fo times the word occurs in the review. We will store this count in a separate column (one for each word)
for word in important_words:
    products['contains_'+word] = products['review_clean'].apply(lambda s: s.split().count(word))
print 'this is prodyucts 0:', products.iloc[0]
#list.count(obj) count occurence of obj in list
#Python supports the creation of anonymous functions (i.e. functions that are not bound to a name) at runtime, using a construct called "lambda".
#Note that the lambda definition does not include a "return" statement -- it always contains an expression which is returned. Also note that you can put a lambda definition anywhere a function is expected, and you don't have to assign it to a variable at all.
#def f (x): return x**2 is equivalent to g = lambda x: x**2

#Number of product reviews that contain the words perfect
print 'Number of reviews with the word perfect :', sum(products['contains_baby']) #pd.sum(dataFrame) doesn ot work

#Generate feature_matrix, sentiment array
dataFrame_arrays = dataFrame2Array(products, important_words, 'sentiment')
feature_matrix = dataFrame_arrays[0] #item #9
Nbr_of_examples = np.shape(feature_matrix)[0]
sentiment = dataFrame_arrays[1]
initial_coefficients = np.zeros((nbr_features+1,1))
step_size = 1e-7
max_iter = 301
#How many features?
print 'The feature_matrix has < ', np.shape(feature_matrix)[1],' > features, including the bias/intercept'
#Convert data frame to a multidimentional array

#Estimate conditional probability with link function
#P(y=+1|x,w) = the probability that sentiment of example 1 is 1 parametrized by  w:
#P(y=+1|x,w) = sigmoid(wT x) = 1/(1+exp(-wT x))

coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter)
score = np.dot(feature_matrix, coefficients)
score[score > 0] = 1
score[score <= 0] = -1
#Number of product reviews that contain the words perfect
comparison = np.zeros((Nbr_of_examples,1))
comparison = (sentiment==score)
print Nbr_of_examples
print sum(comparison==1)
accuracy = float(sum(comparison==1))/Nbr_of_examples*100
print 'What is the accuracy of the model on predictions made above: ', accuracy

#Which words contribute more to the positive& negative 
#create a tuple (word,coefficient_value)
#coefficients = list(coefficients[1:]) #exclude interface
#word_coeffcient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)]
#Convert coefficient numpy array to list
coefficients = coefficients.tolist()[1:]
word_coefficient_tuple = sorted(zip(coefficients, important_words), key=lambda x:x[1], reverse=True)
print 'The 10 words that have the most positive coeffcient values:', word_coefficient_tuple[0:10]

word_coefficient_tuple = sorted(zip(coefficients, important_words), key=lambda x:x[1])
print 'The 10 words that have the most negative coeffcient values:', word_coefficient_tuple[0:10]