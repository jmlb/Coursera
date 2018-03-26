%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('../ex3data1.mat'); % training data stored in arrays X(5000*400), y(5000*1)
m = size(X, 1);
[m n] = size(X);

% Randomly select 100 data points to display
rand_indices = randperm(m); %create a 1D matrix of randomized/permutated Nbrs between 1 and m
sel = X(rand_indices(1:100), :); %Take the first 100 rows of rand_indices and select the examples x^i with matching indices i
%sel must be a 100samples * 400 features (100*400 matrix)
example_width = round(sqrt(size(X, 2))); 

fprintf('%f\n', example_width);
