%% Single Variable Regression Using Gradient Descent by Anshuman Das Gupta

%%Predict profits for a food truck

%%Suppose you are the CEO of a restaurant franchise and are considering 
%%different cities for opening a new outlet. The chain already has trucks
%%in various cities and you have data for profits and populations from the cities.

%importing the dataset in which the first column is the population of a city
%and the second column is profit

clear; close all;

data = load('TrainingData.txt');

X = data(:,1);  %copying features to a matrix X
y = data(:,2);  %copying the outcome variable to a vector Y
fprintf('The DataSet is as follows : \n');  %printing the dataset
data

fprintf('Program has been paused. Press enter to continue...\n\n');
pause;

%visualizing the dataset
fprintf('Making a graph for the Dataset : \n \n');
subplot(1,2,1);
plot(X,y,'rx','MarkerSize',10);
xlabel('Population of City in 10,000s');
ylabel('Profit in $10,000s');

fprintf('Program has been paused. Press enter to continue...\n\n');
pause;

%Normalizing the data
mu = mean(X); %calculating the mean
sigma = std(X); %calculating the standard deviation
X  = X .- mu; %subtracting mean from all the features
X = X ./sigma;  %dividing all the features by the standard deviation

%The data is now normalised
fprintf('The normalised DataSet is as follows : \n');
X

fprintf('Program has been paused. Press enter to continue...\n\n');
pause;

%visualizing the dataset
fprintf('Making a graph for the normalised Dataset : \n \n');
subplot(1,2,2);
plot(X,y,'rx','MarkerSize',10);
xlabel('Population of City');
ylabel('Profit');

fprintf('Program has been paused. Press enter to continue...\n\n');
pause;

%Lets take the hypothesis of the form a1+a2x ie a straight line

m = length(y);  
alpha = 0.01;
theta = zeros(2,1);
iterations = 500;

X = [ones(m,1),X];

[theta,allcosts] = gradientDescent(X,y,theta,alpha,iterations);

fprintf('The cost function varies as follows: \n \n ');
figure(2);
plot(1:iterations, allcosts,'-');
xlabel('Number of iterations');
ylabel('Cost');
fprintf('Program has been paused. Press enter to continue...\n\n');
pause;
fprintf('Plotting the hypothesis onto the normalised DataSet \n \n ');
figure(1);
subplot(1,2,2);
hold on;
plot(X(:,2),X*theta,'-');
hold off;

