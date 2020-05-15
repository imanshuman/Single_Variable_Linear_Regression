function [theta,allcosts] = gradientDescent(X,y,theta,alpha,iterations)

%this is a function for gradient descent which returns the theta
%for which the curve is optimal

m = length(y);
allcosts = zeros(m,1);

for iter = 1:iterations
    allcosts(iter) = computeCost(X,y,theta);
    
    A = X * theta;
    A = A - y;
    A = X .* A;
    d = (alpha/m)*sum(A);
    theta = theta - d';
    
    end

end