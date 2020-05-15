function J = computeCost(X,y,theta)

m = length(y);

J = 0;
A = X * theta;
A = A - y;
s = sum( A .^ 2);
J = s / (2 * m);

end