clear ; close all; clc

data = load('ex1data1.txt'); % read comma separated data
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

plot(X, y, 'rx', 'MarkerSize', 10); % Plot the data
ylabel('Profit in $10,000s'); % Set the y-axis label
xlabel('Population of City in 10,000s');
theta = zeros(2, 1); % initialize fitting parameters
X = [ones(m, 1), data(:,1)];
j = computeCost(X,y,theta);
iterations = 1500;
alpha = 0.01;
[theta, J_history] = gradientDescent(X,y,theta,alpha,iterations);

hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-');
legend('Training data', 'Linear regression');
hold off % don't overlay any more plots on this figure