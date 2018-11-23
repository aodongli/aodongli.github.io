% This is an example for illustrating missing at random (MAR).
% This example comes from Example 3 in the comments of the paper 
% [Rubin, Donald B. "Inference and missing data." Biometrika 63.3 (1976):
% 581-592.]


rng('default');
% Generate data
N = 5000;
mu = [0 0];
Sigma = [1 1; 1 2]; R = chol(Sigma);
X = repmat(mu,N,1) + randn(N,2)*R;

% corrcoef(X)
% ans =
% 
%    1.000000000000000   0.707920397423750
%    0.707920397423750   1.000000000000000

% Plot
scatter(X(:,1), X(:,2))

% MAR: remove yi if xi > 0
I = find(X(:, 1) > 0); 
% NOT MAR: remove yi if yi > 0
%I = find(X(:, 2) > 0); 
X_obs = X(I,:);
M = length(X_obs)

% maximum likelihood estimator
mu1 = sum(X(:,1)) / N
sq_sigma1 = sum((X(:,1)-mu1).^2) / N
p = polyfit(X_obs(:,1), X_obs(:,2), 1);
mu2 = p(2)+p(1)*mu1
sigma12 = p(1)*sq_sigma1
sq_sigma_mis = sum((X_obs(:,2)-p(2)-p(1)*X_obs(:,1)).^2) / M
sq_sigma2 = sq_sigma_mis + (sigma12^2/sq_sigma1)
p_coeff = sigma12/(sqrt(sq_sigma2*sq_sigma1))

% variance is very high when N = 50