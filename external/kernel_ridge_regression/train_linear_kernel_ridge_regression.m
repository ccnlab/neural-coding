function [BETA_hat, H_0, MU, SIGMA, X_hat, lambda_hat, r_hat] = train_linear_kernel_ridge_regression(X, Y, alpha, k, n)
%TRAIN_LINEAR_KERNEL_RIDGE_REGRESSION Train linear kernel ridge regression.
%
%   Inputs
%
%   X    : cell array of # observations x # dimensions matrix of
%          regressors (training set).
%   Y    : # observations x # dimensions matrix of regressands (training
%          set).
%   alpha: critical value (in statistical hypothesis testing).
%   k    : number of folds (in cross-validation).
%   n    : number of effective degrees of freedoms / lambdas (in
%          hyperparameter optimization).
%
%   Outputs
%
%   BETA_hat  : parameter vectors.
%   H_0       : statistical hypothesis testing results (p-values > alpha).
%   MU        : means of X.
%   SIGMA     : standard deviations of X.
%   X_hat     : indices of optimal X.
%   lambda_hat: optimal lambdas.
%   r_hat     : optimal rs.
%
%   Example
%
%   close all;
%   clear all;
%   clc;
%
%   X     = {randn(100, 10); randn(100, 100); randn(100, 200)};
%   Y     = randn(100, 100);
%   alpha = 2.5e-4;
%   k     = 5;
%   n     = 10;
%
%   [BETA_hat, H_0, MU, SIGMA, X_hat, lambda_hat, r_hat] =
%   train_linear_kernel_ridge_regression(X, Y, alpha, k, n);
%
%   Y_hat = test_linear_kernel_ridge_regression(BETA_hat, MU, SIGMA, X,
%   X_hat);
%
%   See also TEST_LINEAR_KERNEL_RIDGE_REGRESSION.
%
%   Umut Güçlü

L      = length(X);
MU     = cell(L, 1);
SIGMA  = cell(L, 1);
K      = cell(L, 1);
LAMBDA = NaN(n, L, 'single');
d      = size(Y);
R      = NaN(d(2), n, L, 'single');

for index = 1 : L
    
    fprintf('TRAIN_LINEAR_KERNEL_RIDGE_REGRESSION (a / c): %d / %d\n', index, L);
    
    [X{index}, MU{index}, SIGMA{index}] = zscore(X{index});
    
    K{index} = X{index} * X{index}';
    
    [R(:, :, index), LAMBDA(:, index)] = get_R_and_lambda(K{index}, Y, k, n);
    
end

r_hat      = NaN(d(2), 1, 'single');
lambda_hat = NaN(d(2), 1, 'single');
X_hat      = NaN(d(2), 1, 'single');

for index = 1 : d(2)
    
    fprintf('TRAIN_LINEAR_KERNEL_RIDGE_REGRESSION (b / c): %d / %d\n', index, d(2));
    
    [r_hat(index), I] = max(subsref(R(index, :, :), substruct('()', {':'})));
    [I, J]            = ind2sub([n, L], I);
    
    lambda_hat(index) = LAMBDA(I, J);
    X_hat(index)      = J;
    
end

BETA_hat = cell(d(2), 1);
H_0      = 1 - tcdf(double(r_hat) .* sqrt((d(1) - 2) ./ (1 - double(r_hat) .^ 2)), d(1) - 2) >= alpha;

X_hat(H_0)      = NaN;
lambda_hat(H_0) = NaN;

for index = 1 : L
    
    fprintf('TRAIN_LINEAR_KERNEL_RIDGE_REGRESSION (c / c): %d / %d\n', index, L);
    
    BETA_hat(X_hat == index) = subsref(get_BETA_hat(K{index}, X{index}, Y(:, X_hat == index), lambda_hat(X_hat == index)), substruct('()', {':'}));
    
end

end

function [R, lambda] = get_R_and_lambda(K, Y, k, n)

d       = size(Y);
Indices = sort(crossvalind('Kfold', d(1), k));
lambda  = single(get_lambda(double(K), n));
Y_hat   = NaN(d(1), d(2), n, 'single');

for index_1 = 1 : k
    
    Train = index_1 ~= Indices;
    S     = sum(Train);
    N     = NaN(S, d(2), n, 'single');
    I     = eye(S, 'single');
    Test  = index_1 == Indices;
    
    foo = K(Train, Train);
    bar = Y(Train, :);
    
    parfor index_2 = 1 : n
        
        N(:, :, index_2) = (foo + lambda(index_2) * I) \ bar;
        
    end
    
    Y_hat(Test, :, :) = reshape(K(Test, Train) * reshape(N, S, d(2) * n), sum(Test), d(2), n);
    
end

R = NaN(d(2), n, 'single');

for index = 1 : n
    
    C_1 = bsxfun(@minus, Y, mean(Y));
    C_2 = bsxfun(@minus, Y_hat(:, :, index), mean(Y_hat(:, :, index)));
    
    R(:, index) = sum(C_1 .* C_2) ./ (sqrt(sum(C_1 .^ 2)) .* sqrt(sum(C_2 .^ 2)));
    
end

end

function lambda = get_lambda(K, n)

s      = svd(K);
s      = s(s > 0);
lambda = NaN(1, n);
L      = length(s);
df     = linspace(L, 1, n);
M      = mean(1 ./ s);

f       = @(df, lambda) df - sum(s ./ (s + lambda)     );
f_prime = @(    lambda)      sum(s ./ (s + lambda) .^ 2);

for index = 1 : n
    
    if index == 1
        
        lambda(index) = 0;
        
    else
        
        lambda(index) = lambda(index - 1);
        
    end
    
    lambda(index) = max(lambda(index), (L / df(index) - 1) / M);
    
    temp = f(df(index), lambda(index));
    
    tic;
    
    while abs(temp) > 1e-10
        
        lambda(index) = max(0, lambda(index) - temp / f_prime(lambda(index)));
        
        temp = f(df(index), lambda(index));
        
        if toc > 1
            
            warning('GET_LAMBDA did not converge.')
            
            break;
            
        end
        
    end
    
end

end

function BETA_hat = get_BETA_hat(K, X, Y, lambda_hat)

C        = unique(lambda_hat);
d        = size(Y);
BETA_hat = NaN(d(1), d(2), 'single');
I        = eye(d(1), 'single');

for index = 1 : length(C)
    
    BETA_hat(:, C(index) == lambda_hat) = (K + C(index) * I) \ Y(:, C(index) == lambda_hat);
    
end

BETA_hat = num2cell(X' * BETA_hat, 1);

end

