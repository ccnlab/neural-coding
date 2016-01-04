function Y_hat = test_linear_kernel_ridge_regression(BETA_hat, MU, SIGMA, X, X_hat)
%TEST_LINEAR_KERNEL_RIDGE_REGRESSION Test linear kernel ridge regression.
%
%   Inputs
%
%   BETA_hat: parameter vectors (output of
%             TRAIN_LINEAR_KERNEL_RIDGE_REGRESSION).
%   MU      : means of X (output of TRAIN_LINEAR_KERNEL_RIDGE_REGRESSION).
%   SIGMA   : standard deviations of X (output of
%             TRAIN_LINEAR_KERNEL_RIDGE_REGRESSION).
%   X       : cell array of # observations x # dimensions matrix of
%             regressors (test set).
%   X_hat   : indices of optimal X (output of
%             TRAIN_LINEAR_KERNEL_RIDGE_REGRESSION).
%
%   Output
%
%   Y_hat: predictions.
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
%   See also TRAIN_LINEAR_KERNEL_RIDGE_REGRESSION.
%
%   Umut Güçlü

L     = length(X);
Y_hat = NaN(size(X{1}, 1), length(BETA_hat), 'single');

for index = 1 : L
    
    fprintf('TEST_LINEAR_KERNEL_RIDGE_REGRESSION: %d / %d\n', index, L);
    
    if sum(X_hat == index) > 0
        
        X{index} = bsxfun(@rdivide, bsxfun(@minus, X{index}, MU{index}), SIGMA{index});
        
        X{index}(~isfinite(X{index})) = 0;
        
        Y_hat(:, X_hat == index) = X{index} * cat(2, BETA_hat{X_hat == index});
        
    end
    
end

end

