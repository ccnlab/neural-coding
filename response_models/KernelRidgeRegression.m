classdef KernelRidgeRegression < ResponseModel
    
    properties
        
        alpha=2.5e-4;   % critical value (in statistical hypothesis testing)
        k=3;            % number of folds for nested cross-validation
        n=10;           % nr of lambda values to evaluate
        
        BETA_hat   % parameter vectors.
        H_0        % statistical hypothesis testing results (p-values > alpha).
        MU         % means of X.
        SIGMA      % standard deviations of X.
        X_hat      % indices of optimal X.
        lambda_hat % optimal lambdas.
        r_hat      % optimal rs.

    end
    
    methods
        
        function obj = KernelRidgeRegression(varargin)
            
            obj = obj@ResponseModel(varargin{:});
            
        end
        
        function obj = fit(obj,X,Y)            
            
            [obj.BETA_hat, obj.H_0, obj.MU, obj.SIGMA, obj.X_hat, obj.lambda_hat, obj.r_hat] = train_linear_kernel_ridge_regression({X}, Y, obj.alpha, obj.k, obj.n);
            
        end
        
        function Y = predict(obj,X)
            
            Y = test_linear_kernel_ridge_regression(obj.BETA_hat, obj.MU, obj.SIGMA, {X}, obj.X_hat);
            
        end
             
    end
end