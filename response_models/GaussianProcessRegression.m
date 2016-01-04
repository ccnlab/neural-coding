classdef GaussianProcessRegression < ResponseModel    
% GP regression where we assume that the data of interest is a tensor whose
% kernel function can be written in Kronecker form
%
% NOTE: output signal Y must have zero mean since the gp has assumed mean zero!

    properties
              
        gp % gaussian process

    end
    
    methods
        
        function obj = GaussianProcessRegression(varargin)
            
            obj = obj@ResponseModel(varargin{:});
            
        end
        
        function obj = fit(obj,X,Y)            
            
            lik = lik_gaussian('sigma2', 0.2^2);
            gpcf = gpcf_sexp('lengthScale', 1.1, 'magnSigma2', 0.2^2);
            
            % Set some priors
            pn = prior_logunif();
            lik = lik_gaussian(lik,'sigma2_prior', pn);
            pl = prior_unif();
            pm = prior_sqrtunif();
            gpcf = gpcf_sexp(gpcf, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
            
            % create gp object
            gpt = gp_set('lik', lik, 'cf', gpcf);
            
            % Set the options for the optimization
            opt=optimset('TolFun',1e-3,'TolX',1e-3);
            
            % Optimize with the scaled conjugate gradient method
            obj.gp=gp_optim(gpt,X,Y,'opt',opt);
          
        end
        
        function [EY,VarY] = predict(obj,X,Xtrain,Ytrain)
                        
            [EY, VarY] = gp_pred(obj.gp, Xtrain, Ytrain, X);
                     
        end        
             
    end
end