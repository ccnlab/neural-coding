classdef GaussianProcessRegression < ResponseModel    
% GP regression where we assume that the data of interest is a tensor whose
% kernel function can be written in Kronecker form
    
    properties
              
        X % training data points
        Y % training responses
        gp % gaussian process

    end
    
    methods
        
        function obj = GaussianProcessRegression(varargin)
            
            obj = obj@ResponseModel(varargin{:});
            
        end
        
        function obj = fit(obj,X,Y)            
            
            % log uniform (uniform prior for log(parameter))
            pl = prior_logunif();
            
            % gaussian likelihood
            lik = lik_gaussian('sigma2', 0.1^2,'sigma2_prior', pl);

            % linear covariance
            gpcf=gpcf_linear('coeffSigma2',1,'coeffSigma2_prior',pl);
            
            % Create GP
            gpt = gp_set('lik', lik, 'cf', gpcf);            
            
            % Set the options for the scaled conjugate optimization
            opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','iter');

            % Optimize with the scaled conjugate gradient method
            nout = size(Y,2);
            gpc = cell(1,nout);
            
            parfor c=1:nout
                gpc{c}=gp_optim(gpt,X(:,:),Y(:,c),'optimf',@fminscg,'opt',opt);
            end
            
            obj.gp=gpc;
            obj.X = X(:,:);
            obj.Y = Y;

        end
        
        function [EY,VarY] = predict(obj,X)
            
            nout = numel(obj.gp);
            gpt = obj.gp;
            Xt = obj.X;
            Yt = obj.Y;
            
            EY=zeros(size(X,1),nout);
            VarY=zeros(size(X,1),nout);
            parfor c=1:nout        
                [EY(:,c), VarY(:,c)] = gp_pred(gpt{c}, Xt, Yt(:,c), X(:,:));
            end        
        end
        
        function e = marginal_likelihood(obj)
           
            [e, edata, eprior] = gp_e(w, gp, x, y, varargin)
            
        end
             
    end
end