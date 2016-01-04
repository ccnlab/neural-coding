classdef GaussianProcessRegression < ResponseModel
    
    properties
             
        % normalization parameters
        mux, sdx, muy, sdy
        
        tau; % prior precisions
        sigma; % noise SDs
        
        X % training stimuli
        L % cholesky decomposition
        alpha % L' \ (L \ Y);
        
        evidence % marginal likelihoods for each output

    end
    
    methods
        
        function obj = GaussianProcessRegression(varargin)
            
            obj = obj@ResponseModel(varargin{:});
            
        end
        
        function obj = fit(obj,X,Y)            
            
            % normalize (required for Y given zero mean GP; useful in
            % general when setting hyper-parameters; take with LOO settings)
            [X,obj.mux,obj.sdx] = zscore(X);
            [Y,obj.muy,obj.sdy] = zscore(Y);
            
            % grid search in log domain
            xtau = linspace(0.1,10,20);
            xsigma = linspace(0.1,10,15);
            
            noutputs = size(Y,2);
            
            egrid = -(size(X,1)/2)*log(2*pi)*ones(numel(xtau),numel(xsigma),noutputs);                  
            for i=1:numel(xtau)
                
                K = (1/xtau(i))*(X*X');
                
                for j=1:numel(xsigma)
                    
                    L = chol(K + xsigma(j)^2*eye(size(K)));
                    A = L' \ (L \ Y);
                    
                    egrid(i,j,:) = bsxfun(@minus,egrid(i,j,:),diag(Y'*A));
                    egrid(i,j,:) = bsxfun(@minus,egrid(i,j,:),sum(log(diag(L))));    
                    
                end
                
            end
            
%             for k=1:noutputs
%                 [U,V] = meshgrid(xsigma,xtau);
%                 surf(U,V,obj.grid(:,:,k));
%                 xlabel('sigma');
%                 ylabel('tau');
%                 %imagesc(obj.grid(:,:,k));
%                 pause
%             end

            % get best tau and sigma + associated evidence
            obj.tau = zeros(1,noutputs);
            obj.sigma = zeros(1,noutputs);
            obj.evidence = zeros(1,noutputs);
            for k=1:noutputs
                g = egrid(:,:,k);
                [obj.evidence(k),b] = max(g(:));
                [obj.tau(k),obj.sigma(k)] = ind2sub(size(egrid),b);
            end
            
            % store training data; very inefficient!
            obj.X = X;
            
            K = (1/obj.tau)*(X*X');
            obj.L = chol(K + obj.sigma^2*eye(size(K)));
            obj.alpha = obj.L' \ (obj.L \ Y);
 
        end
        
        function [EY,VarY] = predict(obj,X)
            
            Ks = (1/obj.tau)*X*obj.X';
                            
            EY = zeros(size(X,1),size(obj.alpha,2));
            VarY = zeros(size(X,1),1);
            for i=1:size(X,1)
                
                clc
                disp(i)
                
                ks = (1/obj.tau)*X(i,:)*X(i,:)';

                v = obj.L \ Ks(i,:)';
                
                EY(i,:) = Ks(i,:)*obj.alpha;
                VarY(i) = ks - v'*v + obj.sigma^2;
                            
            end
            
            % rescale Y
            EY = bsxfun(@plus,EY,obj.muy);
            
        end
             
    end
end