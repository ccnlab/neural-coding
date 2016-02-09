classdef ICAModel < FeatureModel
    %Feature model provides identity mapping
    
    properties
        
        W % filters 
        
        feature_size = [16 16 12 16];
        patch_number = 1000; %50000;
        
        static_nonlinearity = @(input_arg) log(1 + abs(input_arg));
        
        patch
        
    end
    
    methods
        
        function obj = ICAModel(varargin)
            
            obj = obj@FeatureModel(varargin{:});
            
        end
        
        function obj = fit(obj,X)            
            
            % required permutation (can be optimized)
            X = permute(X,[2 3 1]);
            
            stim_size = size(X);
            for index = obj.patch_number : -1 : 1
                obj.patch(:, index) = subsref(X((1 : obj.feature_size(1)) + randi(stim_size(1) - obj.feature_size(1)), (1 : obj.feature_size(2)) + randi(stim_size(2)  - obj.feature_size(2)), randi(stim_size(3))), substruct('()', {':'}));
            end
            
            [~, obj.W] = fastica(obj.patch, 'approach', 'symm', 'g', 'tanh', 'lastEig', obj.feature_size(3) * obj.feature_size(4));
            
        end
        
        function Y = predict(obj,X)
            
            % required permutation (can be optimized)
            X = permute(X,[2 3 1]);
      
            stim_size = size(X);
            nfeatures = size(obj.W,1) * size(X,1)*size(X,2) / prod(obj.feature_size([1 2]));
            Y = zeros(stim_size(3),nfeatures);
            
            % this function here is shared by PCA and ICA
            % overload in subclass for TICA
            for index = stim_size(3) : -1 : 1
                Y(index,:) = subsref(obj.static_nonlinearity(obj.W * im2col(X(:, :, index), obj.feature_size([1 2]), 'distinct')), substruct('()', {':'}))';
            end
             
        end
        
    end
    
end
