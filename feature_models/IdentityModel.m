classdef Identity < FeatureModel
    %Feature model provides identity mapping
    
    properties
        
        
    end
    
    methods
        
        function obj = Identity(varargin)
            
            obj = obj@FeatureModel(varargin{:});
            
        end
        
        function obj = fit(obj,X)            
            
        end
        
        function Y = predict(obj,X)
            
            Y = X(:,:);
            
        end
        
    end
    
end
