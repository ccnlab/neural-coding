classdef FeatureModel < handle
    %Abstract feature model
    
    properties
        
        
        
    end
    
    % public methods
    methods
        
        function obj = method(varargin)
            
            % parse options
            for i=1:2:length(varargin)
                if ismember(varargin{i},fieldnames(obj))
                    obj.(varargin{i}) = varargin{i+1};
                else
                    error(sprintf('unrecognized fieldname %s',varargin{i}));
                end
            end
            
        end        
        
    end
    
end


