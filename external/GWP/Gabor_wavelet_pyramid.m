function G = GWP(b, FOV, gamma, lambda, sigma, theta)

numberOfElements_1 = length(FOV);
numberOfElements_2 = length(theta);
numberOfElements_3 = numberOfElements_1 ./ lambda;
G                  = cell(2, 1);
G{1}               = zeros(sum(numberOfElements_2 * numberOfElements_3 .^ 2), numberOfElements_1 * numberOfElements_1);
G{2}               = zeros(size(G{1}));

index_1 = 1;

for index_2 = 1 : length(lambda)
    
    x_0 = linspace(FOV(1), FOV(end), numberOfElements_3(index_2));
    
    for index_3 = 1 : numberOfElements_2
        
        for index_4 = 1 : numberOfElements_3(index_2)
            
            for index_5 = 1 : numberOfElements_3(index_2)
                
                G{1}(index_1, :) = subsref(Gabor_wavelet(b, FOV, gamma, lambda(index_2), 0     , sigma, theta(index_3), x_0(index_4), x_0(index_5)), substruct('()', {':'}));
                G{2}(index_1, :) = subsref(Gabor_wavelet(b, FOV, gamma, lambda(index_2), pi / 2, sigma, theta(index_3), x_0(index_4), x_0(index_5)), substruct('()', {':'}));
                index_1          = index_1 + 1;
                
            end
            
        end
        
    end
    
end


end

