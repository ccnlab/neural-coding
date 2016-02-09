function [G, output_arg] = Gabor_wavelet(b, FOV, gamma, lambda, phi, sigma, theta, x_0, y_0)

[X, Y] = meshgrid(FOV);

if ~xor(isempty(b), isempty(sigma)) 
    
    error('~xor(isempty(b), isempty(sigma))');
    
elseif isempty(b)
    
    b = (sigma / lambda * pi + sqrt(log(2) / 2)) / (sigma / lambda * pi - sqrt(log(2) / 2));
    
    output_arg = b;
    
elseif isempty(sigma)
    
    sigma = lambda * 1 / pi * sqrt(log(2) / 2) * (2 ^ b + 1) / (2 ^ b - 1);
    
    output_arg = sigma;
    
end

X_prime =  (X - x_0) * cos(theta) + (Y + y_0) * sin(theta);
Y_prime = -(X - x_0) * sin(theta) + (Y + y_0) * cos(theta);

G = exp(-(X_prime .^ 2 + gamma ^ 2 * Y_prime .^ 2) / (2 * sigma ^ 2)) .* cos(2 * pi * X_prime / lambda + phi);

end

