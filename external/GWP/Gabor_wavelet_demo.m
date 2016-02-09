b      = 1;
FOV    = -64 : 64;
gamma  = 1;
lambda = 32;
phi    = pi / 2;
sigma  = [];
theta  = pi / 4;
x_0    = -16;
y_0    = 16;

[G, output_arg] = Gabor_wavelet(b, FOV, gamma, lambda, phi, sigma, theta, x_0, y_0);

imagesc(G);