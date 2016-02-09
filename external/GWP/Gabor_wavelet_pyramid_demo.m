%%

close all;
clear all;
clc;

%%

b      = 1;
FOV    = -63.5 : 63.5;
gamma  = 1;
lambda = 2 .^ (2 : 7);
sigma  = [];
theta  = 0 : pi / 8 : 7 * pi / 8;

G = Gabor_wavelet_pyramid(b, FOV, gamma, lambda, sigma, theta);

%%

g    = cell(2, 1);
g{1} = reshape(G{1}, size(G{1}, 1) * size(G{1}, 2), size(G{1}, 3));
g{2} = reshape(G{2}, size(g{1}));

%%

X = mat2gray(imread('cameraman.tif'));
X = imresize(X, [length(FOV) length(FOV)]);

%%

x = reshape(X, size(X, 1) * size(X, 2), 1);

%%

% Simple cell responses

s = g{1}' * x; % or s = G{2}' * x;

% Complex cell responses

c = sqrt((g{1}' * x) .^ 2 + (g{2}' * x) .^ 2);
