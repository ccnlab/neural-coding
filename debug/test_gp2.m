clear classes
clc

%% create data

% nr of data points
nsamples = 100; 

% noise on the observations
noise = 0.1;

% generating signal S (a sine wave) and noisy measurements Y

ncycles = 10; 
S = sin(ncycles * 2 * pi * (1:nsamples) ./ nsamples)';
Y = S + noise*randn(size(S));

% Y MUST BE ZERO MEAN
Y = zscore(Y);

% input is time
X = (1:nsamples)';

% train on a subset of points
Xtrain = X(1:3:end);
Ytrain = Y(1:3:end);

rm = GaussianProcessRegression;

rm.fit(Xtrain, Ytrain);

[EY,VarY] = rm.predict(X,Xtrain,Ytrain);

close all
shadedErrorBar(X,EY,2*sqrt(VarY));
hold on;
plot(X,EY,'ko');
%plot(X,S,'b');
plot(Xtrain,Ytrain,'k+');
title('+ = observations, o = predictions');
%legend('uncertainty','predicted signal','','','observations');

