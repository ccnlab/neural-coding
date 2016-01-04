% Requires GPStuff toolbox

clear classes
clc

%% create data

% nr of data points
nsamples = 1000;

% noise on the observations
noise = 0.5;

% generating signal S (a sine wave) and noisy measurements Y

ncycles = 10;
S = sin(ncycles * 2 * pi * (1:nsamples) ./ nsamples)';
Y = S + noise*randn(size(S));

% Y MUST HAVE ZERO MEAN
Y = zscore(Y);

% input is time
X = (1:nsamples)';

% train on a subset of points
Xtrain = X(1:5:end);
Ytrain = Y(1:5:end);

lik = lik_gaussian('sigma2', 0.2^2);

% add multiple length scales here in case you have multiple inputs
% e.g. 1.1*ones(1,ninputs)
gpcf = gpcf_sexp('lengthScale', 1.1, 'magnSigma2', 0.2^2);

% Set some priors
pn = prior_logunif();
lik = lik_gaussian(lik,'sigma2_prior', pn);
pl = prior_unif();
pm = prior_sqrtunif();
gpcf = gpcf_sexp(gpcf, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% create gp object
gp = gp_set('lik', lik, 'cf', gpcf);

% Optimize with the scaled conjugate gradient method
opt=optimset('TolFun',1e-3,'TolX',1e-3);
gp=gp_optim(gp,Xtrain,Ytrain,'opt',opt);

% predict on the test data X (requires training data a additional input
[EY, VarY] = gp_pred(gp, Xtrain, Ytrain, X);

% plot stuff
close all
shadedErrorBar(X,EY,2*sqrt(VarY));
hold on;
plot(X,EY,'ko');
%plot(X,S,'b');
plot(Xtrain,Ytrain,'k+');
title('+ = observations, o = predictions');
%legend('uncertainty','predicted signal','','','observations');

