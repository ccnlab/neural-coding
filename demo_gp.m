%% DEMO

% This is a demo of the neural-coding toolbox

%% Add neural-coding toolbox to search path, cleanup and seed random number generator

addpath(genpath(pwd));

close all; clear all; clc;

rng(1); 

%% Load data

Stimuli           = load('Stimuli.mat'); % from the vim-1 data set.
training_stimulus = double(Stimuli.stimTrn);
training_stimulus = training_stimulus(:,:);
test_stimulus     = double(Stimuli.stimVal);
test_stimulus     = test_stimulus(:,:);
clear Stimuli

EstimatedResponses = load('EstimatedResponses.mat'); % from the vim-1 data set.
ROI                = 1;
training_response  = EstimatedResponses.dataTrnS1(ROI == EstimatedResponses.roiS1 & all(isfinite(EstimatedResponses.dataTrnS1), 2) & all(isfinite(EstimatedResponses.dataValS1), 2), :)';
test_response      = EstimatedResponses.dataValS1(ROI == EstimatedResponses.roiS1 & all(isfinite(EstimatedResponses.dataTrnS1), 2) & all(isfinite(EstimatedResponses.dataValS1), 2), :)';
clear EstimatedResponses

%% choose random subset of voxels for demonstration purposes

nvoxels = 3;
prm = randperm(size(training_response,2));
training_response = training_response(:,prm(1:nvoxels));
test_response = test_response(:,prm(1:nvoxels));

%% Define feature model

fm = Identity;

%% Train feature model

fm.fit(training_stimulus);

%% Simulate feature model

training_feature = fm.predict(training_stimulus);
test_feature     = fm.predict(test_stimulus);

%% Define response model

rm = GaussianProcessRegression;
    
%% Train response model

rm.fit(training_feature, training_response);

%% Simulate response model

[EY,VarY] = rm.predict(training_feature);

for i=1:nvoxels

    plot([training_response(:,i) EY(:,i)]);
    pause

end


test_response_hat     = rm.predict(test_feature);

%% Analyze encoding performance

R = diag(corr(test_response, test_response_hat));

disp(['encoding performance: ' num2str(mean(R)) ' (mean R)'])

figure(2); semilogx(sort(R, 'descend')); xlabel('voxel'); ylabel('R'); title('encoding performance');

