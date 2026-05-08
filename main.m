clc;
clear;
close all;

%% =========================
%  SZTUCZNE DANE TESTOWE
% =========================

numFeatures = 100;
numClasses = 13;
numSamples = 300;

X = randn(numFeatures, numSamples);

labels = randi(numClasses, 1, numSamples);

Y = zeros(numClasses, numSamples);

for i = 1:numSamples
    Y(labels(i), i) = 1;
end

%% =========================
%  PODZIAŁ DANYCH
% =========================

dataset.Xtrain = X(:, 1:200);
dataset.Ytrain = Y(:, 1:200);

dataset.Xval = X(:, 201:250);
dataset.Yval = Y(:, 201:250);

dataset.Xtest = X(:, 251:end);
dataset.Ytest = Y(:, 251:end);

dataset.numFeatures = numFeatures;
dataset.numClasses = numClasses;

%% =========================
%  INICJALIZACJA SIECI
% =========================

layerSizes = [
    dataset.numFeatures, ...
    64, ...
    32, ...
    dataset.numClasses
];

model = initializeNetwork(layerSizes);

%% =========================
%  PARAMETRY TRENINGU
% =========================

params.alpha = 0.01;
params.dropoutRate = 0.2;

params.learningRate = 1e-3;
params.beta1 = 0.9;
params.beta2 = 0.999;
params.epsilon = 1e-8;
params.weightDecay = 1e-4;

params.batchSize = 32;
params.numEpochs = 100;
params.patience = 15;

%% =========================
%  TRENING
% =========================

[model, history] = trainMLP(model, dataset, params);

%% =========================
%  EWALUACJA
% =========================

fprintf('\nTRAIN SET:\n');
trainMetrics = evaluateModel(model, dataset.Xtrain, dataset.Ytrain, params);

fprintf('\nVALIDATION SET:\n');
valMetrics = evaluateModel(model, dataset.Xval, dataset.Yval, params);

fprintf('\nTEST SET:\n');
testMetrics = evaluateModel(model, dataset.Xtest, dataset.Ytest, params);

%% =========================
%  WYKRES LOSS
% =========================

figure;

plot(history.trainLoss, 'LineWidth', 1.5);
hold on;
plot(history.valLoss, 'LineWidth', 1.5);

legend('Train loss', 'Validation loss');
xlabel('Epoch');
ylabel('Loss');
title('Training history');
grid on;