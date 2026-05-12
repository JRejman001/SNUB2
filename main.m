preprocess;
analizeMissingData;
removeRedundantFeatures;
params.minData = 5;
normalizeData;
splitData;

dataset.numFeatures = size(dataset.Xtest, 1);
dataset.numClasses = size(Y_One_Hot, 1);

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
params.dropoutRate = 0.3;

params.learningRate = 1e-3;
params.beta1 = 0.9;
params.beta2 = 0.999;
params.epsilon = 1e-8;
params.weightDecay = 1e-4;

params.batchSize = 32;
params.numEpochs = 100;
params.patience = 10;

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
