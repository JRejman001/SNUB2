clc;
clear;

numFeatures = 200;
numClasses = 2;

layerSizes = [numFeatures, 64, 32, numClasses];

model = initializeNetwork(layerSizes);

disp(size(model.W{1}));
disp(size(model.b{1}));

disp(size(model.W{2}));
disp(size(model.b{2}));

disp(size(model.W{3}));
disp(size(model.b{3}));
