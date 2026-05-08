function metrics = evaluateModel(model, X, Y, params)
% evaluateModel
% Ocenia jakość modelu na podanych danych.
%
% Wejście:
%   model  - wytrenowany model
%   X      - dane wejściowe [numFeatures x numSamples]
%   Y      - prawdziwe etykiety one-hot [numClasses x numSamples]
%   params - parametry sieci
%
% Wyjście:
%   metrics - struktura z wynikami:
%             metrics.accuracy
%             metrics.confusionMatrix

    predLabels = predictMLP(model, X, params);

    [~, trueLabels] = max(Y, [], 1);

    accuracy = mean(predLabels == trueLabels);

    numClasses = size(Y, 1);

    C = zeros(numClasses, numClasses);

    for i = 1:length(trueLabels)

        trueClass = trueLabels(i);
        predClass = predLabels(i);

        C(trueClass, predClass) = C(trueClass, predClass) + 1;

    end

    metrics.accuracy = accuracy;
    metrics.confusionMatrix = C;

    fprintf('Accuracy: %.2f%%\n', accuracy * 100);
    disp('Confusion matrix:');
    disp(C);

end