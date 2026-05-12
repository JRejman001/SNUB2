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

    disp('Macierz pomyłek:');
    disp(C);

    metrics.accuracy = accuracy;
    metrics.confusionMatrix = C;

    fprintf('Dokładność klasyfikacji: %.2f%%\n', accuracy * 100);    
    
    figure;
    imagesc(C);
    colorbar;
    title('Macierz pomyłek');
    xlabel('Klasa przewidziana');
    ylabel('Klasa rzeczywista');
    axis equal tight;

    for i = 1:numClasses
        TP = C(i,i);
        FP = sum(C(:,i)) - TP;
        FN = sum(C(i,:)) - TP;
        TN = sum(C(:)) - TP - FP - FN;
        
        specificity = TN / (TN + FP);
        sensitivity = TP / (TP + FN);
        
        metrics.specificity(i) = specificity;
        metrics.sensitivity(i) = sensitivity;
    end

    figure;
    subplot(1,2,1);
    bar(metrics.specificity);
    title('Specyficzność dla poszczególnych klas');
    xlabel('Klasa');
    ylabel('Specyficzność');
    
    subplot(1,2,2);
    bar(metrics.sensitivity);
    title('Czułość dla poszczególnych klas');
    xlabel('Klasa');
    ylabel('Czułość');

end