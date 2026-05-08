function loss = computeLoss(Yhat, Y, model, weightDecay)
% computeLoss
% Liczy funkcję straty cross-entropy dla klasyfikacji wieloklasowej.
%
% Wejście:
%   Yhat        - przewidywania sieci [numClasses x batchSize]
%   Y           - prawdziwe etykiety one-hot [numClasses x batchSize]
%   model       - struktura sieci
%   weightDecay - współczynnik regularizacji
%
% Wyjście:
%   loss        - jedna liczba mówiąca, jak bardzo sieć się myli

    m = size(Y, 2);

    epsVal = 1e-12;

    ceLoss = -sum(sum(Y .* log(Yhat + epsVal))) / m;

    l2 = 0;

    for l = 1:model.numLayers
        l2 = l2 + sum(model.W{l}.^2, 'all');
    end

    loss = ceLoss + 0.5 * weightDecay * l2;

end