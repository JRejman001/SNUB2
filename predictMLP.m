function labels = predictMLP(model, X, params)
% predictMLP
% Zwraca przewidywane klasy dla danych X.
%
% Wejście:
%   model  - wytrenowany model
%   X      - dane wejściowe [numFeatures x numSamples]
%   params - parametry sieci
%
% Wyjście:
%   labels - przewidziane klasy [1 x numSamples]

    [Yhat, ~] = forward(model, X, params, false);

    [~, labels] = max(Yhat, [], 1);

end