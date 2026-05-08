function [Yhat, cache] = forward(model, X, params, isTraining)
% forward
% Wykonuje forward propagation przez całą sieć.
%
% Wejście:
%   model       - struktura sieci
%   X           - dane wejściowe [numFeatures x batchSize]
%   params      - parametry sieci
%   isTraining  - true/false (czy używać dropout)
%
% Wyjście:
%   Yhat        - predykcja sieci
%   cache       - zapisane wartości do backpropagation

    % Na początku aktywacja = wejście
    A = X;

    % Zapisujemy wejście
    cache.A{1} = X;

    % Liczba warstw
    L = model.numLayers;

    % =========================
    % WARSTWY UKRYTE
    % =========================
    for l = 1:L-1

        % Pobranie wag i biasów
        W = model.W{l};
        b = model.b{l};

        % Dense layer
        Z = W * A + b;

        % Aktywacja
        A = leakyRelu(Z, params.alpha);

        % Zapisujemy Z
        cache.Z{l} = Z;

        % =========================
        % DROPOUT
        % =========================
        if isTraining && params.dropoutRate > 0

            % Losowa maska
            mask = rand(size(A)) > params.dropoutRate;

            % Zerowanie części neuronów
            A = A .* mask;

            % Skalowanie
            A = A / (1 - params.dropoutRate);

            % Zapis maski
            cache.dropoutMask{l} = mask;

        else

            cache.dropoutMask{l} = [];

        end

        % Zapis aktywacji
        cache.A{l+1} = A;

    end

    % =========================
    % OSTATNIA WARSTWA
    % =========================

    ZL = model.W{L} * A + model.b{L};

    % Softmax
    Yhat = softmaxStable(ZL);

    % Zapis
    cache.Z{L} = ZL;
    cache.A{L+1} = Yhat;

end