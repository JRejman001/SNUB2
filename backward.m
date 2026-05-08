function grads = backward(model, cache, Y, params)
% backward
% Liczy gradienty dla wszystkich warstw sieci.
%
% Wejście:
%   model  - struktura sieci z wagami i biasami
%   cache  - zapisane wartości z forward propagation
%   Y      - prawdziwe etykiety one-hot [numClasses x batchSize]
%   params - parametry, np. alpha i dropoutRate
%
% Wyjście:
%   grads  - struktura gradientów:
%            grads.dW{l}
%            grads.db{l}

    L = model.numLayers;
    m = size(Y, 2);

    Yhat = cache.A{L+1};

    % Dla softmax + cross-entropy gradient upraszcza się do:
    dZ = Yhat - Y;

    for l = L:-1:1

        Aprev = cache.A{l};

        grads.dW{l} = (dZ * Aprev') / m;
        grads.db{l} = sum(dZ, 2) / m;

        if l > 1

            dAprev = model.W{l}' * dZ;

            if ~isempty(cache.dropoutMask{l-1})
                dAprev = dAprev .* cache.dropoutMask{l-1};
                dAprev = dAprev / (1 - params.dropoutRate);
            end

            dZ = dLeakyRelu(dAprev, cache.Z{l-1}, params.alpha);

        end

    end

end