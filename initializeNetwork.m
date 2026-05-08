function model = initializeNetwork(layerSizes)

% Wejście:
%   layerSizes - wektor rozmiarów warstw, np. [numFeatures, 64, 32, numClasses]
%
% Wyjście:
%   model - struktura zawierająca wagi, biasy i bufory AdamW

    numLayers = length(layerSizes) - 1;

    model.numLayers = numLayers;
    model.layerSizes = layerSizes;

    for l = 1:numLayers

        fanIn = layerSizes(l);
        fanOut = layerSizes(l+1);

        % He initialization dobra dla ReLU / LeakyReLU
        model.W{l} = randn(fanOut, fanIn) * sqrt(2 / fanIn);

        % Biasy startują od zera
        model.b{l} = zeros(fanOut, 1);

        % Bufory AdamW dla wag
        model.mW{l} = zeros(size(model.W{l}));
        model.vW{l} = zeros(size(model.W{l}));

        % Bufory AdamW dla biasów
        model.mb{l} = zeros(size(model.b{l}));
        model.vb{l} = zeros(size(model.b{l}));

    end

end