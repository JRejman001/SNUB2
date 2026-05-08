function batches = createMiniBatches(X, Y, batchSize)
% createMiniBatches
% Dzieli dane na mini-batche.
%
% Wejście:
%   X         - dane [numFeatures x numSamples]
%   Y         - etykiety one-hot [numClasses x numSamples]
%   batchSize - liczba próbek w jednym batchu
%
% Wyjście:
%   batches   - cell array z batchami:
%               batches{i}.X
%               batches{i}.Y

    m = size(X, 2);

    idx = randperm(m);

    X = X(:, idx);
    Y = Y(:, idx);

    numBatches = ceil(m / batchSize);

    batches = cell(1, numBatches);

    for i = 1:numBatches

        startIdx = (i - 1) * batchSize + 1;
        endIdx = min(i * batchSize, m);

        batches{i}.X = X(:, startIdx:endIdx);
        batches{i}.Y = Y(:, startIdx:endIdx);

    end

end