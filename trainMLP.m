function [model, history] = trainMLP(model, dataset, params)
% trainMLP
% Główna pętla treningowa sieci MLP.
%
% Wejście:
%   model   - struktura sieci
%   dataset - struktura z danymi:
%             dataset.Xtrain, dataset.Ytrain
%             dataset.Xval, dataset.Yval
%   params  - hiperparametry treningu
%
% Wyjście:
%   model   - wytrenowany model
%   history - historia strat treningowych i walidacyjnych

    bestValLoss = inf;
    patienceCounter = 0;
    t = 0;

    history.trainLoss = [];
    history.valLoss = [];

    bestModel = model;

    for epoch = 1:params.numEpochs

        batches = createMiniBatches(dataset.Xtrain, dataset.Ytrain, params.batchSize);

        trainLossSum = 0;

        for i = 1:length(batches)

            Xbatch = batches{i}.X;
            Ybatch = batches{i}.Y;

            t = t + 1;

            [Yhat, cache] = forward(model, Xbatch, params, true);

            loss = computeLoss(Yhat, Ybatch, model, params.weightDecay);

            grads = backward(model, cache, Ybatch, params);

            model = adamwUpdate(model, grads, params, t);

            trainLossSum = trainLossSum + loss;

        end

        trainLoss = trainLossSum / length(batches);

        [YhatVal, ~] = forward(model, dataset.Xval, params, false);

        valLoss = computeLoss(YhatVal, dataset.Yval, model, params.weightDecay);

        history.trainLoss(epoch) = trainLoss;
        history.valLoss(epoch) = valLoss;

        fprintf('Epoch %d/%d | train loss = %.4f | val loss = %.4f\n', ...
            epoch, params.numEpochs, trainLoss, valLoss);

        if valLoss < bestValLoss
            bestValLoss = valLoss;
            bestModel = model;
            patienceCounter = 0;
        else
            patienceCounter = patienceCounter + 1;
        end

        if patienceCounter >= params.patience
            fprintf('Early stopping at epoch %d\n', epoch);
            break;
        end

    end

    model = bestModel;

end