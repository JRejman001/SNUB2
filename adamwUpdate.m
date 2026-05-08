function model = adamwUpdate(model, grads, params, t)
% adamwUpdate
% Aktualizuje wagi i biasy modelu optymalizatorem AdamW.
%
% Wejście:
%   model  - struktura sieci
%   grads  - gradienty z backward.m
%   params - hiperparametry optymalizatora
%   t      - numer kroku treningowego
%
% Wyjście:
%   model  - model po aktualizacji wag i biasów

    lr = params.learningRate;
    beta1 = params.beta1;
    beta2 = params.beta2;
    epsVal = params.epsilon;
    wd = params.weightDecay;

    for l = 1:model.numLayers

        % =====================
        % AKTUALIZACJA WAG W
        % =====================
        
        %Aktualizacja Średniego gradientu
        model.mW{l} = beta1 * model.mW{l} + (1 - beta1) * grads.dW{l};
        model.vW{l} = beta2 * model.vW{l} + (1 - beta2) * (grads.dW{l}.^2);
    
        mHatW = model.mW{l} / (1 - beta1^t);
        vHatW = model.vW{l} / (1 - beta2^t);

        model.W{l} = model.W{l} - lr * mHatW ./ (sqrt(vHatW) + epsVal) - lr * wd * model.W{l};

        % =====================
        % AKTUALIZACJA BIASÓW b
        % =====================

        model.mb{l} = beta1 * model.mb{l} + (1 - beta1) * grads.db{l};
        model.vb{l} = beta2 * model.vb{l} + (1 - beta2) * (grads.db{l}.^2);

        mHatb = model.mb{l} / (1 - beta1^t);
        vHatb = model.vb{l} / (1 - beta2^t);

        model.b{l} = model.b{l} - lr * mHatb ./ (sqrt(vHatb) + epsVal);

    end

end