function A = softmaxStable(Z)
% softmaxStable
% Zamienia logity Z na prawdopodobieństwa klas.
%
% Wejście:
%   Z - macierz [numClasses x batchSize]
%
% Wyjście:
%   A - macierz prawdopodobieństw [numClasses x batchSize]
%       każda kolumna sumuje się do 1

    Z = Z - max(Z, [], 1);

    expZ = exp(Z);

    A = expZ ./ sum(expZ, 1);

end