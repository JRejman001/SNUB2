N = size(X,1);

trainEnd = round(0.8 * N); %Podział danych w formule 80:10:10
valEnd   = round(0.9 * N);

Xtrain = X(1:trainEnd,:);
Ytrain = Y_One_Hot(:,1:trainEnd);

Xval = X(trainEnd+1:valEnd,:);
Yval = Y_One_Hot(:,trainEnd+1:valEnd);

Xtest = X(valEnd+1:end,:);
Ytest = Y_One_Hot(:,valEnd+1:end);

%%Transpozycja żeby sample były w kolumnach a cechy w rzędach
dataset.Xtrain = Xtrain';
dataset.Xval   = Xval';
dataset.Xtest  = Xtest';

dataset.Ytrain = Ytrain;
dataset.Yval   = Yval;
dataset.Ytest  = Ytest;