cv1 = cvpartition(Y,'HoldOut',0.15);
idxTrainVal = training(cv1);
idxTest     = test(cv1);
YtrainVal = Y(idxTrainVal);

cv2 = cvpartition(YtrainVal,'HoldOut',0.1765);
trainValIndices = find(idxTrainVal);

idxTrain = trainValIndices(training(cv2));

idxVal = trainValIndices(test(cv2));

Xtrain = X(idxTrain,:);
Ytrain = Y_One_Hot(:,idxTrain);

Xval = X(idxVal,:);
Yval = Y_One_Hot(:,idxVal);

Xtest = X(idxTest,:);
Ytest = Y_One_Hot(:,idxTest);

%%Transpozycja żeby sample były w kolumnach a cechy w rzędach
dataset.Xtrain = Xtrain';
dataset.Xval   = Xval';
dataset.Xtest  = Xtest';

dataset.Ytrain = Ytrain;
dataset.Yval   = Yval;
dataset.Ytest  = Ytest;