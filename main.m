Y = [
    0 1;
    1 0;
    0 0
];

Yhat = [
    0.1 0.8;
    0.8 0.1;
    0.1 0.1
];

model.numLayers = 1;
model.W{1} = randn(3, 3);

loss = computeLoss(Yhat, Y, model, 1e-4);

disp(loss);