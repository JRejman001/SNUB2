%%Analiza klas
classes = unique(Y);

counts = zeros(length(classes),1);

for i = 1:length(classes)
    counts(i) = sum(Y == classes(i));
end

[classes counts]

figure;
bar(classes, counts);
title('Rozkład klas arytmii');
xlabel('Klasa');
ylabel('Liczba próbek');
grid on;

%% One-Hot encoding
K = length(classes);
N = length(Y);

Y_One_Hot = zeros(K, N);

for i = 1:N
    classIndex = find(classes == Y(i));
    Y_One_Hot(classIndex, i) = 1;
end
%% Z-score

mu = mean(X);
sigma = std(X);

sigma(sigma == 0) = 1;

X = (X - mu) ./ sigma;
%% Mieszanie danych
idx = randperm(size(X,1));
X = X(idx,:);
Y_One_Hot = Y_One_Hot(:,idx);