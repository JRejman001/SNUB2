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
    Y_One_Hot(classIndex, i) = 1; %Jedynka ustawiana jest dla klasy do której przynależy dana próbka
end
%% Z-score

mu = mean(X);
sigma = std(X);

sigma(sigma == 0) = 1;

X = (X - mu) ./ sigma; %Normalizacja danych metodą Z-score
%% Mieszanie danych
idx = randperm(size(X,1)); %Mieszanie danych losowe w celu zapewnienia równomiernego podziału zbiorów
X = X(idx,:);
Y_One_Hot = Y_One_Hot(:,idx);