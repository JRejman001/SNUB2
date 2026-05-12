%%Analiza klas
classes = unique(Y);

counts = zeros(length(classes),1);

for i = 1:length(classes)
    counts(i) = sum(Y == classes(i));
end

figure;
bar(classes, counts);
title('Rozkład klas arytmii');
xlabel('Klasa');
ylabel('Liczba próbek');
grid on;

%% Uuswanie klas z liczbą próbek poniżej minData
idx = find(counts < params.minData);
class_to_delete = classes(idx);
for i = 1:length(class_to_delete)
    idx_to_delete = find(Y == class_to_delete(i));
    % Usuwanie klas z liczbą próbek poniżej minData
    Y(idx_to_delete) = []; % Usunięcie próbek z klasami poniżej minData
    X(idx_to_delete, :) = []; % Usunięcie odpowiadających próbek z macierzy X
end

classes = unique(Y);
counts = zeros(length(classes), 1);

for i = 1:length(classes)
    counts(i) = sum(Y == classes(i));
end
%KOD TESTOWY
% idx = find(Y == 1);
% idx = idx(1:round(end/2));
% Y(idx) = []; % Usunięcie próbek z klasami poniżej minData
% X(idx, :) = []; % Usunięcie odpowiadających próbek z macierzy X
%KOD TESTOWY
classes = unique(Y);
counts = zeros(length(classes), 1);
for i = 1:length(classes)
    counts(i) = sum(Y == classes(i));
end
figure;
bar(classes, counts);
title('Rozkład klas arytmii, po usunięciu klas');
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