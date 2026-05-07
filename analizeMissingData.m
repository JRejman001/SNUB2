%% Analiza braków danych

missingCount = sum(isnan(X));

missingRatio = missingCount / size(X,1);

fprintf("\n=== Braki danych ===\n");

fprintf("Maksymalna liczba braków: %d\n", max(missingCount));

fprintf("Liczba cech z brakami: %d\n", ...
    sum(missingCount > 0));

figure;

bar(missingRatio);

title('Procent braków danych w cechach');
xlabel('Numer cechy');
ylabel('Procent braków');
grid on;

%% Usuwanie cech z dużą liczbą braków

threshold = 0.3;

featuresToRemove = missingRatio > threshold;

fprintf("\nUsuwanych cech: %d\n", ...
    sum(featuresToRemove));
fprintf("Usunięta cecha nr. %d\n", find(featuresToRemove == 1));

X(:,featuresToRemove) = [];
fprintf("Nowa liczba cech: %d\n", size(X,2));