%% Tymczasowe uzupełnienie braków medianą

Xtemp = X;

for i = 1:size(Xtemp,2)

    col = Xtemp(:,i);

    med = median(col,'omitnan');

    col(isnan(col)) = med;

    Xtemp(:,i) = col;
end

%% Macierz korelacji

R = corrcoef(Xtemp);

figure;

imagesc(R);

colorbar;

title('Macierz korelacji cech');

%% Wyszukiwanie redundantnych cech

corrThreshold = 0.95;

numFeatures = size(R,1);

featuresToDelete = false(1,numFeatures);

for i = 1:numFeatures

    for j = i+1:numFeatures

        if abs(R(i,j)) > corrThreshold

            featuresToDelete(j) = true;

        end
    end
end

fprintf("\nRedundantnych cech: %d\n", ...
    sum(featuresToDelete));
fprintf("Cechy Redundantne: %d\n", find(featuresToDelete == 1));
%% Usuwanie Redundantych cech
X(:,featuresToDelete) = [];

fprintf("Liczba cech po usunięciu redundancji: %d\n", ...
    size(X,2));
%% Uzupełnienie medianą
for i = 1:size(X,2)

    col = X(:,i);

    med = median(col,'omitnan');

    col(isnan(col)) = med;

    X(:,i) = col;
end

%% Kontrola jakości

fprintf("\nPozostałe NaN: %d\n", ...
    sum(isnan(X),'all'));

fprintf("Pozostałe Inf: %d\n", ...
    sum(isinf(X),'all'));
