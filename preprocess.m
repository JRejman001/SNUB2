clc;
clear;
close all;
%Ścierzka do pliku
filename = 'C:\Users\julia\OneDrive\Pulpit\Studia\Studia sem.6\SNUB\SNB_26L_projekt_dane\SNB_26L_projekt_dane\Arrhythmia_MLR\arrhythmia.data';

% Otwórz plik
fid = fopen(filename,'r');
if fid==-1
    error('Nie można otworzyć pliku: %s', filename);
end

data = [];          
lineIdx = 0;

while ~feof(fid) %Wczytanie linijek pliku, do końca pliku
    tline = fgetl(fid);
    if isempty(tline) %Pominięcie pustych linii
        continue
    end
    lineIdx = lineIdx + 1;
    parts = strsplit(tline, ',');       % celowo zwraca komórki tekstu
    if numel(parts) ~= 280
        warning('Linia %d ma %d wartości (oczekiwano 280).', lineIdx, numel(parts));      
        parts(end+1:280) = {''};
    end
    % konwersja '?' -> NaN, pozostałe -> double
    row = nan(1,280);
    for k = 1:280
        s = strtrim(parts{k});
        if isempty(s) || strcmp(s,'?')
            row(k) = NaN;
        else
            row(k) = str2double(s);    % str2double zwraca NaN przy nieprawidłowym
        end
    end
    data(lineIdx,:) = row;
end

fclose(fid);

%% Podział

X = data(:,1:end-1); %Macierz danych
Y = data(:,end); %Wektor klas

%% Podstawowe informacje

fprintf("Liczba próbek: %d\n", size(X,1));
fprintf("Liczba cech: %d\n", size(X,2));

classes = unique(Y);

fprintf("Liczba klas: %d\n", length(classes));

disp("Klasy:");
disp(classes');