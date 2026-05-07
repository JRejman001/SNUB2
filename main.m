preprocess; %Skrypt wczytujący dane z pliku i dzielący je na macierz danych i wektor cech.
analizeMissingData; %Skrypt analizujący ilość brakujących danych w różnych cechach.
                    %Skrypt usuwa cechy, gdzie ponad 30% danych jest wybrakowanych.
removeRedundantFeatures; %Skrypt analizuje korelacje cech w poszukiwaniu cech, 
                         %które są do siebie bardzo podobne.
                         %Te cechy są następnie usuwane ze zbioru, w celu
                         %przyśpieszenia uczenia. Brakujące dane
                         %uzupełniane są mediana.
normalizeData; %Skrypt normalizuje dane tak aby wszytskie były na podobnym poziomie wielkości
               %Dodatkowo wyprowadza macierz kodującą klasy w formule
               %one-hot
splitData; %Skrypt zapewnia odpowiedni podział danych na test, walidacje i trening