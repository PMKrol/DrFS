/*
 * compile: g++ -std=c++17 -o measureWear measureWear.cpp `pkg-config --cflags --libs opencv4`
 * run: ./measureWear <ścieżka_do_obrazu1> <ścieżka_do_obrazu2> ...
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>  // Dla operacji na plikach
#include <string>   // Dla std::string
#include <cmath>

#define LOW_THRESHOLD 50
#define HIGH_THRESHOLD 100
#define BLUR_STRENGTH 3
#define NEIGHBOR_RADIUS_X 5  // Przykładowa wartość dla osi X
#define NEIGHBOR_RADIUS_Y 2  // Przykładowa wartość dla osi Y
#define MIN_NEIGHBORS 10     // Minimalna liczba sąsiadów, aby zachować punkt krawędziowy

#define NEIGHBOR_RADIUS 3    // Promień sprawdzania sąsiadujących pikseli


using namespace cv;
using namespace std;

Mat imgOriginal, imgGray, imgEdges, imgEdgesOriginal, imgResult;
int maxY = 0, minY = 0, maxX = 0; // Zmienna dla max i min wartości Y, X
int prevSliderX = -1; // Nowa zmienna do przechowywania poprzedniej wartości slidera maxX

int yDifference;

void saveResults(const std::string& filePath) {
    // Tworzenie ścieżki do pliku .txt obok obrazu
    std::string outputFilePath = filePath.substr(0, filePath.find_last_of(".")) + ".txt";
    
    // Otwieranie pliku do zapisu
    std::ofstream outputFile(outputFilePath);

    // Sprawdzanie, czy plik się otworzył
    if (!outputFile.is_open()) {
        std::cerr << "Nie udało się otworzyć pliku do zapisu: " << outputFilePath << std::endl;
        return;
    }

    // Zapisanie danych do pliku: filePath, maxX, maxY, minY, minY - maxY
    outputFile << filePath << "\t"  // Ścieżka do pliku obrazu
               << maxX << "\t"       // maxX
               << maxY << "\t"       // maxY
               << minY << "\t"       // minY
               << (minY - maxY) << "\n"; // minY - maxY

    // Zamknięcie pliku
    outputFile.close();

    // Informacja o zapisaniu pliku
    std::cout << "Wyniki zapisane w pliku: " << outputFilePath << std::endl;
}

// Funkcja do obliczenia punktów o największej odległości w kolumnie
void findMaxDistanceColumn() {
    int maxDistance = 0;
    std::vector<int> firstYValues; // Zbiera wartości firstY
    std::vector<int> lastYValues;  // Zbiera wartości lastY

    // Najpierw przejrzyj wszystkie kolumny i zidentyfikuj firstY i lastY
    for (int x = 0; x < imgEdges.cols * 0.66; ++x) {
        int firstY = -1, lastY = -1;
        for (int y = 0; y < imgEdges.rows; ++y) {
            if (imgEdges.at<uchar>(y, x) > 0) {
                if (firstY == -1) {
                    firstY = y;
                } else {
                    lastY = y;
                    break;
                }
            }
        }
        
        if (firstY != -1 && lastY != -1) {
            firstYValues.push_back(firstY);
            lastYValues.push_back(lastY);
        }
    }

    // Oblicz średnie i odchylenia standardowe dla firstY i lastY
    double meanFirstY = 0, meanLastY = 0;
    double stdDevFirstY = 0, stdDevLastY = 0;

    // Oblicz średnie
    for (size_t i = 0; i < firstYValues.size(); ++i) {
        meanFirstY += firstYValues[i];
        meanLastY += lastYValues[i];
    }
    meanFirstY /= firstYValues.size();
    meanLastY /= lastYValues.size();

    // Oblicz odchylenie standardowe
    for (size_t i = 0; i < firstYValues.size(); ++i) {
        stdDevFirstY += std::pow(firstYValues[i] - meanFirstY, 2);
        stdDevLastY += std::pow(lastYValues[i] - meanLastY, 2);
    }
    stdDevFirstY = std::sqrt(stdDevFirstY / firstYValues.size());
    stdDevLastY = std::sqrt(stdDevLastY / lastYValues.size());

    // Przejdź ponownie przez wszystkie kolumny, aby wyliczyć odległości, ale uwzględnij średnią i odchylenie
    for (int x = 0; x < imgEdges.cols * 0.66; ++x) {
        int firstY = -1, lastY = -1;
        for (int y = 0; y < imgEdges.rows; ++y) {
            if (imgEdges.at<uchar>(y, x) > 0) {
                if (firstY == -1) {
                    firstY = y;
                } else {
                    lastY = y;
                    break;
                }
            }
        }
        
        if (firstY != -1 && lastY != -1) {
            // Sprawdzanie, czy firstY i lastY mieszczą się w zakresie ±3 odchyleń standardowych
            if (std::abs(firstY - meanFirstY) <= 3 * stdDevFirstY && std::abs(lastY - meanLastY) <= 3 * stdDevLastY) {
                int distance = lastY - firstY;
                if (distance > maxDistance) {
                    maxDistance = distance;
                    maxX = x;
                    maxY = firstY; // Możemy przyjąć, że maxY to początek tej linii
                    minY = lastY;
                }
            }
        }
    }
}

void addEdgeFromNextColumn() {
    int cols = imgEdges.cols;
    int rows = imgEdges.rows;

    // Przechodzimy przez każdą kolumnę (oprócz ostatniej)
    for (int x = 0; x < cols - 1; ++x) {
        bool hasEdgeInCurrentColumn = false;

        // Sprawdzamy, czy w obecnej kolumnie (x) jest jakikolwiek piksel krawędziowy
        for (int y = 0; y < rows; ++y) {
            if (imgEdges.at<uchar>(y, x) > 0) {
                hasEdgeInCurrentColumn = true;
                break;
            }
        }

        // Jeśli w obecnej kolumnie (x) są krawędzie, a w następnej (x+1) nie ma
        if (hasEdgeInCurrentColumn) {
            // Iteracja po wszystkich pikselach w bieżącej kolumnie (x)
            for (int y = 0; y < rows; ++y) {
                if (imgEdges.at<uchar>(y, x) > 0) {
                    // Szukamy krawędzi w następnej kolumnie (x+1) w obrębie promienia
                    for (int dy = -NEIGHBOR_RADIUS; dy <= NEIGHBOR_RADIUS; ++dy) {
                        int ny = y + dy;

                        // Sprawdzamy, czy sąsiedni piksel w kolumnie +1 (nx = x+1) jest w obrębie obrazu
                        if (ny >= 0 && ny < rows) {
                            if (imgEdgesOriginal.at<uchar>(ny, x + 1) > 0) {
                                // Jeśli znajdziemy krawędź, dodajemy ją do obrazu `imgEdges`
                                imgEdges.at<uchar>(ny, x + 1) = 255;  // Dodajemy krawędź
                                break;  // Przechodzimy do następnego piksela w obecnej kolumnie
                            }
                        }
                    }
                }
            }
        }
    }
}

void applyCannyAndBlur() {
    // Zastosowanie rozmycia poziomego (tylko w poziomie) za pomocą rozmycia Gaussa
    Mat tempGray = imgGray.clone(); // Kopiowanie oryginalnego obrazu szaro-skalowego

    if (BLUR_STRENGTH > 0) {
        // Rozmycie tylko w poziomie
        GaussianBlur(tempGray, tempGray, Size(15, 1), 0);
    }

    // Zastosowanie detekcji krawędzi Canny na obrazku
    Canny(tempGray, imgEdges, LOW_THRESHOLD, HIGH_THRESHOLD);

    // Zachowanie oryginalnego obrazu krawędzi Canny
    imgEdgesOriginal = imgEdges.clone();

    // Konwersja do BGR, aby móc nałożyć tekst
    Mat imgDisplay;
    cvtColor(imgEdges, imgDisplay, COLOR_GRAY2BGR);

    // Wyświetlanie krawędzi
    //imshow("Canny Edges", imgDisplay);
}

void cleanColumns() {
    int cols = imgEdges.cols;
    int rows = imgEdges.rows;

    // Analiza każdej kolumny
    for (int col = 0; col < cols; ++col) {
        int edgeCount = 0;

        // Liczymy punkty krawędzi w kolumnie
        for (int row = 0; row < rows; ++row) {
            if (imgEdges.at<uchar>(row, col) > 0) {
                edgeCount++;
            }
        }

        // Jeśli punktów krawędzi w kolumnie jest więcej niż 2, czyścimy całą kolumnę
        if (edgeCount > 2) {
            for (int row = 0; row < rows; ++row) {
                imgEdges.at<uchar>(row, col) = 0; // Czyszczenie kolumny
            }
        }
    }
}

void removeWeakEdges() {
    int cols = imgEdges.cols;
    int rows = imgEdges.rows;

    // Tworzymy kopię oryginalnego obrazu krawędziowego, aby nie modyfikować go podczas iteracji
    Mat imgEdgesCopy = imgEdges.clone();

    // Iteracja po wszystkich punktach krawędzi
    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            if (imgEdges.at<uchar>(y, x) > 0) {
                int neighborCount = 0;

                // Sprawdzanie sąsiadów w zadanym promieniu w osi X i Y
                for (int dy = -NEIGHBOR_RADIUS_Y; dy <= NEIGHBOR_RADIUS_Y; ++dy) {
                    for (int dx = -NEIGHBOR_RADIUS_X; dx <= NEIGHBOR_RADIUS_X; ++dx) {
                        int nx = x + dx;
                        int ny = y + dy;

                        // Sprawdzanie czy sąsiedni piksel jest w obrębie obrazu i czy jest punktem krawędziowym
                        if (nx >= 0 && ny >= 0 && nx < cols && ny < rows) {
                            if (imgEdgesOriginal.at<uchar>(ny, nx) > 0) {
                                neighborCount++;
                            }
                        }
                    }
                }

                // Jeśli liczba sąsiadów jest mniejsza niż minimalna liczba, usuwamy punkt krawędziowy
                if (neighborCount < MIN_NEIGHBORS) {
                    imgEdgesCopy.at<uchar>(y, x) = 0; // Usuwamy punkt krawędziowy
                }
            }
        }
    }

    // Przypisanie zmodyfikowanych krawędzi z powrotem
    imgEdges = imgEdgesCopy.clone();
}

void displayResult() {
    // Tworzymy kopię oryginalnego obrazu, aby nanieść krawędzie
    imgResult = imgOriginal.clone();

    // Nanosimy krawędzie Canny na oryginalny obraz
    for (int y = 0; y < imgEdges.rows; ++y) {
        for (int x = 0; x < imgEdges.cols; ++x) {
            if (imgEdges.at<uchar>(y, x) > 0) {
                imgResult.at<Vec3b>(y, x) = Vec3b(0, 0, 255); // Kolor czerwony dla krawędzi
            }
        }
    }

    // Wyświetlanie oryginalnego obrazu z nałożonymi krawędziami
    //imshow("Result", imgResult);
}

// Funkcja do obliczenia punktów o największej odległości w kolumnie maxX
void findMaxDistanceForX(int x) {
    int firstY = -1, lastY = -1;

    // Przeszukaj w kolumnie x, aby znaleźć pierwszą i ostatnią krawędź
    for (int y = 0; y < imgEdges.rows; ++y) {
        if (imgEdges.at<uchar>(y, x) > 0) {
            if (firstY == -1) {
                firstY = y;  // Znaleziono pierwszy punkt krawędzi
            } else {
                lastY = y;   // Znaleziono ostatni punkt krawędzi
            }
        }
    }

    // Jeśli znaleziono dwa punkty krawędzi, zaktualizuj maxY i minY
    if (firstY != -1 && lastY != -1) {
        maxY = firstY;   // Przypisujemy pierwszy punkt do maxY
        minY = lastY;    // Przypisujemy ostatni punkt do minY
    }
}

// Funkcja do rysowania linii poziomych i pionowej w oparciu o suwaki
void displayResultWithSliders(int, void*) {
    int sliderX = getTrackbarPos("Max X", "Result");  // Pobranie wartości suwaka maxX
    
    // Sprawdzanie, czy suwak maxX się zmienił w porównaniu do poprzedniej wartości
    if (sliderX != prevSliderX) {
        prevSliderX = sliderX;                    // Aktualizacja poprzedniego slidera maxX
        findMaxDistanceForX(sliderX);             // Przeszukiwanie kolumny maxX i aktualizacja maxY, minY
        
        // Ustawienie suwaków na nowe wartości maxY i minY
        setTrackbarPos("Max Y", "Result", maxY);
        setTrackbarPos("Min Y", "Result", minY);
    }

    // Tworzenie kopii obrazu do rysowania
    imgResult = imgOriginal.clone();

    // Rysowanie krawędzi
    for (int y = 0; y < imgEdges.rows; ++y) {
        for (int x = 0; x < imgEdges.cols; ++x) {
            if (imgEdges.at<uchar>(y, x) > 0) {
                imgResult.at<Vec3b>(y, x) = Vec3b(0, 0, 255); // Kolor czerwony dla krawędzi
            }
        }
    }

    // Dodanie linii poziomych w oparciu o suwaki pionowe
    int maxSlider = getTrackbarPos("Max Y", "Result") - 1;
    int minSlider = getTrackbarPos("Min Y", "Result") - 1;
    
    // Wyświetlenie różnicy między Y'kami
    yDifference = minY - maxY;
    std::string text = "Y Difference: " + std::to_string(yDifference);
    
    // Dodanie tekstu do obrazu (pozycja tekstu, czcionka, rozmiar, kolor, grubość)
    putText(imgResult, text, Point(20, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

    // Linie poziome
    if (maxSlider >= 0 && maxSlider < imgEdges.rows) {
        line(imgResult, Point(0, maxSlider), Point(imgResult.cols, maxSlider), Scalar(255, 0, 0), 1); // Czerwona linia pozioma
    }
    if (minSlider >= 0 && minSlider < imgEdges.rows) {
        line(imgResult, Point(0, minSlider), Point(imgResult.cols, minSlider), Scalar(0, 255, 0), 1); // Zielona linia pozioma
    }

    // Linia pionowa
    line(imgResult, Point(sliderX, 0), Point(sliderX, imgResult.rows), Scalar(0, 255, 255), 1); // Żółta linia pionowa

    // Wyświetlanie obrazu z nałożonymi liniami
    imshow("Result", imgResult);
}


// Funkcja inicjalizująca suwaki
void initSliders(){
    // Obliczenie punktu z największą odległością na początku
    findMaxDistanceColumn();  // Początkowe ustawienie wartości maxX, maxY, minY na podstawie obrazu

    // Inicjalizacja okna
    imshow("Result", imgResult);

    // Dodanie suwaków
    createTrackbar("Max Y", "Result", &maxY, imgEdges.rows, displayResultWithSliders);
    createTrackbar("Min Y", "Result", &minY, imgEdges.rows, displayResultWithSliders);
    createTrackbar("Max X", "Result", &maxX, imgEdges.cols, displayResultWithSliders);

    // Ustawienie początkowych wartości suwaków
    setTrackbarPos("Max Y", "Result", maxY);
    setTrackbarPos("Min Y", "Result", minY);
    setTrackbarPos("Max X", "Result", maxX);

    // Wywołanie funkcji na początku
    displayResultWithSliders(0, 0);
}


void processImage(const string& filePath) {
    // Wczytanie obrazu
    imgOriginal = imread(filePath, IMREAD_COLOR);
    if (imgOriginal.empty()) {
        cout << "Błąd: Nie można wczytać obrazu " << filePath << endl;
        return;
    }

    // Konwersja obrazu do skali szarości
    cvtColor(imgOriginal, imgGray, COLOR_BGR2GRAY);

    // Zastosowanie Canny'ego z rozmyciem
    applyCannyAndBlur();

    // Czyszczenie kolumn z więcej niż 2 punktami krawędzi
    cleanColumns();

    // Usuwanie punktów krawędziowych z niewielką liczbą sąsiadów
    removeWeakEdges();
    
    addEdgeFromNextColumn();
    
    // Wyświetlanie wyników
    displayResult();
    
    initSliders();


    // Oczekiwanie na naciśnięcie klawisza
    //waitKey(100);
    waitKey(0);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Użycie: program <ścieżka_do_obrazu1> <ścieżka_do_obrazu2> ..." << endl;
        return -1;
    }

    // Iteracja po wszystkich argumentach (plikach)
    for (int i = 1; i < argc; ++i) {
        string filePath = argv[i];
        cout << "Przetwarzanie obrazu: " << filePath << endl;
        processImage(filePath); // Przetwarzanie każdego obrazu    
        
        // Zapisz wyniki do pliku
        saveResults(filePath);
    }

    return 0;
}
