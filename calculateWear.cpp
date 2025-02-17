#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

using namespace cv;
using namespace std;

// Parametry Canny'ego i Hougha
#define CANNY_THRESH1 50
#define CANNY_THRESH2 150
#define ANGLE_TOLERANCE 20
//#define CUT_LEFT 50


#define RATIO_MIN_LENGTH 1/6
#define RATIO_MAX_GAP   1/40

#define SMOOTH_WINDOW 5

// Globalne zmienne do przechowywania wykrytych linii poziomych i pionowych
vector<Vec4i> linesHorizontal;
vector<Vec4i> linesVertical;

// Funkcja do filtrowania linii poziomych (±10 stopni)
void filterHorizontalLines(vector<Vec4i>& lines) {
    vector<Vec4i> filteredLines;

    // Przechodzimy przez wszystkie wykryte linie
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i line = lines[i];

        // Obliczanie kąta linii
        double angle = atan2(line[3] - line[1], line[2] - line[0]) * 180.0 / CV_PI;  // Kąt w stopniach

        // Jeśli kąt jest w przedziale od -10 do 10 stopni, to jest linia pozioma
        if (fabs(angle) <= ANGLE_TOLERANCE) {
            filteredLines.push_back(line);
        }
    }

    // Zamiana wynikowych linii na zaktualizowaną listę
    lines = filteredLines;
}

// Funkcja do filtrowania linii pionowych (±10 stopni od 90 stopni)
void filterVerticalLines(vector<Vec4i>& lines) {
    vector<Vec4i> filteredLines;

    // Przechodzimy przez wszystkie wykryte linie
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i line = lines[i];

        // Obliczanie kąta linii
        double angle = atan2(line[3] - line[1], line[2] - line[0]) * 180.0 / CV_PI;  // Kąt w stopniach

        //cout << angle << " ";
        // Jeśli kąt jest w przedziale od 80 do 100 stopni, to jest linia pionowa
        if (fabs(angle - 90) <= ANGLE_TOLERANCE || fabs(angle + 90) <= ANGLE_TOLERANCE) {
            filteredLines.push_back(line);
        }
    }
    
    //cout << " = " << filteredLines.size() << endl;

    // Zamiana wynikowych linii na zaktualizowaną listę
    lines = filteredLines;
}

// Funkcja odpowiedzialna za rozbudowę punktów w lewo lub prawo
void expandPointsInDirection(vector<Point>& points, int direction, const Mat& edges, vector<bool>& processedColumns) {
    for (size_t i = 0; i < points.size(); i++) {
        Point current = points[i];
        int x = current.x;
        int y = current.y;
        
        // Sprawdzanie punktów w kierunku lewym lub prawym w zależności od direction
        int startOffset = (direction == -1) ? 1 : -1;  // W lewo lub w prawo
        int maxOffset = (direction == -1) ? x : (edges.cols - x - 1);  // Maksymalny zakres w kierunku lewym lub prawym

        for (int offset = startOffset; abs(offset) <= 5 && abs(offset) <= maxOffset; offset += startOffset) {
            for (int newY = max(0, y - 5); newY <= min(edges.rows - 1, y + 5); newY++) {
                if (edges.at<uchar>(newY, x + offset) == 255) {  // Punkt Canny'ego w pobliżu
                    Point p(x + offset, newY);
                    if (!processedColumns[x + offset]) {
                        points.push_back(p);  // Dodajemy nowy punkt "pewny"
                        processedColumns[x + offset] = true;
                        break;  // Przerywamy po znalezieniu punktu
                    }
                }
            }
        }
    }
}

void interpolateMissingPoints(vector<Point>& certainPoints) {
    if (certainPoints.empty()) return;

    // Sortujemy punkty po współrzędnej X, żeby móc łatwo interpolować
    sort(certainPoints.begin(), certainPoints.end(), [](const Point& a, const Point& b) {
        return a.x < b.x;
    });

    Point lastPositive = certainPoints[0];  // Pierwszy punkt w wektorze
    vector<Point> interpolatedPoints;  // Przechowujemy interpolowane punkty

    // Przeszukiwanie od drugiego punktu
    for (size_t i = 1; i < certainPoints.size(); i++) {
        Point currentPoint = certainPoints[i];

        // Sprawdzamy, czy jest luka między ostatnim punktem a bieżącym
        if (currentPoint.x > lastPositive.x + 1) {
            // Interpolacja punktów między lastPositive a currentPoint
            float slope = float(currentPoint.y - lastPositive.y) / float(currentPoint.x - lastPositive.x);  // Współczynnik kierunkowy

            // Interpolacja dla brakujących punktów
            for (int x = lastPositive.x + 1; x < currentPoint.x; x++) {
                int interpolatedY = cvRound(lastPositive.y + slope * (x - lastPositive.x));  // Obliczamy Y dla punktu
                Point p(x, interpolatedY);  // Interpolowany punkt
                interpolatedPoints.push_back(p);
            }
        }

        // Dodajemy bieżący punkt do wektora
        interpolatedPoints.push_back(currentPoint);
        lastPositive = currentPoint;  // Aktualizujemy ostatni punkt
    }

    // Zastępujemy oryginalny wektor nowymi punktami
    certainPoints = interpolatedPoints;
}

void smoothPoints(vector<Point>& certainPoints, int windowSize = 3) {
    if (certainPoints.size() < 2) return;

    vector<Point> smoothedPoints;
    int n = certainPoints.size();

    // Iterujemy po punktach w wektorze
    for (int i = 0; i < n; i++) {
        int sumX = 0, sumY = 0, weightSum = 0;

        // Przeszukujemy otoczenie punktu, biorąc pod uwagę windowSize (rozmiar okna)
        for (int j = std::max(0, i - windowSize); j <= std::min(n - 1, i + windowSize); j++) {
            // Wagi punktów: im bliżej punktu, tym wyższa waga
            int weight = windowSize - std::abs(i - j);  // Waga rośnie im bliżej
            sumX += certainPoints[j].x * weight;
            sumY += certainPoints[j].y * weight;
            weightSum += weight;
        }

        // Obliczamy nową wartość punktu na podstawie średniej ważonej
        int smoothedX = sumX / weightSum;
        int smoothedY = sumY / weightSum;

        smoothedPoints.push_back(Point(smoothedX, smoothedY));
    }

    // Zastępujemy oryginalny wektor wygładzonymi punktami
    certainPoints = smoothedPoints;
}

Mat copyAbovePointsToBelow(const Mat& srcImage, Mat& imgCopy, const vector<Point>& certainPoints) {

    // Iterujemy przez każdy punkt pewny
    for (const Point& p : certainPoints) {
        // Sprawdzamy, czy punkt nie jest na samej górze obrazu
        if (p.y > 0) {
            // Pobieramy kolor piksela powyżej punktu w obrazie srcImage
            Vec3b colorAbove = srcImage.at<Vec3b>(p.y - 1, p.x);

            // Kopiujemy wartość koloru powyższego piksela na wszystkie piksele poniżej w imgCopy
            for (int y = p.y; y < srcImage.rows; y++) {
                imgCopy.at<Vec3b>(y, p.x) = colorAbove;
            }
        }
    }

    return imgCopy;
}

bool isPointBelowLine(const Point& p, const Vec4i& line) {
    // Ekstrahujemy punkty końcowe linii
    Point p1(line[0], line[1]);  // Pierwszy punkt linii
    Point p2(line[2], line[3]);  // Drugi punkt linii

    // Sprawdzamy, czy punkt p leży poniżej linii
    // Używamy wzoru na prostą (ax + by + c = 0), aby sprawdzić położenie punktu względem linii.
    // Dla linii (p1, p2) obliczamy współczynniki a, b, c w równaniu prostej:
    int a = p2.y - p1.y;
    int b = p1.x - p2.x;
    int c = p2.x * p1.y - p2.y * p1.x;

    // Sprawdzamy, czy punkt leży poniżej linii
    int linePosition = a * p.x + b * p.y + c;

    // Jeżeli wynik jest większy od 0, punkt znajduje się powyżej linii, w przeciwnym razie poniżej
    return linePosition > 0;
}

void filterLinesAboveCertainPoints(vector<Vec4i>& lines, const vector<Point>& certainPoints) {
    // Usuwamy linie, które są poniżej punktów w certainPoints
    auto it = lines.begin();
    while (it != lines.end()) {
        bool shouldRemoveLine = false;

        // Sprawdzamy, czy linia przechodzi przez punkt poniżej jakiegokolwiek z punktów w certainPoints
        for (const auto& p : certainPoints) {
            if (isPointBelowLine(p, *it)) {
                shouldRemoveLine = true;
                break;  // Jeżeli linia jest poniżej punktu, odrzucamy ją
            }
        }

        // Jeśli linia nie powinna być usunięta, przechodzimy do następnej
        if (shouldRemoveLine) {
            it = lines.erase(it);  // Usuwamy linię z wektora
        } else {
            ++it;
        }
    }
}

void removePointsBelowCertainPoints(Mat& cannyEdges, const vector<Point>& certainPoints) {
    // Tworzymy kopię obrazu Canny'ego
    Mat modifiedEdges = cannyEdges.clone();

    // Iterujemy przez każdą kolumnę obrazu Canny'ego
    for (int x = 0; x < cannyEdges.cols; x++) {
        // Szukamy najwyższego punktu z certainPoints w tej kolumnie
        int maxY = -1;
        
        for (const Point& p : certainPoints) {
            if (p.x == x) {  // Sprawdzamy tylko punkty w tej samej kolumnie
                maxY = max(maxY, p.y);  // Ustawiamy najwyższy punkt w tej kolumnie
            }
        }

        // Jeśli nie znaleziono punktu w tej kolumnie, wyczyść całą kolumnę
        if (maxY == -1) {
            modifiedEdges.col(x).setTo(Scalar(0));  // Wyczyść całą kolumnę
        }
        else {
            // Usuwamy wszystkie punkty poniżej maxY
            for (int y = maxY-5; y < cannyEdges.rows; y++) {
                modifiedEdges.at<uchar>(y, x) = 0;  // Zerujemy piksele poniżej punktu
            }
        }
    }
    
    // Wyświetlamy obraz
    //imshow("Result Image", modifiedEdges);  // Wyświetlanie obrazu z wykrytymi liniami i punktami

    // Kopiujemy zmodyfikowany obraz z powrotem do oryginalnego
    cannyEdges = modifiedEdges;
}


// Funkcja, która zwraca punkty pomiędzy dwoma punktami końcowymi linii
vector<Point> getPointsBetween(int x1, int y1, int x2, int y2) {
    vector<Point> points;

    // Obliczamy różnicę w osiach X i Y
    int dx = x2 - x1;
    int dy = y2 - y1;

    // Obliczamy maksymalną liczbę kroków (odległość między punktami)
    int steps = max(abs(dx), abs(dy));

    // Normalizujemy różnicę w X i Y, aby każdy krok miał jednostkową długość
    float xIncrement = dx / (float)steps;
    float yIncrement = dy / (float)steps;

    // Dodajemy punkty pomiędzy końcami
    for (int i = 0; i <= steps; i++) {
        int x = cvRound(x1 + i * xIncrement);
        int y = cvRound(y1 + i * yIncrement);
        points.push_back(Point(x, y));
    }

    return points;
}

// Funkcja do porównywania punktów według współrzędnej X
bool compareByX(const Point& a, const Point& b) {
    return a.x < b.x;
}

double calculateMean(const std::vector<int>& intensities) {
    // Obliczanie średniej przy użyciu std::accumulate
    double sum = std::accumulate(intensities.begin(), intensities.end(), 0.0);
    return sum / intensities.size();
}

double calculateStdDev(const vector<int>& intensities, double mean) {
    double sum = 0.0;
    for (int value : intensities) {
        sum += pow(value - mean, 2);
    }
    return sqrt(sum / intensities.size());
}

double calculateMedian(vector<int>& intensities) {
    sort(intensities.begin(), intensities.end());
    int size = intensities.size();
    if (size % 2 == 0) {
        return (intensities[size / 2 - 1] + intensities[size / 2]) / 2.0;
    } else {
        return intensities[size / 2];
    }
}

/*vector<Point> findSecondEdgePoints(const Mat& srcImage, Mat& outImage, vector<Point>& certainPoints, int windowSize) {
    if (certainPoints.size() < 100) {
        cerr << "Lista certainPoints ma mniej niż 100 punktów!" << endl;
        return {};
    }

    // Sortowanie punktów według współrzędnej X
    sort(certainPoints.begin(), certainPoints.end(), [](const Point& a, const Point& b) { return a.x < b.x; });

    // Wybór setnego punktu (indeks 99)
    Point point = certainPoints[99];

    if (point.x >= 0 && point.x < srcImage.cols && point.y >= 0 && point.y < srcImage.rows) {
        // Tworzymy kopię obrazu i stosujemy rozmycie (blur)
        Mat blurredImage = srcImage.clone();
        GaussianBlur(blurredImage, blurredImage, Size(25, 5), 0);  // Zastosowanie rozmycia Gaussa

        // Weź intensywność pikseli w kolumnie
        vector<int> intensities;

        // Zbieranie danych z kolumny od góry do dołu (od 0 do Y punktu)
        for (int i = 0; i < point.y; ++i) {
            Vec3b pixel = blurredImage.at<Vec3b>(i, point.x);
            int intensity = (pixel[0] + pixel[1] + pixel[2]) / 3;  // Skala szarości
            intensities.push_back(intensity);
        }

        // Obliczenie statystyk z pierwszych 80% intensywności
        int num80 = static_cast<int>(intensities.size() * 0.8);
        vector<int> first80(intensities.begin(), intensities.begin() + num80);

        double minVal = *min_element(first80.begin(), first80.end());
        double meanVal = calculateMean(first80);
        double stdDev = calculateStdDev(first80, meanVal);
        double medianVal = calculateMedian(first80);

        // Wypisywanie statystyk
        cout << "Min: " << minVal << endl;
        cout << "Mean: " << meanVal << endl;
        cout << "Standard Deviation: " << stdDev << endl;
        cout << "Median: " << medianVal << endl;

        // Szukanie pierwszego spadku w pozostałych 20%
        vector<int> last20(intensities.begin() + num80, intensities.end());

        for (int i = 1; i < last20.size(); ++i) {
            // Szukamy spadku poniżej min - std
            if (last20[i] < minVal - stdDev) {
                cout << "Pierwszy spadek poniżej wartości minimalnej - odch. std. wykryty na indeksie: " << num80 + i << endl;
                outImage.at<Vec3b>(num80 + i, point.x) = Vec3b(0, 255, 255);
                //circle(outImage, last20[i], 3, Scalar(0, 0, 255), -1);
                break;
            }
        }

        // Kolumna przed
        for (int y = 0; y < srcImage.rows; ++y) {
            outImage.at<Vec3b>(y, point.x + 1) = Vec3b(0, 255, 255); // Żółty kolor (BGR)
        }

        // Kolumna po
        for (int y = 0; y < srcImage.rows; ++y) {
            outImage.at<Vec3b>(y, point.x - 1) = Vec3b(0, 255, 255); // Żółty kolor (BGR)
        }
    }

    return {}; // Na razie zwracamy pustą listę
}*/

vector<Point> findSecondEdgePoints(const Mat& srcImage, Mat& outImage, vector<Point>& certainPoints, int windowSize) {
    vector<Point> resultPoints; // Lista, która będzie zawierała punkty przełamania

    if (certainPoints.size() < 100) {
        cerr << "Lista certainPoints ma mniej niż 100 punktów!" << endl;
        return resultPoints;
    }

    // Sortowanie punktów według współrzędnej X
    sort(certainPoints.begin(), certainPoints.end(), [](const Point& a, const Point& b) { return a.x < b.x; });
    
    // Tworzymy kopię obrazu i stosujemy rozmycie (blur)
    Mat blurredImage = srcImage.clone();
    GaussianBlur(blurredImage, blurredImage, Size(25, 1), 0);  // Zastosowanie rozmycia Gaussa

    // Iteracja po wszystkich punktach
    for (const Point& point : certainPoints) {
        if (point.x >= 0 && point.x < srcImage.cols && point.y >= 0 && point.y < srcImage.rows) {

            // Weź intensywność pikseli w kolumnie
            vector<int> intensities;

            // Zbieranie danych z kolumny od góry do dołu (od 0 do Y punktu)
            for (int i = 0; i < point.y; ++i) {
                Vec3b pixel = blurredImage.at<Vec3b>(i, point.x);
                int intensity = (pixel[0] + pixel[1] + pixel[2]) / 3;  // Skala szarości
                intensities.push_back(intensity);
            }

            // Obliczenie statystyk z pierwszych 80% intensywności
            int num80 = static_cast<int>(intensities.size() * 0.8);
            vector<int> first80(intensities.begin(), intensities.begin() + num80);

            double minVal = *min_element(first80.begin(), first80.end());
            double meanVal = calculateMean(first80);
            double stdDev = calculateStdDev(first80, meanVal);
            double medianVal = calculateMedian(first80);

            // Szukanie pierwszego spadku w pozostałych 20%
            vector<int> last20(intensities.begin() + num80, intensities.end());

            for (int i = 1; i < last20.size(); ++i) {
                // Szukamy spadku poniżej min - std
                if (last20[i] < meanVal - stdDev) {
                    //cout << "Pierwszy spadek poniżej wartości minimalnej - odch. std. wykryty na indeksie: " << num80 + i << endl;

                    // Dodajemy punkt przełamania do wynikowej listy
                    resultPoints.push_back(Point(point.x, num80 + i));

                    // Przerywamy szukanie dla tego punktu
                    break;
                }
            }
        }
    }

    return resultPoints; // Zwracamy listę punktów przełamania
}


void findAndDrawHorizontalLines(const Mat& srcImage, Mat& imgCopy) {
    // Konwertowanie obrazu na odcienie szarości
    Mat grayImage;
    cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);

    // Wykonanie detekcji Canny'ego
    Mat edges;
    Canny(grayImage, edges, CANNY_THRESH1, CANNY_THRESH2);

    // Obliczanie początkowych wartości minLineLength i maxLineGap
    int minLineLength = imgCopy.cols * RATIO_MIN_LENGTH; 
    int maxLineGap = imgCopy.cols * RATIO_MAX_GAP;   
    int lineVoters = 100;

    vector<Vec4i> linesHorizontal;  // Vector do przechowywania wykrytych linii
    vector<Vec4i> detectedLines;  // Vector do przechowywania wykrytych linii

    // Pierwsza iteracja - wykrywanie pierwszych linii
    while (detectedLines.size() < 1) {  // Zatrzymujemy się, gdy wykryjemy pierwszą linię
        // Wykrywanie linii metodą HoughLinesP
        HoughLinesP(edges, linesHorizontal, 1, CV_PI / 180, lineVoters, minLineLength, maxLineGap);

        // Filtracja linii, pozostawiając tylko linie poziome
        filterHorizontalLines(linesHorizontal);

        // Przechodzimy przez wykryte linie i zapisujemy je
        for (size_t i = 0; i < linesHorizontal.size(); i++) {
            detectedLines.push_back(linesHorizontal[i]);
        }

        lineVoters--;  // Zmniejsz liczbę głosów, aby wykrywać mniejsze linie
    }
    
    // Po wykryciu pierwszej linii zmieniamy parametry HoughLinesP
    lineVoters /= 2;  // Zmniejszamy lineVoters o połowę
    minLineLength /= 2;  // Zmniejszamy minLineLength o połowę

    // Drugie wykrywanie linii z nowymi parametrami (brak pętli, wykrywa tyle, ile da się znaleźć)
    vector<Vec4i> additionalLines;  // Nowe linie wykryte po zmianie parametrów

    // Wykrywanie kolejnych linii z nowymi parametrami
    HoughLinesP(edges, linesHorizontal, 1, CV_PI / 180, lineVoters, minLineLength, maxLineGap);

    // Filtracja linii, pozostawiając tylko linie poziome
    filterHorizontalLines(linesHorizontal);

    // Dodajemy wykryte linie do listy
    for (size_t i = 0; i < linesHorizontal.size(); i++) {
        additionalLines.push_back(linesHorizontal[i]);
    }

    // Łączymy linie z obu etapów wykrywania
    vector<Vec4i> allLines = detectedLines;
    allLines.insert(allLines.end(), additionalLines.begin(), additionalLines.end());  // Połączenie wszystkich linii

    // Macierze do przechowywania punktów
    vector<Point> certainPoints;  // Punkty "pewne"
    vector<Point> potentialPoints;  // Punkty "potencjalne"

    // Nałożenie punktów Canny'ego na punkty pewne
    int searchRadius = 5;  // Zakres +/- 5 px w górę i w dół
    vector<bool> processedColumns(edges.cols, false);  // Tablica do śledzenia, które kolumny mają już przypisane punkty pewne

    // Dla każdej linii wykrytej w obu etapach
    for (size_t i = 0; i < allLines.size(); i++) {
        Vec4i l = allLines[i];

        // Sprawdzamy punkty Canny'ego w okolicy tej linii
        for (int x = max(0, l[0] - searchRadius); x <= min(edges.cols - 1, l[2] + searchRadius); x++) {
            for (int y = max(0, l[1] - searchRadius); y <= min(edges.rows - 1, l[3] + searchRadius); y++) {
                if (edges.at<uchar>(y, x) == 255) {  // Jeśli jest punkt Canny'ego
                    Point p(x, y);

                    // Dodajemy punkt do "pewnych", jeśli jeszcze nie ma punktu w tej kolumnie
                    if (!processedColumns[x]) {
                        certainPoints.push_back(p);
                        processedColumns[x] = true;  // Oznaczamy, że kolumna ma punkt pewny
                    }
                }
            }
        }
    }

    // Rozbudowa punktów "pewnych" w lewo
    expandPointsInDirection(certainPoints, -1, edges, processedColumns);

    // Rozbudowa punktów "pewnych" w prawo
    expandPointsInDirection(certainPoints, 1, edges, processedColumns);
    
    // Uzupełnij brakujące punkty rysując pomiędzy znanymi proste
    interpolateMissingPoints(certainPoints);
    
    // Wygładź linie 
    smoothPoints(certainPoints, SMOOTH_WINDOW);
    
    //imgCopy = copyAbovePointsToBelow(srcImage, imgCopy, certainPoints);
    //znajdź krawędź nad...
    vector<Point> secondEdgePoints = findSecondEdgePoints(srcImage, imgCopy, certainPoints, 5);
    
    // Wygładź linie 
    smoothPoints(secondEdgePoints, SMOOTH_WINDOW * 3);

    // Wizualizacja wyników
    for (const auto& p : secondEdgePoints) {
        circle(imgCopy, p, 1, Scalar(0, 0, 255), -1);  // Rysowanie punktów krawędziowych (czerwonych)
    }

   
    // Na obraz nanosimy punkty "pewne" (zielone)
    for (const auto& p : certainPoints) {
        circle(imgCopy, p, 1, Scalar(0, 255, 0), -1);  // Zielone punkty
    }

    
    // Na obraz nanosimy punkty "potencjalne" (niebieskie)
    for (const auto& p : potentialPoints) {
        circle(imgCopy, p, 2, Scalar(255, 0, 0), -1);  // Niebieskie punkty
    }

    // Wyświetlamy obraz
    //imshow("Result Image", imgCopy);  // Wyświetlanie obrazu z wykrytymi liniami i punktami
}


// Funkcja do znajdowania i rysowania linii pionowych
void findAndDrawVerticalLines(const Mat& srcImage, Mat& imgCopy) {
    // Przycinamy obraz do lewej połowy
    Rect roi(0, 0, srcImage.cols / 2, srcImage.rows);
    Mat croppedImage = srcImage(roi);

    // Konwertowanie przyciętego obrazu na odcienie szarości
    Mat grayImage;
    cvtColor(croppedImage, grayImage, COLOR_BGR2GRAY);

    // Wykonanie detekcji Canny'ego
    Mat edges;
    Canny(grayImage, edges, CANNY_THRESH1, CANNY_THRESH2);

    // Obliczanie początkowych wartości minLineLength i maxLineGap
    int minLineLength = max(10, srcImage.rows / 4);  // 1/4 wysokości obrazu
    int maxLineGap = max(10, minLineLength / 10);    // 1/10 minLineLength

    int lineVoters = 100; 

    // Iteracyjne zwiększanie maxLineGap aż do wykrycia 10 linii
    while (linesVertical.size() < 1) {
        // Wykrywanie linii metodą HoughLinesP na przyciętym obrazie
        HoughLinesP(edges, linesVertical, 1, CV_PI / 180, lineVoters, minLineLength, maxLineGap);

        // Zwiększanie maxLineGap o 1
        lineVoters--;
        
        filterVerticalLines(linesVertical);
    }

    // Rysowanie wykrytych linii pionowych na kopii obrazu z uwzględnieniem przesunięcia o CUT_LEFT
    for (size_t i = 0; i < linesVertical.size(); i++) {
        Vec4i l = linesVertical[i];

        // Przesunięcie współrzędnych linii o szerokość prawej połowy obrazu
        //l[0] += srcImage.cols / 2;  // Przesunięcie x początkowego punktu
        //l[2] += srcImage.cols / 2;  // Przesunięcie x końcowego punktu

        // Rysowanie linii na pełnym obrazie
        line(imgCopy, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 2, LINE_AA);
    }
}

// Funkcja do rysowania punktów Canny'ego na obrazie
void drawCannyPoints(const Mat& srcImage, Mat& imgCopy) {
    // Konwertowanie obrazu na odcienie szarości
    Mat grayImage;
    cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);

    // Wykonanie detekcji Canny'ego
    Mat edges;
    Canny(grayImage, edges, CANNY_THRESH1, CANNY_THRESH2);

    // Iterowanie po wszystkich pikselach obrazu i rysowanie białych punktów, gdzie Canny wykrył krawędzie
    for (int y = 0; y < edges.rows; ++y) {
        for (int x = 0; x < edges.cols; ++x) {
            if (edges.at<uchar>(y, x) > 0) {  // Jeśli Canny wykrył krawędź
                // Rysowanie białego punktu na obrazie wyjściowym
                imgCopy.at<Vec3b>(y, x) = Vec3b(255, 255, 255);  // Rysowanie białego punktu
            }
        }
    }

    // Wyświetlanie wynikowego obrazu z punktami Canny'ego
//     imshow("Canny Points", imgCopy);
//     waitKey(0);  // Czeka na naciśnięcie klawisza przed zamknięciem okna
}

int main(int argc, char** argv) {
    // Sprawdzanie argumentów
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    // Wczytanie obrazu
    Mat srcImage = imread(argv[1], IMREAD_COLOR);
    if (srcImage.empty()) {
        cout << "Nie udało się wczytać obrazu!" << endl;
        return -1;
    }

    // Tworzenie kopii obrazu do późniejszego wyświetlania
    Mat imgForDisplay = srcImage.clone();
    
    drawCannyPoints(srcImage, imgForDisplay);

    // Znalezienie i narysowanie linii poziomych na kopii obrazu
    findAndDrawHorizontalLines(srcImage, imgForDisplay);
    
    // Znalezienie i narysowanie linii pionowych na kopii obrazu
    //findAndDrawVerticalLines(srcImage, imgForDisplay);

    // Wyświetlanie finalnego obrazu z narysowanymi liniami poziomymi i pionowymi
    imshow("Detected Lines", imgForDisplay);
    waitKey(0);  // Czeka na naciśnięcie klawisza przed zamknięciem okna
    
    
    return 0;
}
