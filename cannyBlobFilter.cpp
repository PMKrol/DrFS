#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Parametry Canny'ego i Hougha
#define CANNY_THRESH1 50
#define CANNY_THRESH2 150
#define ANGLE_TOLERANCE 20
//#define CUT_LEFT 50


#define RATIO_MIN_LENGTH 1/8
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

vector<Point> findSecondEdgePoints(const Mat& srcImage, const vector<Point>& certainPoints, int windowSize) {
    // Zakładając, że chcesz zwrócić punkty, które są drugą krawędzią
    vector<Point> secondEdgePoints;

    // Iteracja przez wszystkie punkty w wektorze
    for (const auto& point : certainPoints) {
        // Sprawdzenie, czy punkt znajduje się w obrębie obrazu
        if (point.x >= 0 && point.x < srcImage.cols && point.y >= 0 && point.y < srcImage.rows) {
            // Tworzymy histogram dla tej kolumny
            vector<int> hist(256, 0); // Histogram dla intensywności pikseli (zakładając skalę szarości)

            // Zbieranie danych z kolumny nad punktem (przyjmiemy np. od punktu do `windowSize` w górę)
            for (int i = max(0, point.y - windowSize); i < point.y; ++i) {
                // Zakładając, że mamy kolorowy obraz (BGR), to bierzemy średnią wartość pikseli
                Vec3b pixel = srcImage.at<Vec3b>(i, point.x);
                // Możemy użyć wartości intensywności w skali szarości (średnia z kanałów BGR)
                int intensity = (pixel[0] + pixel[1] + pixel[2]) / 3;  // Skala szarości
                hist[intensity]++;
            }

            // Wyświetlenie histogramu
            Mat histImage(400, 256, CV_8UC1, Scalar(255));
            int maxVal = *max_element(hist.begin(), hist.end());
            for (int i = 0; i < 256; ++i) {
                line(histImage, Point(i, 400), Point(i, 400 - (hist[i] * 400 / maxVal)), Scalar(0), 1, 8, 0);
            }

            // Wyświetlenie histogramu
            imshow("Histogram for column " + to_string(point.x), histImage);
            waitKey(0); // Czekanie na naciśnięcie klawisza, aby zamknąć okno

            // Jeśli masz jakąś logikę do wykrywania drugiego punktu krawędzi (np. zmiana intensywności),
            // dodaj ją tutaj i dodaj punkt do `secondEdgePoints`.
        }
    }

    return secondEdgePoints;
}

vector<Point> findSecondEdgePoints1(const Mat& srcImage, const vector<Point>& certainPoints, int windowSize) {
    vector<Point> secondEdgePoints;

    // Konwersja obrazu do odcieni szarości
    Mat grayImage;
    cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);

    Mat resizedImage;
    resize(grayImage, resizedImage, Size(srcImage.cols, srcImage.rows / 2));  // Nowa szerokość i nowa wysokość
    
    // Wykonanie detekcji krawędzi Canny'ego
    Mat edgesHalf, edges;
    Canny(resizedImage, edgesHalf, CANNY_THRESH1*0.8, CANNY_THRESH2);
    
    resize(edgesHalf, edges, Size(srcImage.cols, srcImage.rows));  // Nowa szerokość i nowa wysokość
    
    
    removePointsBelowCertainPoints(edges, certainPoints);

    vector<Vec4i> lines;
    int lineVoters = 200;       // Początkowa liczba głosów dla Hough
    int minLineLength = srcImage.cols * RATIO_MIN_LENGTH; 
    int maxLineGap = srcImage.cols * RATIO_MAX_GAP;   

    // Pętla do wykrywania linii, aż znajdzie co najmniej jedną linię
    while (lines.empty() && lineVoters > 0) {  // Dopóki brak linii i lineVoters > 0
        lines.clear();  // Wyczyść poprzednie wykryte linie

        // Wykrywanie linii metodą HoughLinesP
        HoughLinesP(edges, lines, 1, CV_PI / 180, lineVoters, minLineLength, maxLineGap);
        
        // Filtracja linii, pozostawiając tylko linie poziome
        filterHorizontalLines(lines);
        
        //filterLinesAboveCertainPoints(lines, certainPoints);

        // Jeśli wciąż nie znaleziono linii, zmniejsz liczbę głosów
        lineVoters--;
    }

    // Przechodzimy przez wykryte linie i sprawdzamy, które nie przechodzą przez punkty z certainPoints
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i line = lines[i];
        
        secondEdgePoints.push_back(Point(line[0], line[1]));
        secondEdgePoints.push_back(Point(line[2], line[3]));

        //Dodajemy punkty pomiędzy końcami linii
        vector<Point> pointsBetween = getPointsBetween(line[0], line[1], line[2], line[3]);
        secondEdgePoints.insert(secondEdgePoints.end(), pointsBetween.begin(), pointsBetween.end());
    }
    
    cout << "Voters: " << lineVoters << ". Wykryte punkty: " << endl;
    for (const auto& p : secondEdgePoints) {
        cout << "(" << p.x << ", " << p.y << ")" << endl;
    }
    cout << "Koniec. " << endl;
    
    return secondEdgePoints;
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
    vector<Point> secondEdgePoints = findSecondEdgePoints(srcImage, certainPoints, 5);

    // Wizualizacja wyników
    for (const auto& p : secondEdgePoints) {
        circle(imgCopy, p, 3, Scalar(0, 0, 255), -1);  // Rysowanie punktów krawędziowych (czerwonych)
    }

    // Na obraz nanosimy punkty "pewne" (zielone)
    for (const auto& p : certainPoints) {
        circle(imgCopy, p, 2, Scalar(0, 255, 0), -1);  // Zielone punkty
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
    findAndDrawVerticalLines(srcImage, imgForDisplay);

    // Wyświetlanie finalnego obrazu z narysowanymi liniami poziomymi i pionowymi
    imshow("Detected Lines", imgForDisplay);
    waitKey(0);  // Czeka na naciśnięcie klawisza przed zamknięciem okna

    return 0;
}
