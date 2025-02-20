/*
 * g++ -std=c++17 -o findEdges findEdges.cpp `pkg-config --cflags --libs opencv4` -lboost_filesystem -lboost_system
 * 
 * sudo apt-get install libboost-filesystem-dev libopencv-imgcodecs-dev libopencv-highgui-dev libopencv-imgproc-dev
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

// Parametry Canny'ego i Hougha
int lowThreshold = 50;
int highThreshold = 100;
//int minLineLength = 300;
//int maxLineGap = 10;

#define CROP_SIZE 0.1

#define LINE_VOTERS 100
#define RATIO_MIN_LENGTH 1/3
#define MIN_LENGTH 100
#define RATIO_MAX_GAP   1/20
#define MAX_GAP   15

#define ANGLE_DIFF 5.0

// Definiowanie makra do zmniejszania obrazu
#define SCALE_FACTOR 0.5

#define THRESHOLD 100;
int houghThreshold = THRESHOLD;  // Początkowa wartość progu głosów

// Globalne zmienne do zapisywania informacji o przycięciu obrazu w pikselach i współczynniku skali
int cropWidthPx = 0;  // Szerokość przycięcia w pikselach
int cropHeightPx = 0; // Wysokość przycięcia w pikselach
float scaleFactor = 1.0f; // Współczynnik skali obrazu

// Zmienna przechowująca ścieżkę do obrazu
string imagePath;

// Zmienna globalna przechowująca wykryte linie
vector<Vec4i> linesP; // Teraz linesP jest globalną zmienną
vector<Vec2f> lines;

// Funkcja do wczytania obrazu
Mat loadImage(const string& imagePath) {
    Mat src = imread(imagePath, IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "Error opening image" << endl;
        return Mat(); // Zwraca pusty obraz w przypadku błędu
    }
    return src;
}

// Funkcja do przycinania obrazu o 2.5% z każdej strony
Mat cropImage(const Mat& src) {
    int width = src.cols;
    int height = src.rows;

    // 2.5% szerokości i wysokości
    cropWidthPx = width * CROP_SIZE;
    cropHeightPx = height * CROP_SIZE;

    // Przycinanie obrazu: lewa, prawa, góra, dół
    Rect cropRegion(cropWidthPx, cropHeightPx, width - 2 * cropWidthPx, height - 2 * cropHeightPx);
    
    return src(cropRegion);
}

// Funkcja do skalowania obrazu do określonego rozmiaru
Mat resizeImage(const Mat& src, int maxWidth = 1280, int maxHeight = 800) {
    // Sprawdzamy, czy obraz jest pusty
    if (src.empty()) {
        throw std::invalid_argument("Input image is empty.");
    }

    double aspectRatio = (double)src.cols / src.rows;
    int newWidth = maxWidth;
    int newHeight = (int)(maxWidth / aspectRatio);

    if (newHeight > maxHeight) {
        newHeight = maxHeight;
        newWidth = (int)(maxHeight * aspectRatio);
    }

    // Obliczanie współczynnika skali
    double scaleFactor = (double)newWidth / src.cols; // Proporcja szerokości

    // Tworzymy macierz do przechowywania przeskalowanego obrazu
    Mat resizedSrc;
    
    // Sprawdzamy, czy nowe wymiary są poprawne
    if (newWidth <= 0 || newHeight <= 0) {
        throw std::invalid_argument("Calculated image dimensions are invalid.");
    }

    // Skalujemy obraz do nowego rozmiaru
    resize(src, resizedSrc, Size(newWidth, newHeight));

    return resizedSrc;
}


// Funkcja do wykrywania krawędzi przy użyciu algorytmu Canny'ego
Mat detectEdges(const Mat& src, int lowThreshold, int highThreshold) {
    Mat edges;
    Canny(src, edges, lowThreshold, highThreshold, 3);
    return edges;
}

// Funkcja do wykrywania linii metodą probabilistyczną Hough Lines P
void detectProbabilisticHoughLines(const Mat& edges, int minLineLength, int maxLineGap) {
    // Używamy globalnej zmiennej linesP
    HoughLinesP(edges, linesP, 1, CV_PI / 180, 50, minLineLength, maxLineGap); // Parametry Hougha P
}

/*// Funkcja do rysowania linii metodą probabilistyczną
void drawProbabilisticLines(Mat& img) {
    // Sortowanie linii po kącie
    sort(linesP.begin(), linesP.end(), [&](const Vec4i& a, const Vec4i& b) {
        // Obliczanie kątów dla obu linii
        Point pt1a(a[0], a[1]), pt2a(a[2], a[3]);
        Point pt1b(b[0], b[1]), pt2b(b[2], b[3]);

        float angleA = atan2(pt2a.y - pt1a.y, pt2a.x - pt1a.x);
        float angleB = atan2(pt2b.y - pt1b.y, pt2b.x - pt1b.x);

        // Konwersja na stopnie
        angleA = angleA * 180 / CV_PI;
        angleB = angleB * 180 / CV_PI;

        return angleA < angleB; // Porównanie kątów
    });

    // Rysowanie posortowanych linii na obrazie
    for (size_t i = 0; i < linesP.size(); i++) {
        Vec4i l = linesP[i];
        Point pt1(l[0], l[1]);
        Point pt2(l[2], l[3]);

        // Obliczanie długości linii
        float length = sqrt(pow(pt2.x - pt1.x, 2) + pow(pt2.y - pt1.y, 2));

        // Obliczanie kąta linii (w radianach)
        float angle = atan2(pt2.y - pt1.y, pt2.x - pt1.x);
        angle = angle * 180 / CV_PI; // Konwersja na stopnie

        // Rysowanie linii na obrazie
        line(img, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);

        // Wypisanie informacji o linii
        cout << "Kąt: " << angle << "°"
             << ", Długość: " << length
             << ", Punkt początkowy: (" << pt1.x << ", " << pt1.y << ")"
             << ", Punkt końcowy: (" << pt2.x << ", " << pt2.y << ")"
             << endl;
    }
}*/

double calculateAngle(Point p1, Point p2) {
    // Obliczamy kąt pomiędzy punktami p1 (środek obrazu) a p2 (środek linii)
    double deltaX = p2.x - p1.x;
    double deltaY = p2.y - p1.y;
    double angle = atan2(deltaY, deltaX) * 180.0 / CV_PI;  // kąt w stopniach
    return angle < 0 ? angle + 360.0 : angle;  // Upewniamy się, że kąt mieści się w zakresie 0°-360°
}

// Funkcja obliczająca kąty dla linii
/*std::vector<std::pair<cv::Vec4i, double>> calculateLineAngles(const std::vector<cv::Vec4i>& linesP, const cv::Point& center) {
    std::vector<std::pair<cv::Vec4i, double>> lineAngles;

    for (size_t i = 0; i < linesP.size(); ++i) {
        cv::Vec4i line = linesP[i];
        cv::Point lineStart(line[0], line[1]);
        cv::Point lineEnd(line[2], line[3]);

        // Środek linii
        cv::Point lineCenter((lineStart.x + lineEnd.x) / 2, (lineStart.y + lineEnd.y) / 2);

        // Obliczamy kąt względem środka obrazu
        double angle = calculateAngle(center, lineCenter);

        // Przechowujemy linię i jej kąt
        lineAngles.push_back(std::make_pair(line, angle));
    }

    return lineAngles;
}*/

std::vector<std::pair<cv::Vec4i, double>> calculateLineAngles(const std::vector<cv::Vec4i>& linesP, const cv::Point& center) {
    std::vector<std::pair<cv::Vec4i, double>> lineAngles;

    for (size_t i = 0; i < linesP.size(); ++i) {
        cv::Vec4i line = linesP[i];
        cv::Point lineStart(line[0], line[1]);
        cv::Point lineEnd(line[2], line[3]);

        // Obliczamy odległość od środka obrazu do obu końców linii
        double distStart = cv::norm(lineStart - center);
        double distEnd = cv::norm(lineEnd - center);

        // Wybieramy dalszy koniec linii (ten, który jest dalej od środka)
        cv::Point furtherEnd = (distStart > distEnd) ? lineStart : lineEnd;

        // Obliczamy kąt względem dalszego końca
        double angle = calculateAngle(center, furtherEnd);

        // Przechowujemy linię i jej kąt
        lineAngles.push_back(std::make_pair(line, angle));
    }

    return lineAngles;
}


void filterLinesBasedOnAngleDifference(vector<Vec4i>& linesP, Mat& edges) {
    // Środek obrazu
    Point center(edges.cols / 2, edges.rows / 2);

    // Przechowujemy linie z ich kątami
    vector<pair<Vec4i, double>> lineAngles;

    // Obliczamy kąt dla każdej linii
    lineAngles = calculateLineAngles(linesP, center);

    // Sprawdzamy różnice kątów pomiędzy parami linii
    double minAngleDiff = std::numeric_limits<double>::infinity();  // Zmienna do przechowywania minimalnej różnicy
    size_t bestI = 0, bestJ = 0;  // Indeksy najlepszej pary

    for (size_t i = 0; i < lineAngles.size(); ++i) {
        for (size_t j = i + 1; j < lineAngles.size(); ++j) {
            double angleDiff = fabs(lineAngles[i].second - lineAngles[j].second);
            
            cout << i << " & " << j << ": " << angleDiff << " >= " << 180.0 - ANGLE_DIFF << " && " << angleDiff << " <= " << 180.0 + ANGLE_DIFF << endl;

            // Sprawdzamy, czy różnica kątów mieści się w przedziale 130°-230°
            if (angleDiff >= 180.0 - ANGLE_DIFF && angleDiff <= 180.0 + ANGLE_DIFF) {
                // Szukamy najmniejszej różnicy od 180°
                double diffFrom180 = fabs(angleDiff - 180.0);
                
                // Jeśli ta para jest bliższa 180° niż dotychczasowa, zapisz ją
                if (diffFrom180 < minAngleDiff) {
                    minAngleDiff = diffFrom180;
                    bestI = i;
                    bestJ = j;
                }
            }
        }
    }

    // Jeśli znaleziono najlepszą parę, zaktualizuj linesP
    if (minAngleDiff != std::numeric_limits<double>::infinity()) {
        
        cout << "Best: " << bestI << " & " << bestJ << ": " << minAngleDiff + 180 << " deg." << endl;

        linesP.clear();
        linesP.push_back(lineAngles[bestI].first);
        linesP.push_back(lineAngles[bestJ].first);
        return;
    }

    // Jeśli nie znaleziono pary, linesP będzie puste (lub możesz pozostawić je bez zmian)
    linesP.clear();  // Ewentualnie wyczyść, jeśli nie chcesz żadnej pary linii.
}

void removeEdgeCrop(Mat &edges) {
    // Sprawdzamy, czy CROP_IMAGE jest w odpowiednim zakresie
    if (CROP_SIZE < 0.0f || CROP_SIZE > 1.0f) {
        std::cerr << "CROP_SIZE musi być w zakresie [0, 1]" << std::endl;
        return;
    }
    // Obliczamy szerokość i wysokość obszaru do usunięcia na podstawie CROP_SIZE
    int cropWidth = static_cast<int>(edges.cols * CROP_SIZE);
    int cropHeight = static_cast<int>(edges.rows * CROP_SIZE);

    // Usuwamy krawędzie w lewym, prawym, górnym i dolnym marginesie obrazu
    // Ustawiamy piksele na 0 (czarne) w tych obszarach

    // Usuwamy krawędzie z lewej strony obrazu (wszystkie wiersze, od 0 do cropWidth)
    edges(Range(0, edges.rows), Range(0, cropWidth)) = Scalar(0);  // Lewy margines

    // Usuwamy krawędzie z prawej strony obrazu (wszystkie wiersze, od edges.cols - cropWidth do końca)
    edges(Range(0, edges.rows), Range(edges.cols - cropWidth, edges.cols)) = Scalar(0);  // Prawy margines

    // Usuwamy krawędzie z górnej strony obrazu (wszystkie kolumny, od 0 do cropHeight)
    edges(Range(0, cropHeight), Range(0, edges.cols)) = Scalar(0);  // Górny margines

    // Usuwamy krawędzie z dolnej strony obrazu (wszystkie kolumny, od edges.rows - cropHeight do końca)
    edges(Range(edges.rows - cropHeight, edges.rows), Range(0, edges.cols)) = Scalar(0);  // Dolny margines


}

void removeEdgeCropUpDown(Mat &edges) {
    // Sprawdzamy, czy obraz jest pusty
    if (edges.empty()) {
        std::cerr << "Obraz jest pusty!" << std::endl;
        return;
    }

    // Obliczamy wysokość, która będzie czyszczona w górnej i dolnej części obrazu
    int topHeight = edges.rows / 3;    // 1/3 wysokości obrazu
    int bottomHeight = edges.rows - topHeight; // Druga 1/3 na dole obrazu

    // Usuwamy krawędzie w górnej 1/3 obrazu (wszystkie kolumny, od 0 do topHeight)
    edges(Range(0, topHeight), Range(0, edges.cols)) = Scalar(0);  // Górny margines

    // Usuwamy krawędzie w dolnej 1/3 obrazu (wszystkie kolumny, od bottomHeight do końca)
    edges(Range(bottomHeight, edges.rows), Range(0, edges.cols)) = Scalar(0);  // Dolny margines
}


// Funkcja do obrócenia obrazu w taki sposób, by linie były poziome
Mat rotateImageToHorizontal(const string& imagePath) {
    // Wczytanie obrazu
    Mat image = imread(imagePath, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Nie udało się wczytać obrazu!" << endl;
        return Mat();
    }

    // Obliczanie średniego kąta linii
    float sumAngles = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        float rho = lines[i][0];
        float theta = lines[i][1];

        // Konwertujemy kąt na stopnie
        float angle = theta * 180 / CV_PI;
        sumAngles += angle;
    }

    // Średni kąt
    float averageAngle = sumAngles / lines.size();

    // Obracamy obraz, aby linie były poziome
    float rotationAngle = averageAngle - 90;  // Korygujemy kąt, aby linie były poziome
    Point2f center(image.cols / 2.0f, image.rows / 2.0f);  // Punkt obrotu (centrum obrazu)

    // Obliczanie nowego rozmiaru obrazu, aby pomieścić wszystkie piksele po obrocie
    Rect bbox = RotatedRect(center, image.size(), rotationAngle).boundingRect();

    // Zmieniamy macierz obrotu, aby zachować cały obraz w obrębie nowego rozmiaru
    Mat rotMatrix = getRotationMatrix2D(center, rotationAngle, 1.0);

    // Przesunięcie macierzy obrotu, by przesunąć obraz do nowego rozmiaru
    rotMatrix.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rotMatrix.at<double>(1, 2) += bbox.height / 2.0 - center.y;

    // Tworzymy pusty obraz o wymiarach odpowiadających nowemu prostokątnemu rozmiarowi
    Mat rotatedImage(bbox.size(), image.type());

    // Obracamy obraz z nowym rozmiarem, aby pomieścić całość
    warpAffine(image, rotatedImage, rotMatrix, bbox.size(), INTER_CUBIC);

    return rotatedImage;
}

void drawHoughLines(cv::Mat& src, const std::vector<cv::Vec2f>& lines)
{
    // Kopia obrazu wejściowego
    cv::Mat imgWithLines = src.clone();

    // Rysowanie linii na kopii obrazu
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0];
        float theta = lines[i][1];
        
        // Obliczanie punktów początkowego i końcowego na obrazie
        double cosTheta = cos(theta);
        double sinTheta = sin(theta);
        
        double x0 = rho * cosTheta;
        double y0 = rho * sinTheta;
        
        double x1 = cvRound(x0 + 1000 * (-sinTheta));
        double y1 = cvRound(y0 + 1000 * (cosTheta));
        double x2 = cvRound(x0 - 1000 * (-sinTheta));
        double y2 = cvRound(y0 - 1000 * (cosTheta));
        
        // Rysowanie linii na obrazie
        cv::line(imgWithLines, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    // Wyświetlanie obrazu z liniami
    cv::imshow("Lines Detected", resizeImage(imgWithLines));
    cv::waitKey(0);  // Czekanie na naciśnięcie dowolnego przycisku
}

// Funkcja do poszukiwania bazy - wykrywanie linii metodą Hougha P z iteracyjnym zwiększaniem maxLineGap
void baseFind(int, void*) {
    
    lines.clear();
    linesP.clear();
    
    // Wczytanie obrazu
    Mat src = loadImage(imagePath);
    if (src.empty()) {
        cout << "Error loading image" << endl;
        return; // Sprawdzanie, czy obraz został poprawnie wczytany
    }

    // Przycinanie obrazu
    //Mat croppedImage = cropImage(src);

    // Skalowanie obrazu
    //Mat resizedForCanny;
    //Mat resizedSrc = resizeImage(croppedImage);
    //Mat resizedSrc = croppedImage;
    //resize(croppedImage, resizedForCanny, Size(), SCALE_FACTOR, SCALE_FACTOR, INTER_LINEAR);
    
    //Mat blurredSrc;
    //GaussianBlur(resizedSrc, blurredSrc, Size(5, 5), 2);
    //blurredSrc = resizedSrc;
    
    // Wykrywanie krawędzi
    //Mat edges = detectEdges(resizedSrc, lowThreshold, highThreshold);
    Mat edges;
    //Canny(blurredSrc, edges, lowThreshold, highThreshold, 3);
    //Canny(resizedForCanny, edges, lowThreshold, highThreshold, 3);
    Canny(src, edges, lowThreshold, highThreshold, 3);
    //resize(edges, edges, croppedImage.size(), 0, 0, INTER_LINEAR);

    removeEdgeCrop(edges);
    
    //imshow("Canny", resizeImage(edges));
    
    // Obliczanie początkowych wartości minLineLength i maxLineGap
    int minLineLength = src.cols * RATIO_MIN_LENGTH;
    //int maxLineGap = src.cols * RATIO_MAX_GAP;   
    int maxLineGap = MAX_GAP;   
    int lineVoters = LINE_VOTERS;
    int it = 0;
        
    // Ustawienie początkowej wartości threshold
    //int threshold = 100;  // Możesz dostosować tę wartość w zależności od potrzeb

    while (true) {
        //vector<Vec2f> lines;
        // Wykrywanie linii metodą Hough Lines
        HoughLines(edges, lines, 1, CV_PI / 180, houghThreshold);  // Parametry Hougha

        // Wyświetlanie informacji o threshold i liczbie wykrytych linii
        cout << "[Stage 0] Threshold: " << houghThreshold << ", Liczba linii: " << lines.size() << endl;

        // Jeśli liczba wykrytych linii jest mniejsza niż 2, zmniejsz threshold o 10
        if (lines.size() > 0) {
            houghThreshold += 50;
        }
        // Jeśli dokładnie 2 linie zostały wykryte, przechodzimy dalej
        else {
            //threshold += 10;
            break;
        }

    }

    while (true) {
        //vector<Vec2f> lines;
        // Wykrywanie linii metodą Hough Lines
        HoughLines(edges, lines, 1, CV_PI / 180, houghThreshold);  // Parametry Hougha

        // Wyświetlanie informacji o threshold i liczbie wykrytych linii
        cout << "Threshold: " << houghThreshold << ", Liczba linii: " << lines.size() << endl;

        // Jeśli liczba wykrytych linii jest mniejsza niż 2, zmniejsz threshold o 10
        if (lines.size() < 2) {
            houghThreshold = max(houghThreshold - 1, 1);  // Zapewnia, że threshold nie będzie poniżej 1
        }
        else {
            //threshold += 10;
            break;
        }

    }

    while (true) {
        // Wykrywanie linii metodą Hough Lines
        HoughLines(edges, lines, 1, CV_PI / 180, houghThreshold);  // Parametry Hougha
        
        //drawHoughLines(src, lines);

        vector<Vec2f> validLines;
        validLines.clear();

        cout << "[Stage 2] Threshold: " << houghThreshold << ", lines: " << lines.size() << endl;

        if(lines.size() > 1){
            // Sprawdzamy każdą parę linii
            for (size_t i = 0; i < lines.size(); ++i) {
                if(validLines.size() > 0){
                    break;
                }
                for (size_t j = i + 1; j < lines.size(); ++j) {
                    // Sprawdzanie równoległości linii (kąt różni się o mniej niż 10 stopni)
                    float angle1 = lines[i][1];
                    float angle2 = lines[j][1];
                    float angleDiff = fabs(angle1 - angle2);


                    // Wyświetlanie informacji o threshold i liczbie wykrytych linii

                    // Jeśli kąty różnią się o więcej niż 10 stopni, linie nie są równoległe
    //                if (angleDiff > CV_PI / 18) {  // 10 stopni

    //                    cout << "   Kąt za duży: " << angleDiff  << "( > " << CV_PI / 18 << ")" << endl;
    //                    continue;
    //                }else{
    //                    //cout << "   Kąt ok: " << angleDiff  << "( > " << CV_PI / 18 << ")" << endl;
    //                }

                    // Obliczamy odległość między liniami (minimalna odległość względem szerokości obrazu)
                    float width = edges.cols;
                    float minDistance = width / 20.0f;

//                    // Punkt początkowy i końcowy dla obu linii
//                    Point pt1_1(cos(angle1) * lines[i][0] + lines[i][0], sin(angle1) * lines[i][0] + lines[i][1]);
//                    Point pt1_2(cos(angle1) * lines[i][0] + lines[i][0], sin(angle1) * lines[i][0] + lines[i][1]);
//                    Point pt2_1(cos(angle2) * lines[j][0] + lines[j][0], sin(angle2) * lines[j][0] + lines[j][1]);
//                    Point pt2_2(cos(angle2) * lines[j][0] + lines[j][0], sin(angle2) * lines[j][0] + lines[j][1]);

//                    // Obliczamy odległość między liniami
//                    float dist = sqrt(pow(pt1_1.x - pt2_1.x, 2) + pow(pt1_1.y - pt2_1.y, 2));

                    // Jeśli linie są równoległe, różnica w rho to odległość między nimi
                    float rho1 = lines[i][0];  // Pierwszy parametr rho
                    float rho2 = lines[j][0];  // Drugi parametr rho

                    // Odległość między prostymi równoległymi
                    float dist = fabs(rho1 - rho2);  // Zwracamy wartość bezwzględną różnicy

                    //cout << "[Stage 2] Threshold: " << threshold << ", dist: " << minDistance << endl;

                    // Jeśli odległość między liniami jest większa niż minimalna, linie są validne
                    if (dist >= minDistance && angleDiff < CV_PI / 18) {
                        cout << "  [OK!] Kąt ok: " << angleDiff  << "( < " << CV_PI / 18 << "), dist: " << dist << " > " << minDistance << endl;

                        validLines.push_back(lines[i]);
                        validLines.push_back(lines[j]);
                        break;
                    }else{
                        cout << "  Kąt: " << angleDiff  << "( > " << CV_PI / 18 << ") || dist: " << dist << " < " << minDistance << endl;

                        //cout << "[Stage 2] Threshold: " << threshold << ", dist: " << minDistance << endl;
                    }
                }
            }
        }

        // Jeśli znaleziono > 1 linie, przechodzimy dalej
        if (validLines.size() > 1) {
            lines = validLines;
            break;
        }

        // Jeśli linie nie spełniają warunków, zmniejszamy threshold o 1
        houghThreshold = max(houghThreshold - 1, 1);

        // Wyświetlanie informacji o threshold i liczbie wykrytych linii
        cout << "Threshold: " << houghThreshold << ", Liczba validnych linii: " << validLines.size() << endl;
    }

    // Narysowanie wykrytych linii na kopii obrazu
    Mat src_copy = src.clone();  // Tworzymy kopię obrazu, aby nie modyfikować oryginału
    for (size_t i = 0; i < lines.size(); ++i) {
        float rho = lines[i][0];
        float theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));

        // Rysowanie linii na kopii obrazu
        line(src_copy, pt1, pt2, Scalar(0, 0, 255), 2);  // Czerwony kolor linii
    }

    // Wyświetlenie obrazu z narysowanymi liniami
    //src_copy = resizeImage(src_copy);
    //imshow("Wykryte linie", src_copy);
    //waitKey(0);  // Czeka na naciśnięcie klawisza

    return;
}

/*// Funkcja do sprawdzania i obcinania obrazu
void trimImageFromLeft(Mat& img) {
    int firstNonWhiteCol = -1;

    // Przechodzimy przez każdą kolumnę obrazu
    for (int col = 0; col < img.cols; col++) {
        bool hasWhitePixel = false;

        // Sprawdzamy każdy piksel w kolumnie
        for (int row = 0; row < img.rows; row++) {
            if (img.at<Vec3b>(row, col) == Vec3b(255, 255, 255) || img.at<Vec3b>(row, col) == Vec3b(0, 0, 0)) {
                hasWhitePixel = true;
                break;  // Jeśli znajdziemy biały piksel, przerywamy sprawdzanie
            }
        }

        // Jeśli kolumna nie zawiera białych pikseli, zapiszemy jej numer
        if (!hasWhitePixel) {
            firstNonWhiteCol = col;
            break;  // Zatrzymujemy się po pierwszej kolumnie, która nie zawiera białych pikseli
        }
    }

    // Jeśli znajdziesz kolumnę bez białych pikseli, obcinamy obraz
    if (firstNonWhiteCol != -1) {
        // Wycinamy obraz od znalezionej kolumny do końca
        Rect rect(firstNonWhiteCol, 0, img.cols - firstNonWhiteCol, img.rows);
        cout << "Cut from " << firstNonWhiteCol << endl;
        img = img(rect);
    }
}

// Funkcja przetwarzająca zbiór obrazów
void trimImagesFromLeft(vector<Mat>& images) {
    // Dla każdego obrazu w zbiorze, obcinamy go
    for (size_t i = 0; i < images.size(); ++i) {
        trimImageFromLeft(images[i]);
        // Możesz dodać kod do zapisania lub dalszego przetwarzania obrazów
    }
}*/

// Funkcja do sprawdzania i obcinania obrazów z wektora
void trimImagesFromLeft(vector<Mat>& images) {
    // Przechodzimy przez każdy obraz w wektorze
    for (size_t i = 0; i < images.size(); i++) {
        Mat& img = images[i];  // Pobieramy referencję do obrazu
        int firstNonWhiteCol = -1;

        // Przechodzimy przez każdą kolumnę obrazu
        for (int col = 0; col < img.cols; col++) {
            bool hasWhitePixel = false;

            // Sprawdzamy każdy piksel w kolumnie
            for (int row = 0; row < img.rows; row++) {
                if (img.at<Vec3b>(row, col) == Vec3b(255, 255, 255) || img.at<Vec3b>(row, col) == Vec3b(0, 0, 0)) {
                    hasWhitePixel = true;
                    break;  // Jeśli znajdziemy biały piksel, przerywamy sprawdzanie
                }
            }

            // Jeśli kolumna nie zawiera białych pikseli, zapiszemy jej numer
            if (!hasWhitePixel) {
                firstNonWhiteCol = col;
                break;  // Zatrzymujemy się po pierwszej kolumnie, która nie zawiera białych pikseli
            }
        }

        // Jeśli znajdziesz kolumnę bez białych pikseli, obcinamy obraz
        if (firstNonWhiteCol != -1) {
            // Wycinamy obraz od znalezionej kolumny do końca
            Rect rect(firstNonWhiteCol, 0, img.cols - firstNonWhiteCol, img.rows);
            cout << "Cut from " << firstNonWhiteCol << " on image " << i << endl;
            img = img(rect);
        }
    }
}

std::vector<Mat> extractRectanglesContainingLines(const string& imagePath) {
    std::vector<Mat> croppedImages;  // Wektor do przechowywania wynikowych obrazów

    // Wczytanie obrazu w kolorze
    Mat src = imread(imagePath, IMREAD_COLOR);  // Wczytanie obrazu w kolorze
    if (src.empty()) {
        cout << "Błąd wczytywania obrazu!" << endl;
        return croppedImages;
    }

    // Obliczamy średni kąt wszystkich linii
    double totalAngle = 0;
    int lineCount = linesP.size();
    int x_mid = src.cols / 2;

    for (size_t i = 0; i < linesP.size(); i++) {
        Vec4i l = linesP[i];
        Point pt1(l[0], l[1]);
        Point pt2(l[2], l[3]);

        // Obliczamy kąt linii względem poziomu
        float angle = atan2(pt2.y - pt1.y, pt2.x - pt1.x) * 180.0 / CV_PI;
        totalAngle += angle;
    }

    // Oblicz średni kąt
    double averageAngle = totalAngle / lineCount;

    // Ustalmy środek obrazu do obrotu
    Point center(src.cols / 2, src.rows / 2);

    // Macierz obrotu
    Mat rotMat = getRotationMatrix2D(center, averageAngle, 1.0);  // Macierz obrotu

    // Obrócenie obrazu o średni kąt
    Mat rotatedImg;
    warpAffine(src, rotatedImg, rotMat, src.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255)); // Ustawienie tła na biały

    // Obracamy także współrzędne punktów w linesP
    for (size_t i = 0; i < linesP.size(); i++) {
        Vec4i l = linesP[i];
        Point pt1(l[0], l[1]);
        Point pt2(l[2], l[3]);

        std::vector<Point2f> points(2);
        points[0] = pt1;
        points[1] = pt2;
        std::vector<Point2f> rotatedPoints;
        transform(points, rotatedPoints, rotMat);

        // Zaktualizuj współrzędne linii po obrocie
        linesP[i] = Vec4i(rotatedPoints[0].x, rotatedPoints[0].y, rotatedPoints[1].x, rotatedPoints[1].y);
    }

    // Iteracja po każdej linii w linesP
    for (size_t i = 0; i < linesP.size(); i++) {
        Vec4i l = linesP[i];
        Point pt1(l[0], l[1]);
        Point pt2(l[2], l[3]);

        // Obliczamy punkt dalej od środka
        Point furtherPoint = (abs(pt1.x - x_mid) > abs(pt2.x - x_mid)) ? pt1 : pt2;

        // Dodajemy marginesy 5% wysokości nad i pod linią
        int marginY = rotatedImg.rows * 0.05;

        int x_min, x_max, y_min, y_max;

        if (furtherPoint.x > x_mid) {
            // Linia jest po prawej stronie
            x_min = x_mid;
            x_max = rotatedImg.cols;
        } else {
            // Linia jest po lewej stronie
            x_min = 0;
            x_max = x_mid;
        }

        // Ustalanie wysokości prostokąta
        y_min = max(0, furtherPoint.y - marginY);
        y_max = min(rotatedImg.rows, furtherPoint.y + marginY);

        // Tworzenie prostokąta
        Rect rectangle(x_min, y_min, x_max - x_min, y_max - y_min);
        Mat croppedImage = rotatedImg(rectangle);

        // Usuwanie kolumn zawierających białe/czarne piksele z obrazu
        //trimImageFromLeft(croppedImage);

        // Jeśli linia była po prawej stronie, obracamy wycięty obraz o 180 stopni
        if (furtherPoint.x > x_mid) {
            //Mat flippedImage;
            rotate(croppedImage, croppedImage, ROTATE_180);  // Obrót o 180 stopni

            //flip(croppedImage, flippedImage, -1);  // Obrót o 180 stopni
            //cout << "l";
            //croppedImages.push_back(croppedImage);
        }/* else {
            //cout << "p";
            
        }*/
        croppedImages.push_back(croppedImage);

        // Opcjonalnie: wyświetlanie wyciętego obrazu
        imshow("Cropped Rectangle " + to_string(i), croppedImage);
    }
    
        cout << endl;

    // Zwrócenie wektora obrazów
    return croppedImages;
}



// Funkcja do przekształcania współrzędnych punktów początkowych i końcowych linii
void transformLineCoordinates() {
    // Iteracja po każdej linii w linesP
    for (size_t i = 0; i < linesP.size(); i++) {
        // Pobieramy współrzędne punktów końcowych linii
        Vec4i& l = linesP[i];

        // Przeskalowanie współrzędnych punktów (mnożenie przez odwrotność scaleFactor)
        l[0] = static_cast<int>(l[0] / scaleFactor);  // Przeskalowanie x1
        l[1] = static_cast<int>(l[1] / scaleFactor);  // Przeskalowanie y1
        l[2] = static_cast<int>(l[2] / scaleFactor);  // Przeskalowanie x2
        l[3] = static_cast<int>(l[3] / scaleFactor);  // Przeskalowanie y2

        // Przesunięcie punktów o cropWidthPx i cropHeightPx
        l[0] += cropWidthPx;  // Przesunięcie x1
        l[1] += cropHeightPx;  // Przesunięcie y1
        l[2] += cropWidthPx;  // Przesunięcie x2
        l[3] += cropHeightPx;  // Przesunięcie y2
    }
}

// Mat increaseContrastAndBrightness(const Mat& src, double alpha = 1.5, int beta = 50) {
//     Mat dst;
//     src.convertTo(dst, -1, alpha, beta);  // alpha - współczynnik kontrastu, beta - wartość jasności
//     return dst;
// }
// 
// Mat histogramEqualization(const Mat& src) {
//     Mat dst;
//     equalizeHist(src, dst);
//     return dst;
// }
// 
// Mat sharpenEdges(const Mat& src) {
//     Mat laplacian;
//     Laplacian(src, laplacian, CV_8U, 3);  // Filtr Laplacjana
//     Mat sharpened = src - laplacian;      // Wzmacnianie krawędzi przez odjęcie Laplacjanu
//     return sharpened;
// }
// 
// Mat morphologicallyEnhance(const Mat& src) {
//     Mat dilated;
//     dilate(src, dilated, Mat());  // Dylacja krawędzi
//     return dilated;
// }

/*vector<Mat> edgeDetectionSobel(const vector<Mat>& croppedImages) {
    vector<Mat> processedImages;  // Wektor do przechowywania wynikowych obrazów

    // Iteracja po każdym obrazie w wektorze croppedImages
    for (size_t i = 0; i < croppedImages.size(); i++) {
        // Wykrywanie krawędzi Sobelem w kierunku pionowym (Gx = 0, Gy = 1)
        Mat sobelEdges;
        Sobel(croppedImages[i], sobelEdges, CV_8U, 0, 1, 5);  // Zwiększenie rozmiaru jądra do 5 (silniejsze detekcje)

        // Możemy wykonać dodatkowe skalowanie wyniku Sobela, aby uwydatnić krawędzie
        Mat sobelEdgesScaled;
        sobelEdges.convertTo(sobelEdgesScaled, -1, 2.0, 0);  // Skalowanie wyniku Sobela (zwiększenie intensywności)

        // Przekształcanie krawędzi do formy binarnej (0 lub 255) z mniejszym progiem
        Mat binaryEdges;
        threshold(sobelEdgesScaled, binaryEdges, 30, 255, THRESH_BINARY);  // Zmniejszenie progu (bardziej wrażliwy)

        // Kopiowanie wyciętego obrazu, aby nanosić na niego krawędzie
        Mat imageWithEdges = croppedImages[i].clone();

        // Sprawdzamy, czy obraz jest w odcieniach szarości (jedno-kanalowy)
        if (imageWithEdges.channels() == 1) {
            // Iteracja po pikselach wykrytych krawędzi (obrazy szaro-skalowe)
            for (int y = 0; y < binaryEdges.rows; y++) {
                for (int x = 0; x < binaryEdges.cols; x++) {
                    if (binaryEdges.at<uchar>(y, x) == 255) {
                        // Ustawiamy piksel na biały (255) tam, gdzie jest krawędź
                        imageWithEdges.at<uchar>(y, x) = 255;
                    }
                }
            }
        } else {
            // Iteracja po pikselach wykrytych krawędzi (obrazy kolorowe)
            for (int y = 0; y < binaryEdges.rows; y++) {
                for (int x = 0; x < binaryEdges.cols; x++) {
                    if (binaryEdges.at<uchar>(y, x) == 255) {
                        // Nanosi czerwoną linię na wykryte krawędzie w obrazie kolorowym
                        imageWithEdges.at<Vec3b>(y, x) = Vec3b(0, 0, 255);  // Czerwony kolor (BGR)
                    }
                }
            }
        }

        // Dodajemy obraz z nałożonymi krawędziami do wynikowego wektora
        processedImages.push_back(imageWithEdges);

        // Wyświetlanie obrazu z nałożonymi krawędziami
        imshow("Image with Edges " + to_string(i), imageWithEdges);
    }

    // Zwrócenie wektora obrazów z nałożonymi krawędziami
    return processedImages;
}*/

/*
void saveCroppedImages(const vector<Mat>& croppedImages, const string& imagePath) {
    // Sprawdzamy, czy wektor z wyciętymi obrazami nie jest pusty
    if (croppedImages.empty()) {
        std::cerr << "Brak wyciętych obrazów do zapisania!" << std::endl;
        return;
    }

    // Pobieramy ścieżkę do folderu, w którym znajduje się obraz wejściowy
    fs::path inputImagePath(imagePath);
    fs::path outputDirectory = inputImagePath.parent_path();  // Ścieżka do folderu z plikiem wejściowym

    // Zapisujemy każdy obraz w croppedImages do pliku
    for (size_t i = 0; i < croppedImages.size(); i++) {
        // Tworzymy nazwę pliku: "edge_1.png", "edge_2.png", itd.
        string outputFileName = "edge_" + std::to_string(i + 1) + ".png";
        fs::path outputPath = outputDirectory / outputFileName;  // Ścieżka do pliku wyjściowego

        // Zapisujemy obraz do pliku PNG
        imwrite(outputPath.string(), croppedImages[i]);
        std::cout << "Zapisano obraz: " << outputPath.string() << std::endl;
    }
}*/

// Szablon funkcji cutEdge, którą później zaimplementujemy
Mat cutEdge(const Mat& image) {
    Vec2f line;  // Zmienna, która będzie przechowywać wybraną linię

    if (image.empty()) {
        cerr << "Pusty obraz" << endl;
        return image;  // Zwracamy pusty obraz, jeśli obraz jest pusty
    }

    // Tworzymy obraz do detekcji krawędzi
    Mat edges;
    Canny(image, edges, 50, 150, 3);  // Wykrywanie krawędzi za pomocą Canny'ego

    removeEdgeCrop(edges);
    removeEdgeCropUpDown(edges);

    //imshow("Lewa połowa", resizeImage(edges));
    //waitKey(0);

    // Zmienna na przechowanie wykrytych linii
    vector<Vec2f> lines;
    //int threshold = 100;  // Początkowa wartość progu głosów
    houghThreshold += 100;

    // Pierwsza faza: iteracyjne zwiększanie progu, aby znaleźć jakąś linię
//    while (true) {
//        // Wykrywanie linii metodą Hougha
//        HoughLines(edges, lines, 1, CV_PI / 180, houghThreshold);  // Parametry Hougha

//        // Jeśli znaleziono jakąś linię, zwiększamy próg o 50
//        if (lines.size() > 0) {
//            houghThreshold += 50;
//        } else {
//            break;  // Przerywamy, gdy już nie ma wykrytych linii
//        }
//    }

    int gotIt = 0;

    // Druga faza: iteracyjne zmniejszanie progu, aż znajdziemy dokładnie jedną linię
    while (true) {
        // Wykrywanie linii metodą Hougha
        HoughLines(edges, lines, 1, CV_PI / 180, houghThreshold);  // Parametry Hougha

        cout << "[Halfs] lines: " << lines.size() << ", threshold: " << houghThreshold << endl;

        // Iteracja po wszystkich wykrytych liniach
        for (size_t i = 0; i < lines.size(); ++i) {
            float angle = lines[i][1] * 180 / CV_PI;  // Przemiana kąta w radianach na stopnie
            //cout << "   Angle: " << angle << endl;

            // Sprawdzanie, czy kąt mieści się w przedziale ±10°
            if (angle > 85 && angle < 95) {
                // Zapisanie linii, jeśli kąt mieści się w przedziale
                line = lines[i];
                gotIt = 1;

                cout << "   Angle: " << angle << endl;

                break;  // Jeśli chcesz zatrzymać szukanie po znalezieniu pierwszej linii w tym przedziale
            }
        }

        if(gotIt){
            break;
        }

        // Zmniejszamy próg o 1, jeśli mamy więcej niż 1 linię
        houghThreshold = max(houghThreshold - 1, 1);  // Zapewnia, że próg nie będzie poniżej 1
    }


//        // Parametry linii (rho, theta)
//        float rho = lines[0][0];
//        float theta = lines[0][1];

//        // Obliczamy punkt początkowy i końcowy linii
//        double a = cos(theta), b = sin(theta);
//        double x0 = a * rho, y0 = b * rho;
//        Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
//        Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));

//        // Rysowanie wykrytej linii na obrazie
//        Mat imageWithLine = image.clone();
//        line(imageWithLine, pt1, pt2, Scalar(0, 0, 255), 2);  // Czerwona linia

//        // Wyświetlamy obraz z wykrytą linią
//        imshow("Wykryta pozioma linia", imageWithLine);

//        // Wycinamy obszar obrazu wokół tej linii (np. 100 pikseli powyżej i poniżej)
//        int margin = 100;  // Margines wokół linii
//        int cutHeight = pt2.y - pt1.y + margin;  // Wysokość wycinanego obszaru
//        int cutTop = max(pt1.y - margin, 0);   // Górna granica wycinanego obszaru
//        int cutBottom = min(pt2.y + margin, image.rows); // Dolna granica wycinanego obszaru

//        // Wycinek obrazu
//        Rect roi(0, cutTop, image.cols, cutHeight);
//        Mat croppedImage = image(roi);

        //return image;
//        return croppedImage;  // Zwracamy wycięty fragment obrazu

    // Parametry obrazu
    int imageHeight = image.rows;
    int imageWidth = image.cols;

    // Parametry linii
    float rho = line[0];
    float theta = line[1];

    // Obliczenie punktów startowych i końcowych linii
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;

    // Oblicz punkty, na których linia przecina obraz
    Point pt1, pt2;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));

    // Sprawdzamy, na jakiej wysokości obrazu znajduje się linia (zwykle bierzemy środek)
    int lineY = cvRound(y0);  // Y to punkt, w którym linia przecina oś Y

    // Obliczamy zakres w pionie, uwzględniając ±10% wysokości obrazu
    int top = std::max(0, cvRound(lineY - 0.05 * imageHeight));  // Top boundary (nie może być mniejsze niż 0)
    int bottom = std::min(imageHeight, cvRound(lineY + 0.05 * imageHeight));  // Bottom boundary (nie może być większe niż wysokość obrazu)

    cout << "ROI: 0, " << top << ", " << imageWidth << ", " << bottom - top << endl;

    // Wycinamy obraz na podstawie obliczonych granic
    Rect roi(0, top, imageWidth, bottom - top);
    Mat cutImage = image(roi);  // Wycinamy odpowiednią część obrazu

    return cutImage;

}


// Funkcja do podziału obrazu na dwie części i wycięcia
vector<Mat> splitAndCutImage(const Mat& rotatedImage) {
    // Sprawdzenie, czy obraz jest pusty
    if (rotatedImage.empty()) {
        cerr << "Pusty obraz" << endl;
        return {};
    }

    // Dzielimy obraz na dwie równe części wzdłuż osi pionowej
    int width = rotatedImage.cols;
    int height = rotatedImage.rows;

    // Podział obrazu na dwie części (lewa i prawa połowa)
    Mat leftHalf = rotatedImage(Rect(0, 0, width / 2, height));
    Mat rightHalf = rotatedImage(Rect(width / 2, 0, width / 2, height));

    // Wycinamy oba obrazy używając funkcji cutEdge (jeszcze do zaimplementowania)
    Mat leftProcessed = cutEdge(leftHalf);
    Mat rightProcessed = cutEdge(rightHalf);

    // Zwracamy przetworzone obrazy
    return {leftProcessed, rightProcessed};
}

// Funkcja zapisująca obrazy z wektora do plików PNG
void saveCroppedImages(const vector<Mat>& croppedImages, const string& imagePath) {
    // Ustalamy katalog z pliku wejściowego
    fs::path inputPath(imagePath);
    string directory = inputPath.parent_path().string();

    // Ustalamy nazwę pliku bazowego (nazwa bez rozszerzenia)
    string baseFileName = inputPath.filename().string();
    string baseName = baseFileName.substr(0, baseFileName.find_last_of('.'));

    // Iteracja po wektorze obrazów i zapisanie każdego z nich
    for (size_t i = 0; i < croppedImages.size(); ++i) {
        // Tworzenie pełnej ścieżki do pliku (np. edge_1.png, edge_2.png)
        stringstream ss;
        ss << directory << "/" << baseName << "_edge_" << i + 1 << ".png";
        string outputFileName = ss.str();

        // Zapisanie obrazu
        bool result = imwrite(outputFileName, croppedImages[i]);
        if (result) {
            cout << "Obraz zapisany jako: " << outputFileName << endl;
        } else {
            cout << "Błąd zapisu obrazu: " << outputFileName << endl;
        }
    }
}

int main(int argc, char** argv) {
    // Sprawdzenie, czy został przekazany argument (ścieżka do obrazu)
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <image_path> [image_path2 ...]" << endl;
        return -1;
    }

    for (int i = 1; i < argc; ++i) {
        //string imagePath = argv[i];
        houghThreshold = THRESHOLD;

        // Odczytanie ścieżki do obrazu z argumentu
        imagePath = argv[i];

        // Wywołanie funkcji poszukującej linii
        baseFind(0, 0);

        // Obrócenie obrazu
        Mat rotatedImage = rotateImageToHorizontal(imagePath);

        // Podzielamy obraz na dwie części i wycinamy
        vector<Mat> processedImages = splitAndCutImage(rotatedImage);

        rotate(processedImages[1], processedImages[1], ROTATE_180);  // Obrót o 180 stopni
        
        trimImagesFromLeft(processedImages);

        // Wyświetlamy obrazy
//         if (processedImages.size() >= 2) {
//             imshow("Lewa połowa", resizeImage(processedImages[0]));
//             imshow("Prawa połowa", resizeImage(processedImages[1]));
//         }
        
        // Zapisanie obrazów do plików
        saveCroppedImages(processedImages, imagePath);


        //imshow("Wypoziomowany obraz", resizeImage(rotatedImage));
        //waitKey(0);  // Czeka na naciśnięcie klawisza
        
        // Modyfikacja współrzędnych linii (przeskalowanie przez 2 i przesunięcie o 10px)
        //transformLineCoordinates();

        // Wczytanie obrazu ponownie do ekstrakcji prostokątów
        //Mat src = loadImage(imagePath);
        //if (src.empty()) return -1;

        // Wywołanie funkcji do wycinania prostokątów
        //std::vector<Mat> croppedImages = extractRectanglesContainingLines(src);
        //vector<Mat> croppedImages = extractRectanglesContainingLines(imagePath);

        // Zapisanie obrazów do plików
        //saveCroppedImages(croppedImages, imagePath);
        
        // Wywołanie funkcji rozmywania i detekcji krawędzi
        //vector<Mat> processedImages = edgeDetectionSobel(croppedImages);

        // Zapisanie obrazów do plików
        //saveCroppedImages(croppedImages, imagePath);
        
        // Wywołanie funkcji wstępnej
        waitKey(10);
    }
        
    return 0;
}
