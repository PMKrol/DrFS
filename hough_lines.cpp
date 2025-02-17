/*
 * 
 g++ -o hough_lines hough_lines.cpp `pkg-config --cflags --libs opencv4`
 * 
 */




#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

// Parametry Canny'ego i Hougha
int lowThreshold = 255;
int highThreshold = 255;
int minLineLength = 200;
int maxLineGap = 10;

// Zmienna przechowująca ścieżkę do obrazu
string imagePath;

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
    int cropWidth = width * 0.025;
    int cropHeight = height * 0.025;

    // Przycinanie obrazu: lewa, prawa, góra, dół
    Rect cropRegion(cropWidth, cropHeight, width - 2 * cropWidth, height - 2 * cropHeight);
    return src(cropRegion);
}

// Funkcja do skalowania obrazu do określonego rozmiaru
Mat resizeImage(const Mat& src, int maxWidth = 1280, int maxHeight = 800) {
    double aspectRatio = (double)src.cols / src.rows;
    int newWidth = maxWidth;
    int newHeight = (int)(maxWidth / aspectRatio);

    if (newHeight > maxHeight) {
        newHeight = maxHeight;
        newWidth = (int)(maxHeight * aspectRatio);
    }

    Mat resizedSrc;
    resize(src, resizedSrc, Size(newWidth, newHeight));
    return resizedSrc;
}

// Funkcja do wykrywania krawędzi przy użyciu algorytmu Canny'ego
Mat detectEdges(const Mat& src, int lowThreshold, int highThreshold) {
    Mat edges;
    Canny(src, edges, lowThreshold, highThreshold, 3);
    return edges;
}

// Struktura do przechowywania informacji o liniach
struct LineInfo {
    float angle;       // Kąt
    float length;      // Długość
    Point startPoint;  // Punkt początkowy
    Point endPoint;    // Punkt końcowy
};

// Funkcja do wykrywania linii metodą Hough Lines
vector<LineInfo> detectHoughLines(const Mat& edges) {
    vector<Vec2f> lines;
    HoughLines(edges, lines, 1, CV_PI / 180, 150, 0, 0); // Parametry Hougha mogą być zmieniane przez suwaki

    vector<LineInfo> lineInfos;
    // Obliczanie kąta i długości dla każdej linii
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));

        // Obliczanie kąta i długości
        float angle = theta;
        float length = sqrt(pow(pt2.x - pt1.x, 2) + pow(pt2.y - pt1.y, 2));

        // Zapisz informacje o linii
        lineInfos.push_back({angle, length, pt1, pt2});
    }

    // Sortowanie linii po kącie
    sort(lineInfos.begin(), lineInfos.end(), [](const LineInfo& a, const LineInfo& b) {
        return a.angle < b.angle;
    });

    return lineInfos;
}

// Funkcja do wykrywania linii metodą probabilistyczną Hough Lines P
vector<Vec4i> detectProbabilisticHoughLines(const Mat& edges, int minLineLength, int maxLineGap) {
    vector<Vec4i> linesP;
    HoughLinesP(edges, linesP, 1, CV_PI / 180, 50, minLineLength, maxLineGap); // Parametry Hougha P
    return linesP;
}

// Funkcja do rysowania linii na obrazie
void drawLines(Mat& img, const vector<LineInfo>& lineInfos) {
    for (size_t i = 0; i < lineInfos.size(); i++) {
        LineInfo line = lineInfos[i];
        
        // Rysowanie linii na obrazie
        cv::line(img, line.startPoint, line.endPoint, Scalar(0, 0, 255), 3, LINE_AA);

        // Wypisanie informacji o linii
        cout << "Kąt: " << line.angle * 180 / CV_PI << "°"
            << ", Długość: " << line.length
            << ", Punkt początkowy: (" << line.startPoint.x << ", " << line.startPoint.y << ")"
            << ", Punkt końcowy: (" << line.endPoint.x << ", " << line.endPoint.y << ")"
            << endl;
    }
}

// Funkcja do rysowania linii metodą probabilistyczną, sortowania po kącie i wypisywania informacji o liniach
void drawProbabilisticLines(Mat& img, vector<Vec4i>& linesP) {
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
}


// Główna funkcja do wykrywania krawędzi i linii
void detectLines(int, void*) {
    // Wczytanie obrazu
    Mat src = loadImage(imagePath);
    if (src.empty()) return; // Sprawdzanie, czy obraz został poprawnie wczytany

    // Przycinanie obrazu
    Mat croppedImage = cropImage(src);

    // Skalowanie obrazu
    Mat resizedSrc = resizeImage(croppedImage);

    // Wykrywanie krawędzi
    Mat edges = detectEdges(resizedSrc, lowThreshold, highThreshold);

    // Konwersja do obrazu BGR, aby rysować linie
    Mat cdst;
    cvtColor(edges, cdst, COLOR_GRAY2BGR);
    Mat cdstP = cdst.clone();

    // Wykrywanie linii metodą Hough Lines
    //vector<LineInfo> lineInfos = detectHoughLines(edges);

    // Rysowanie linii na obrazie
    //drawLines(cdst, lineInfos);

    // Wykrywanie linii metodą probabilistyczną Hough Lines P
    vector<Vec4i> linesP = detectProbabilisticHoughLines(edges, minLineLength, maxLineGap);

    // Rysowanie linii probabilistycznych
    drawProbabilisticLines(cdstP, linesP);

    // Wyświetlanie wyników
    imshow("Source", resizedSrc);
    //imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
    imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
}

// Funkcja do poszukiwania bazy - wykrywanie linii metodą Hougha P z iteracyjnym zwiększaniem maxLineGap
void baseFind(int, void*) {
    // Wczytanie obrazu
    Mat src = loadImage(imagePath);
    if (src.empty()) {
        cout << "Error loading image" << endl;
        return; // Sprawdzanie, czy obraz został poprawnie wczytany
    }

    // Przycinanie obrazu
    Mat croppedImage = cropImage(src);

    // Skalowanie obrazu
    Mat resizedSrc = resizeImage(croppedImage);

    // Ustawienia początkowe
//     int lowThreshold = 255;
//     int highThreshold = 255;
//     int minLineLength = resizedSrc.cols / 4; // minLineLength to 1/4 szerokości obrazu
//     int maxLineGap = 10; // Początkowa wartość maxLineGap

    // Wykrywanie krawędzi
    Mat edges = detectEdges(resizedSrc, lowThreshold, highThreshold);

    // Inicjalizacja zmiennej do przechowywania wykrytych linii
    vector<Vec4i> linesP;

    // Iteracyjne zwiększanie maxLineGap aż do wykrycia dwóch linii
    while (linesP.size() < 2) {
        // Wykrywanie linii metodą probabilistyczną Hough Lines P
        linesP = detectProbabilisticHoughLines(edges, minLineLength, maxLineGap);

        // Zwiększanie maxLineGap o 1
        maxLineGap++;
    }

    // Wypisanie wyników
    cout << "Znaleziono " << linesP.size() << " linii." << endl;
    for (size_t i = 0; i < linesP.size(); i++) {
        Vec4i l = linesP[i];
        Point pt1(l[0], l[1]);
        Point pt2(l[2], l[3]);

        // Obliczanie długości linii
        float length = sqrt(pow(pt2.x - pt1.x, 2) + pow(pt2.y - pt1.y, 2));

        // Obliczanie kąta linii (w radianach)
        float angle = atan2(pt2.y - pt1.y, pt2.x - pt1.x);
        angle = angle * 180 / CV_PI; // Konwersja na stopnie

        // Wypisanie szczegółów linii
        cout << "Linia " << i+1 << ":" << endl;
        cout << "  Kąt: " << angle << "°" << endl;
        cout << "  Długość: " << length << endl;
        cout << "  Punkt początkowy: (" << pt1.x << ", " << pt1.y << ")" << endl;
        cout << "  Punkt końcowy: (" << pt2.x << ", " << pt2.y << ")" << endl;
    }
}

int main(int argc, char** argv) {
    // Sprawdzenie, czy został przekazany argument (ścieżka do obrazu)
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    // Odczytanie ścieżki do obrazu z argumentu
    imagePath = argv[1];

    baseFind(0, 0);
    
    // Utworzenie okna do wyświetlania wyników
    namedWindow("Detected Lines", WINDOW_NORMAL);

    // Tworzenie suwaków do zmiany parametrów
    createTrackbar("Low Threshold", "Detected Lines", &lowThreshold, 255, detectLines);
    createTrackbar("High Threshold", "Detected Lines", &highThreshold, 255, detectLines);
    createTrackbar("Min Line Length", "Detected Lines", &minLineLength, 200, detectLines);
    createTrackbar("Max Line Gap", "Detected Lines", &maxLineGap, 100, detectLines);

    // Wywołanie funkcji wstępnej
    detectLines(0, 0);

    // Czekanie na naciśnięcie klawisza, aby zakończyć
    waitKey(0);
    return 0;
}
