/*
 * g++ -o image_analysis image_analysis.cpp `pkg-config --cflags --libs opencv4`
 * 
 * 
 */ 

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

using namespace cv;
using namespace std;

// Parametry dla algorytmu Canny'ego
#define CANNY_THRESH1 100
#define CANNY_THRESH2 200

// Deklaracja globalnych zmiennych
Mat srcImage;          // Przechowuje oryginalny obraz
int sliderPosition = 0; // Pozycja suwaka

// Funkcja do rysowania wykresów
void drawGraphs(Mat& plot, const vector<int>& brightness, const vector<int>& histR, const vector<int>& histG, const vector<int>& histB) {
    plot.setTo(Scalar(255, 255, 255));  // Białe tło dla wykresów

    // Wysokość i szerokość dla wykresów
    int histHeight = plot.rows / 3;

    // Rysowanie wykresu jasności
    int maxBrightness = *max_element(brightness.begin(), brightness.end());
    for (int i = 0; i < brightness.size() - 1; i++) {
        line(plot, Point(i, histHeight - brightness[i] * histHeight / maxBrightness),
            Point(i + 1, histHeight - brightness[i + 1] * histHeight / maxBrightness),
            Scalar(0, 0, 255), 2); // Czerwony kolor dla jasności
    }

    // Rysowanie histogramów kolorów RGB
    int maxHistR = *max_element(histR.begin(), histR.end());
    int maxHistG = *max_element(histG.begin(), histG.end());
    int maxHistB = *max_element(histB.begin(), histB.end());

    // Histogram czerwony (R)
    for (int i = 0; i < histR.size() - 1; i++) {
        line(plot, Point(i + plot.cols / 3, histHeight - histR[i] * histHeight / maxHistR),
            Point(i + plot.cols / 3 + 1, histHeight - histR[i + 1] * histHeight / maxHistR),
            Scalar(0, 0, 255), 2);
    }

    // Histogram zielony (G)
    for (int i = 0; i < histG.size() - 1; i++) {
        line(plot, Point(i + 2 * plot.cols / 3, 2 * histHeight - histG[i] * histHeight / maxHistG),
            Point(i + 2 * plot.cols / 3 + 1, 2 * histHeight - histG[i + 1] * histHeight / maxHistG),
            Scalar(0, 255, 0), 2);
    }

    // Histogram niebieski (B)
    for (int i = 0; i < histB.size() - 1; i++) {
        line(plot, Point(i + plot.cols / 3, 2 * histHeight - histB[i] * histHeight / maxHistB),
            Point(i + plot.cols / 3 + 1, 2 * histHeight - histB[i + 1] * histHeight / maxHistB),
            Scalar(255, 0, 0), 2);
    }

    // Dodanie podpisów
    putText(plot, "Brightness", Point(plot.cols / 6, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
    putText(plot, "Red Histogram", Point(plot.cols / 2.8, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
    putText(plot, "Green Histogram", Point(plot.cols / 1.7, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
    putText(plot, "Blue Histogram", Point(plot.cols / 1.7, histHeight + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
}

// Funkcja do obliczania wykresów pozycjonujących suwaki
void processSlider(int, void*) {
    // Zbieramy dane z wybranej kolumny obrazu
    vector<int> brightness;
    vector<int> histR(256, 0);
    vector<int> histG(256, 0);
    vector<int> histB(256, 0);

    // Przechodzimy po wszystkich pikselach w wybranej kolumnie
    for (int y = 0; y < srcImage.rows; y++) {
        Vec3b pixel = srcImage.at<Vec3b>(y, sliderPosition);

        // Jasność (średnia z kanałów RGB)
        int avgBrightness = (pixel[0] + pixel[1] + pixel[2]) / 3;
        brightness.push_back(avgBrightness);

        // Histogramy kolorów
        histB[pixel[0]]++;
        histG[pixel[1]]++;
        histR[pixel[2]]++;
    }

    // Tworzymy obraz do wykresów
    Mat plot = Mat::zeros(600, 1200, CV_8UC3);  // Szerokość większa, bo wykresy obok siebie

    // Rysujemy wykresy
    drawGraphs(plot, brightness, histR, histG, histB);

    // Wyświetlamy wykres
    imshow("Graph", plot);
}

// Funkcja główna
int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    // Wczytanie obrazu
    srcImage = imread(argv[1], IMREAD_COLOR);
    if (srcImage.empty()) {
        cout << "Nie udało się wczytać obrazu!" << endl;
        return -1;
    }

    // Dopasowanie szerokości okna do szerokości obrazu
    namedWindow("Image", WINDOW_NORMAL);
    resizeWindow("Image", srcImage.cols, srcImage.rows);  // Dopasowanie rozmiaru okna

    // Przeprowadzenie detekcji Canny'ego
    Mat edges;
    Canny(srcImage, edges, CANNY_THRESH1, CANNY_THRESH2);

    // Tworzymy obraz do wyświetlenia Canny'ego
    Mat cannyImage = Mat::zeros(srcImage.size(), CV_8UC3);
    srcImage.copyTo(cannyImage, edges);  // Kopiujemy oryginalny obraz tylko na krawędzie

    // Dodanie suwaka do wyboru kolumny (bez etykiety)
    int maxSliderPos = srcImage.cols - 1;
    createTrackbar(" ", "Image", &sliderPosition, maxSliderPos, processSlider);  // Pusty string zamiast etykiety

    // Ustawienie suwaka na połowie szerokości obrazu
    sliderPosition = srcImage.cols / 2;
    processSlider(0, 0);  // Inicjalizacja wykresów po ustawieniu suwaka

    // Wyświetlanie obrazu i Canny'ego
    imshow("Image", srcImage);      // Oryginalny obraz
    imshow("Canny Edges", cannyImage); // Obraz z nałożonymi krawędziami Canny'ego

    // Czekanie na naciśnięcie klawisza
    waitKey(0);
    return 0;
}
