/*
 * compile: g++ -std=c++17 -o canny_hough_slider canny_hough_slider.cpp `pkg-config --cflags --libs opencv4`
 * run: ./canny_hough_slider /sciezka/do/katalogu
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int lowThreshold = 50;    // Początkowa wartość progu dolnego
int highThreshold = 150;  // Początkowa wartość progu górnego
const int maxThreshold = 300;
int lineVoters = 50;      // Początkowa wartość dla liczby głosów w HoughLinesP
int minLineLength = 50;   // Początkowa wartość minimalnej długości linii
int maxLineGap = 10;      // Początkowa wartość maksymalnego odstępu między liniami

Mat imgOriginal, imgGray, imgEdges, imgResized, imgDisplay;
vector<Vec4i> linesP;

void applyCanny(int, void*) {
    // Zastosowanie detekcji krawędzi Canny na obrazku
    Canny(imgGray, imgEdges, lowThreshold, highThreshold);

    // Konwersja do BGR, aby móc nałożyć tekst
    cvtColor(imgEdges, imgDisplay, COLOR_GRAY2BGR);

    // Wykrywanie linii za pomocą HoughLinesP
    HoughLinesP(imgEdges, linesP, 1, CV_PI / 180, lineVoters, minLineLength, maxLineGap);

    // Rysowanie wykrytych linii na obrazie
    for (size_t i = 0; i < linesP.size(); i++) {
        Vec4i l = linesP[i];
        line(imgDisplay, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2);  // Czerwony kolor linii
    }

    // Dodajemy tekst informujący o sterowaniu
    putText(imgDisplay, "Strzalki gora/dol: Low Threshold", Point(10, 570), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    putText(imgDisplay, "Strzalki lewo/prawo: High Threshold", Point(10, 590), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    putText(imgDisplay, "PageUp/PageDown: Line Voters", Point(10, 610), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    putText(imgDisplay, "Home/End: Min Line Length", Point(10, 630), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    putText(imgDisplay, "Insert/Delete: Max Line Gap", Point(10, 650), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    putText(imgDisplay, "ESC: Wyjscie", Point(10, 670), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

    // Wyświetlamy obraz z krawędziami i instrukcjami
    imshow("Canny + HoughLinesP", imgDisplay);
}

void handleKey(int key) {
    // Sterowanie progami Canny i parametrami HoughLinesP za pomocą klawiatury
    switch (key) {
        case 82:  // Strzałka w górę
            lowThreshold = min(lowThreshold + 1, maxThreshold);
            break;
        case 84:  // Strzałka w dół
            lowThreshold = max(lowThreshold - 1, 0);
            break;
        case 83:  // Strzałka w prawo
            highThreshold = min(highThreshold + 1, maxThreshold);
            break;
        case 81:  // Strzałka w lewo
            highThreshold = max(highThreshold - 1, 0);
            break;
        case 0x21: // Page Up
            lineVoters += 1;
            break;
        case 0x22: // Page Down
            lineVoters = max(lineVoters - 1, 0);
            break;
        case 0x24: // Home
            minLineLength += 1;
            break;
        case 0x23: // End
            minLineLength = max(minLineLength - 1, 0);
            break;
        case 0x2D: // Insert
            maxLineGap += 1;
            break;
        case 0x2E: // Delete
            maxLineGap = max(maxLineGap - 1, 0);
            break;
        default:
            break;
    }
    // Aktualizacja obrazu po zmianie progów
    applyCanny(0, 0);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Użycie: program <nazwa_pliku_graficznego>" << endl;
        return -1;
    }

    // Wczytanie obrazu
    imgOriginal = imread(argv[1], IMREAD_COLOR);
    if (imgOriginal.empty()) {
        cout << "Błąd: Nie można wczytać obrazu " << argv[1] << endl;
        return -1;
    }

    // Sprawdzenie rozmiaru obrazu i obliczenie odpowiedniej skali, aby zmieścił się w 1280x1024
    int maxWidth = 1280;
    int maxHeight = 800;

    float scaleX = (float)maxWidth / imgOriginal.cols;
    float scaleY = (float)maxHeight / imgOriginal.rows;

    // Wybór mniejszego współczynnika skali, aby zachować proporcje
    float scale = std::min(scaleX, scaleY);

    // Jeśli skala jest mniejsza niż 1, zmniejszamy obraz, w przeciwnym razie kopiujemy go bez zmian
    if (scale < 1) {
        Size newSize(imgOriginal.cols * scale, imgOriginal.rows * scale);
        resize(imgOriginal, imgResized, newSize);
    } else {
        imgResized = imgOriginal.clone();
    }

    // Konwersja do skali szarości
    cvtColor(imgResized, imgGray, COLOR_BGR2GRAY);

    // Tworzenie okna do wyświetlania obrazu i suwaków
    namedWindow("Canny + HoughLinesP", WINDOW_AUTOSIZE);

    // Tworzenie suwaków do kontroli progów funkcji Canny i parametrów Hougha
    createTrackbar("Low Threshold", "Canny + HoughLinesP", &lowThreshold, maxThreshold, applyCanny);
    createTrackbar("High Threshold", "Canny + HoughLinesP", &highThreshold, maxThreshold, applyCanny);
    createTrackbar("Line Voters", "Canny + HoughLinesP", &lineVoters, 100, applyCanny);  // Suwak dla lineVoters
    createTrackbar("Min Line Length", "Canny + HoughLinesP", &minLineLength, 200, applyCanny);  // Suwak dla minLineLength
    createTrackbar("Max Line Gap", "Canny + HoughLinesP", &maxLineGap, 100, applyCanny);  // Suwak dla maxLineGap

    // Początkowe zastosowanie funkcji Canny
    applyCanny(0, 0);

    // Obsługa klawiszy strzałek do sterowania progami
    while (true) {
        int key = waitKey(30);  // Oczekiwanie na klawisz
        if (key == 27) {        // Klawisz ESC – wyjście z programu
            break;
        }
        handleKey(key);         // Obsługa sterowania z klawiatury
    }

    return 0;
}
