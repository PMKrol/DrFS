/*
 * compile: g++ -std=c++17 -o canny_slider canny_slider.cpp `pkg-config --cflags --libs opencv4`
 * run: ./canny_slider /sciezka/do/katalogu
 */

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int lowThreshold = 50;    // Początkowa wartość progu dolnego
int highThreshold = 150;  // Początkowa wartość progu górnego
const int maxThreshold = 300;
Mat imgOriginal, imgGray, imgEdges, imgResized;

void applyCanny(int, void*) {
    // Zastosowanie detekcji krawędzi Canny na obrazku
    Canny(imgGray, imgEdges, lowThreshold, highThreshold);

    // Konwersja do BGR, aby móc nałożyć tekst
    Mat imgDisplay;
    cvtColor(imgEdges, imgDisplay, COLOR_GRAY2BGR);

    // Dodajemy tekst informujący o sterowaniu
    putText(imgDisplay, "Strzalki gora/dol: Low Threshold", Point(10, 570), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    putText(imgDisplay, "Strzalki lewo/prawo: High Threshold", Point(10, 590), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    putText(imgDisplay, "ESC: Wyjscie", Point(10, 610), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

    // Wyświetlamy obraz z krawędziami i instrukcjami
    imshow("Canny Edges", imgDisplay);
}

void handleKey(int key) {
    // Sterowanie progami Canny za pomocą klawiatury
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

    // Sprawdzenie rozmiaru i skalowanie obrazu, jeśli jest większy niż 800x600
    Size newSize(800, 600);
    if (imgOriginal.cols > 800 || imgOriginal.rows > 600) {
        resize(imgOriginal, imgResized, newSize);
    } else {
        imgResized = imgOriginal.clone();
    }

    // Konwersja do skali szarości
    cvtColor(imgResized, imgGray, COLOR_BGR2GRAY);

    // Tworzenie okna do wyświetlania obrazu i suwaków
    namedWindow("Canny Edges", WINDOW_AUTOSIZE);

    // Tworzenie suwaków do kontroli progów funkcji Canny
    createTrackbar("Low Threshold", "Canny Edges", &lowThreshold, maxThreshold, applyCanny);
    createTrackbar("High Threshold", "Canny Edges", &highThreshold, maxThreshold, applyCanny);

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
