#include <opencv2/opencv.hpp>
#include <iostream>

// Zmienne do parametrów HoughCircles i Canny
int dp = 1;                 // Parametr dp: odwrotność rozdzielczości akumulatora
int minDist = 20;           // Minimalna odległość między środkami wykrytych okręgów
int param1 = 100;           // Pierwszy próg dla detektora Canny
int param2 = 30;            // Drugi próg dla akumulatora Hougha (mniejsza wartość = więcej okręgów)
int minRadius = 0;          // Minimalny promień okręgu
int maxRadius = 100;        // Maksymalny promień okręgu
int minArcPercentage = 30;  // Minimalny procent łuku okręgu

cv::Mat originalImg, cannyImg, resultImg;
cv::Size displaySize(1200, 800); // Maksymalny rozmiar wyświetlanego obrazu

void detectCircles() {
    cv::Mat gray;
    cv::cvtColor(originalImg, gray, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2);

    // Generowanie obrazu Canny
    cv::Canny(gray, cannyImg, param1, param1 * 2);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius);

    // Tworzymy pusty obraz wynikowy o 3 kanałach BGR
    resultImg = cv::Mat::zeros(cannyImg.size(), CV_8UC3);  
    cv::cvtColor(cannyImg, resultImg, cv::COLOR_GRAY2BGR);  // Konwertujemy obraz Canny do BGR

    for (size_t i = 0; i < circles.size(); i++) {
        cv::Vec3i c = circles[i];
        cv::Point center(c[0], c[1]);
        int radius = c[2];
        
        float arc_length = 2 * CV_PI * radius * (minArcPercentage / 100.0);
        if (arc_length >= (2 * CV_PI * radius * (minArcPercentage / 100.0))) {
            cv::circle(resultImg, center, 3, cv::Scalar(0, 255, 0), -1); // Środek okręgu
            cv::circle(resultImg, center, radius, cv::Scalar(0, 0, 255), 3); // Okrąg
        }
    }

    // Skalowanie obrazu wynikowego do rozmiaru okna
    double scaleFactor = std::min((double)displaySize.width / resultImg.cols, (double)displaySize.height / resultImg.rows);
    cv::Mat displayImg;
    cv::resize(resultImg, displayImg, cv::Size(), scaleFactor, scaleFactor);

    cv::imshow("Processed Image", displayImg);
}

void onTrackbarChange(int, void*) {
    detectCircles();
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    std::string imagePath = argv[1];
    originalImg = cv::imread(imagePath);
    if (originalImg.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    // Ustawienie maksymalnego promienia okręgów na maksymalny wymiar obrazu
    int maxDimension = std::max(originalImg.cols, originalImg.rows);
    maxRadius = maxDimension;

    cv::namedWindow("Processed Image", cv::WINDOW_NORMAL);
    cv::resizeWindow("Processed Image", displaySize.width, displaySize.height);

    // Stwórz suwaki i przypisz funkcję callback
    cv::createTrackbar("dp", "Processed Image", &dp, 10, onTrackbarChange);
    cv::createTrackbar("minDist", "Processed Image", &minDist, 100, onTrackbarChange);
    cv::createTrackbar("param1", "Processed Image", &param1, 200, onTrackbarChange);
    cv::createTrackbar("param2", "Processed Image", &param2, 100, onTrackbarChange);
    cv::createTrackbar("minRadius", "Processed Image", &minRadius, maxDimension, onTrackbarChange);
    cv::createTrackbar("maxRadius", "Processed Image", &maxRadius, maxDimension, onTrackbarChange);
    cv::createTrackbar("Arc %", "Processed Image", &minArcPercentage, 100, onTrackbarChange);

    // Początkowe wykrycie okręgów
    detectCircles();

    while (cv::waitKey(1) != 27) {
        // Pętla oczekująca na zamknięcie okna
    }

    return 0;
}
