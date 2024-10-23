#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>  // dla funkcji ceil
#include <filesystem>

/*
 * TODO: ROI as swich
 * ustalić minimalną ilość pokrywających się punktów, żeby uznać, że dopasowanie jest skuteczne (w canny, żeby nie używać arbitralnie MINIMUM_CANNY)
 * getFilesFromDirectory - poprawić, żeby wybierało tylko zdjęcia z prefixem #define
 */

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

#define CANNY_LOW 50
#define CANNY_HIGH 100
#define MAX_OFFSET 20
//#define ROI Rect(1680, 300, 1750, 1750)  // Definiujemy wycinek ROI (Rect(x, y, width, height))
#define ROI Rect(0, 0, 1750, 1750)  // Definiujemy wycinek ROI (Rect(x, y, width, height))
#define CROP_VALUE 25// Definiujemy wartość przycinania
#define MINIMUM_CANNY 1000  //500 to za mało, 700 raczej też. (dla Canny {50, 150}, dla {50, 100} wychodzą bardzo duże wartości.)
#define OUTPUT_DIR "/output/"  // Definiujemy katalog na pliki wynikowe

// Struktura przechowująca wszystkie dane dla każdego obrazu
struct ImageData {
    string filename;
    Mat originalImage;
    Mat cannyImage;
    int edgePointsCount;
    Point2f offset;
};

// Globalna zmienna imagesData
vector<ImageData> imagesData;

// Funkcja przetwarzająca wszystkie obrazy na krawędzie (Canny)
void processCanny() {
    for (auto& data : imagesData) {
        Mat gray;
        cvtColor(data.originalImage, gray, COLOR_BGR2GRAY);
        Canny(gray, data.cannyImage, CANNY_LOW, CANNY_HIGH);
    }
}

// Funkcja licząca punkty krawędzi dla każdego obrazu w imagesData
void countEdgePoints() {
    for (auto& data : imagesData) {
        data.edgePointsCount = countNonZero(data.cannyImage);
    }
}

// Funkcja generująca macierz translacji
Mat getTranslationMatrix(Point2f offset) {
    return (Mat_<double>(2, 3) << 1, 0, offset.x, 0, 1, offset.y);
}

// Funkcja tworząca katalog na pliki wynikowe (jeśli nie istnieje)
void createOutputDirectory(const string& directory) {
    if (!fs::exists(directory)) {
        fs::create_directory(directory);
    }
}

// Funkcja przesuwająca obraz o dane przesunięcie
Mat shiftImage(const Mat& image, Point2f offset) {
    Mat shifted;
    warpAffine(image, shifted, getTranslationMatrix(offset), image.size());
    return shifted;
}

// Sortowanie imagesData po liczbie punktów krawędzi (malejąco)
void sortImagesByEdgePoints() {
    std::sort(imagesData.begin(), imagesData.end(), [](const ImageData& a, const ImageData& b) {
        return a.edgePointsCount > b.edgePointsCount; // Sortowanie malejąco
    });
}

// Funkcja sumująca krawędzie dla wszystkich obrazów w imagesData
Mat cumulativeCanny() {
    Mat cumulativeEdges = Mat::zeros(imagesData[0].cannyImage.size(), CV_8UC1);
    
    for (const auto& data : imagesData) {
        
        if (data.edgePointsCount < MINIMUM_CANNY) {
            cout << "Pomijam " << data.filename << " ze względu na słaby wynik Canny." << endl;
            continue; // Przechodzimy do następnej iteracji pętli
        }
        
        Mat shiftedEdge = shiftImage(data.cannyImage, data.offset);
        cumulativeEdges += shiftedEdge;
    }

    return cumulativeEdges;
}

// Funkcja generująca kombinacje kolorów co 64 jednostki
vector<Scalar> generateColors() {
    vector<Scalar> colors;
    
    for (int r = 0; r <= 255; r += 64) {
        for (int g = 0; g <= 255; g += 64) {
            for (int b = 0; b <= 255; b += 64) {
                colors.push_back(Scalar(b, g, r)); // Scalar przyjmuje wartości w kolejności BGR, a nie RGB
            }
        }
    }
    
    return colors;
}

// Funkcja tworząca obraz nakładki krawędzi dla wszystkich obrazów w imagesData
Mat overlayColoredEdges() {
    // Sprawdzenie, czy istnieje co najmniej jeden obraz w imagesData
    if (imagesData.empty()) {
        cerr << "Brak obrazów do nałożenia." << endl;
        return Mat();
    }
    
    // Tworzenie obrazu nakładki z odpowiednimi rozmiarami
    Mat overlay = Mat::zeros(imagesData[0].originalImage.size(), CV_8UC3);

//     //Różne kolory do nałożenia krawędzi
//     vector<Scalar> colors = {
//         Scalar(255, 0, 0),   // niebieski
//         Scalar(0, 255, 0),   // zielony
//         Scalar(0, 0, 255),   // czerwony
//         Scalar(255, 255, 0), // cyjan
//         Scalar(255, 0, 255), // magenta
//         Scalar(0, 255, 255), // żółty
//         Scalar(255, 255, 255), // biały
//         Scalar(128, 0, 128), // fioletowy
//         Scalar(128, 128, 0), // oliwkowy
//         Scalar(0, 128, 128)  // niebiesko-zielony
//     };
    
    // Generowanie kolorów
    vector<Scalar> colors = generateColors();

    // Nakładanie krawędzi na obraz
    for (size_t i = 0; i < imagesData.size(); ++i) {
        
        const auto& data = imagesData[i];
        
        if (data.edgePointsCount < MINIMUM_CANNY) {
            cout << "Pomijam " << data.filename << " ze względu na słaby wynik Canny." << endl;
            continue; // Przechodzimy do następnej iteracji pętli
        }

        // Przesuwamy obraz krawędzi o przesunięcie
        Mat shiftedEdge = shiftImage(data.cannyImage, data.offset);

        // Wybieramy kolor z listy, cyklicznie jeśli liczba obrazów > liczba kolorów
        Scalar color = colors[i % colors.size()];

        // Nakładanie krawędzi na obraz nakładki
        for (int y = 0; y < shiftedEdge.rows; y++) {
            for (int x = 0; x < shiftedEdge.cols; x++) {
                if (shiftedEdge.at<uchar>(y, x) > 0) {
                    // Przypisanie koloru po konwersji Scalar do Vec3b
                    overlay.at<Vec3b>(y, x) = Vec3b((uchar)color[0], (uchar)color[1], (uchar)color[2]);
                }
            }
        }
    }
    
    return overlay;
}


// Funkcja do ładowania obrazów z listy plików
void loadImages(const vector<string>& fileList) {
    for (const auto& file : fileList) {
        ImageData data;
        data.filename = file;
        data.originalImage = imread(file);

        if (data.originalImage.empty()) {
            cerr << "Nie udało się załadować obrazu: " << data.filename << endl;
            continue;
        }
        
        // Przycinamy obraz według ROI
        if (ROI.x + ROI.width <= data.originalImage.cols && ROI.y + ROI.height <= data.originalImage.rows) {
            data.originalImage = data.originalImage(ROI).clone(); // Przycinanie i klonowanie obrazu
        } else {
            cerr << "ROI przekracza wymiary obrazu: " << data.filename << endl;
            continue; // Kontynuuj z następnym obrazem
        }
        
        data.offset = Point2f(0, 0);
        data.edgePointsCount = 0;
        imagesData.push_back(data);
    }
}

// Funkcja licząca ilość nakładających się punktów krawędzi
int countEdgeOverlap(const Mat& cumulativeEdges, const Mat& shiftedImage) {
    int overlap = 0;
    for (int y = 0; y < cumulativeEdges.rows; y++) {
        for (int x = 0; x < cumulativeEdges.cols; x++) {
            if (cumulativeEdges.at<uchar>(y, x) > 0 && shiftedImage.at<uchar>(y, x) > 0) {
                overlap++;
            }
        }
    }
    return overlap;
}

// Funkcja zapisująca oryginalne obrazy po przycięciu do plików
void saveCroppedImages(const string& outputDir) {
    // Sprawdzamy, czy istnieje co najmniej jeden obraz w imagesData
    if (imagesData.empty()) {
        cerr << "Brak obrazów do zapisania." << endl;
        return;
    }

    // Obliczamy maksymalne przesunięcia w każdym kierunku
    int maxOffsetX = 0;
    int maxOffsetY = 0;

    for (const auto& data : imagesData) {
        maxOffsetX = std::max(maxOffsetX, static_cast<int>(std::abs(data.offset.x)));
        maxOffsetY = std::max(maxOffsetY, static_cast<int>(std::abs(data.offset.y)));
    }

    // Ustal maksymalne przesunięcie
    int maxOffset = std::max(maxOffsetX, maxOffsetY);
    int cropValue = (maxOffset > CROP_VALUE) ? std::ceil(maxOffset / 10.0) * 10 : CROP_VALUE;

    for (const auto& data : imagesData) {
        if (data.edgePointsCount < MINIMUM_CANNY) {
            cout << "Pomijam " << data.filename << " ze względu na słaby wynik Canny." << endl;
            continue; // Przechodzimy do następnej iteracji pętli
        }
        
        // Przesunięcie obrazu
        Mat shiftedImage = shiftImage(data.originalImage, data.offset);

        // Sprawdzamy, czy przesunięcie nie powoduje, że obraz jest zbyt mały
        if (shiftedImage.cols <= 2 * cropValue || shiftedImage.rows <= 2 * cropValue) {
            cerr << "Obraz jest zbyt mały do przycięcia dla obrazu: " << data.filename << endl;
            continue;
        }

        // Przycinanie obrazu
        Rect roi(cropValue, cropValue, 
                 shiftedImage.cols - 2 * cropValue, 
                 shiftedImage.rows - 2 * cropValue);

        // Sprawdzamy, czy wycinek ROI jest poprawny
        if (roi.width <= 0 || roi.height <= 0) {
            cerr << "Błędny wycinek ROI dla obrazu: " << data.filename << endl;
            continue;
        }

        // Tworzymy obraz przycięty
        Mat croppedImage = shiftedImage(roi);

        // Generujemy nazwę pliku
        //string filenameWithoutExtension = data.filename.substr(0, data.filename.find_last_of('.'));
        //string outputFilename = outputDir + filenameWithoutExtension + "_centered.png";
        string filename = fs::path(data.filename).stem().string();
        string filenameWithoutExtension = filename.substr(0, filename.find_last_of('.'));
        string outputFilename = outputDir + filenameWithoutExtension + "_centered.png";

        // Zapisujemy przycięty obraz
        if (!imwrite(outputFilename, croppedImage)) {
            cerr << "Nie udało się zapisać obrazu: " << outputFilename << endl;
        } else {
            cout << "Zapisano przycięty obraz: " << outputFilename << endl;
        }
    }
}

// Funkcja znajdująca najlepsze przesunięcie obrazu na podstawie dopasowania krawędzi
Point findBestOffset(const Mat& cumulativeEdges, const Mat& currentCanny) {
    // Zakres przesunięcia (przykładowo, od -30 do 30 pikseli w każdą stronę)
    int maxOffset = MAX_OFFSET;
    Point bestOffset(0, 0);
    int maxOverlap = 0;

    // Sprawdzanie przesunięcia w zakresie
    for (int dx = -maxOffset; dx <= maxOffset; ++dx) {
        for (int dy = -maxOffset; dy <= maxOffset; ++dy) {
            Mat shiftedImage = shiftImage(currentCanny, Point(dx, dy));
            int overlap = countEdgeOverlap(cumulativeEdges, shiftedImage);
            
            // Jeśli więcej punktów się pokrywa, zapisujemy to przesunięcie
            if (overlap > maxOverlap) {
                maxOverlap = overlap;
                bestOffset = Point(dx, dy);
            }
        }
    }
    
    return bestOffset; // Zwracamy najlepsze przesunięcie
}


// Funkcja dopasowująca i kumulująca obrazy krawędzi
Mat cumulativeEdgeAlignment() {
    // Inicjalizacja kumulacji krawędzi od pierwszego obrazu
    Mat cumulativeEdges = imagesData[0].cannyImage.clone();
    
    for (size_t i = 1; i < imagesData.size(); ++i) {
        cout << "Porównywanie " << imagesData[i].filename << " do skumulowanego obrazu." << endl;
        
        if (imagesData[i].edgePointsCount < MINIMUM_CANNY) {
            cout << "Pomijam " << imagesData[i].filename << " ze względu na słaby wynik Canny." << endl;
            continue; // Przechodzimy do następnej iteracji pętli
        }
        
        // Dopasowanie bieżącego obrazu krawędzi do kumulacji
        const Mat& currentCanny = imagesData[i].cannyImage;
        
        // Dopasuj obraz do kumulacji (tutaj można zaimplementować algorytm dopasowania)
        Point bestOffset = findBestOffset(cumulativeEdges, currentCanny);
        
        // Zapisz przesunięcie do imagesData
        imagesData[i].offset = bestOffset;

        // Przesuń obraz krawędzi o obliczone przesunięcie
        Mat shiftedEdge = shiftImage(currentCanny, bestOffset);

        // Kumuluj przesunięte krawędzie do kumulacji
        cumulativeEdges = max(cumulativeEdges, shiftedEdge); // Sumujemy binarnie krawędzie
    }
    
    return cumulativeEdges; // Zwracamy ostateczny obraz kumulacji krawędzi
}

// Nowa funkcja przetwarzająca obrazy z listy plików
void processImages(const vector<string>& fileList, const string& outputDir) {
    //new filelist, new data.
    imagesData.clear();
    
    // Załaduj obrazy z listy
    loadImages(fileList);

    if (imagesData.empty()) {
        cerr << "Brak załadowanych obrazów!" << endl;
        return;
    }

    // Przetwarzaj obrazy na krawędzie
    processCanny();

    // Oblicz liczbę punktów krawędzi
    countEdgePoints();

    // Sortowanie według liczby punktów krawędzi (malejąco)
    sortImagesByEdgePoints();

    // Wyświetl posortowane dane
    for (const auto& data : imagesData) {
        cout << "Obraz: " << data.filename
             << ", Punkty krawędzi: " << data.edgePointsCount
             << ", Przesunięcie: (" << data.offset.x << ", " << data.offset.y << ")" << endl;
    }

    // Dopasuj obrazy i uzyskaj kumulację krawędzi
    Mat cumulativeEdges = cumulativeEdgeAlignment();
    // Zapisz wynikowy obraz kumulacji krawędzi
    imwrite(outputDir + "/edges_cumulation.png", cumulativeEdges);  // Zapisujemy w katalogu wynikowym

    // Wywołanie funkcji overlayColoredEdges i zapisanie obrazu
    Mat resultOverlay = overlayColoredEdges();
    imwrite(outputDir + "/edges_overlay.png", resultOverlay);  // Zapisujemy w katalogu wynikowym


    // Zapisz obrazy canny
    saveCroppedImages(outputDir);
}

// Funkcja do generowania listy plików z katalogu
vector<string> getFilesFromDirectory(const string& directory) {
    vector<string> fileList;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            fileList.push_back(entry.path().string());
        }
    }
    return fileList;
}


// Funkcja do generowania katalogu wynikowego na podstawie ścieżki pliku
string generateOutputDirectory(const string& filePath) {
    string dir = fs::path(filePath).parent_path().string();  // Pobiera katalog nadrzędny
    dir = dir + OUTPUT_DIR;
    cout << "Output dir: " << dir << endl;
    createOutputDirectory(dir);
    return dir;  // Tworzymy podkatalog "output"
}

// Funkcja główna
int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Podaj co najmniej jeden plik, katalog, lub listę katalogów jako argument!" << endl;
        return -1;
    }

    vector<vector<string>> fileLists;

    // Iteracja po argumentach
    for (int i = 1; i < argc; ++i) {
        string path = argv[i];
        if (fs::is_directory(path)) {
            // Jeśli argument to katalog, generujemy listę plików
            vector<string> files = getFilesFromDirectory(path);
            if (!files.empty()) {
                fileLists.push_back(files);
            }
        } else if (fs::is_regular_file(path)) {
            // Jeśli to plik, dodajemy go do listy jako pojedynczy element
            fileLists.push_back({path});
        } else {
            cerr << "Ścieżka nie jest prawidłowym plikiem lub katalogiem: " << path << endl;
        }
    }

    // Jeśli nie znaleziono żadnych plików, zwracamy błąd
    if (fileLists.empty()) {
        cerr << "Brak plików do przetworzenia!" << endl;
        return -1;
    }

    // Przetwarzanie każdej listy plików
    for (const auto& fileList : fileLists) {
        string outputDir = generateOutputDirectory(fileList[0]);        
        processImages(fileList, outputDir);
    }

    return 0;
}
