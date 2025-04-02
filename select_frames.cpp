/* 
 * This one selects proper frames from zip file, and creates temperatures txt file.
 * sudo apt install libzip-dev
    g++ -std=c++17 -o select_frames select_frames.cpp `pkg-config --cflags --libs opencv4` -lzip
 * 
 */ 

#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <zip.h>

using namespace std;
namespace fs = std::filesystem;
using namespace cv;  // Używamy przestrzeni nazw cv

// Funkcja konwertująca string na liczbę całkowitą
int parseCoordinate(const string& coord) {
    stringstream ss(coord);
    int result;
    ss >> result;
    return result;
}

// Funkcja parsująca parametry obszaru ROI
bool parseArea(const string& areaStr, int& x, int& y, int& width, int& height) {
    size_t colonPos = areaStr.find(':');
    if (colonPos == string::npos) return false;

    string coords = areaStr.substr(0, colonPos);
    string sizeStr = areaStr.substr(colonPos + 1);

    size_t xPos = coords.find('x');
    if (xPos == string::npos) return false;

    x = parseCoordinate(coords.substr(0, xPos));
    y = parseCoordinate(coords.substr(xPos + 1));

    size_t xSizePos = sizeStr.find('x');
    if (xSizePos == string::npos) return false;

    width = parseCoordinate(sizeStr.substr(0, xSizePos));
    height = parseCoordinate(sizeStr.substr(xSizePos + 1));

    return true;
}

// Funkcja do obliczania temperatury z danych raw
double raw2temp(uint8_t byte0, uint8_t byte1) {
    return (byte0 + byte1 * 256) / 64.0 - 273.15;
}

// Funkcja do konwersji danych raw na macierz temperatur
vector<vector<double>> processRawData(const vector<uint8_t>& rawData, int rows, int cols) {
    vector<vector<double>> temperatures(rows, vector<double>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            size_t idx = (i * cols + j) * 2;
            uint8_t byte0 = rawData[idx];
            uint8_t byte1 = rawData[idx + 1];
            temperatures[i][j] = raw2temp(byte0, byte1);
        }
    }

    return temperatures;
}

vector<size_t> findTwoLargestLocalMaxima(const vector<double>& values) {
    vector<pair<size_t, double>> localMaximaWithValues;

    // Znajdowanie wszystkich lokalnych maksimów wraz z ich wartościami
    for (size_t i = 1; i < values.size() - 1; ++i) {
        if (values[i] > values[i - 1] && values[i] > values[i + 1]) {
            localMaximaWithValues.emplace_back(i, values[i]);
        }
    }

    // Sortowanie lokalnych maksimów według wartości w malejącej kolejności
    sort(localMaximaWithValues.begin(), localMaximaWithValues.end(),
         [](const pair<size_t, double>& a, const pair<size_t, double>& b) {
             return a.second > b.second;
         });

    // Wybieranie dwóch największych lokalnych maksimów, które mają co najmniej 10 klatek odstępu
    vector<size_t> localMaxima;
    if (localMaximaWithValues.size() > 1) {
        // Dodajemy pierwsze maksimum
        localMaxima.push_back(localMaximaWithValues[0].first);

        // Szukamy drugiego maksimum, które jest oddalone o co najmniej 10 klatek
        for (size_t i = 1; i < localMaximaWithValues.size(); ++i) {
            if (abs(localMaximaWithValues[i].first - localMaxima[0]) >= 10) {
                localMaxima.push_back(localMaximaWithValues[i].first);
                break;
            }
        }
    }

    // Jeśli udało się znaleźć dwa maksimy, sortujemy je w kolejności chronologicznej
    if (localMaxima.size() == 2) {
        sort(localMaxima.begin(), localMaxima.end());
    }

    return localMaxima;
}


// Funkcja do obliczania średnich temperatur w ROI
double calculateAverageTemperature(const vector<vector<double>>& temperatures, const Rect& roi) {
    double sum = 0.0;
    int count = 0;
    
    // Przeiterowanie po ROI (dolna połowa)
    for (int i = roi.y; i < roi.y + roi.height; ++i) {
        for (int j = roi.x; j < roi.x + roi.width; ++j) {
            sum += temperatures[i][j];
            count++;
        }
    }

    return sum / count;
}

// Funkcja do zapisywania temperatur z klatki do pliku
void saveTemperaturesToFile(const vector<vector<double>>& temperatures, size_t frameIndex, const string& outputDir) {
    std::filesystem::path wipDir = std::filesystem::path(outputDir) / "wip";
    if (!std::filesystem::exists(wipDir)) {
        std::cerr << "Katalog wip nie istnieje, tworzenie katalogu..." << std::endl;
        std::filesystem::create_directory(wipDir);
    } else {
        //std::cout << wipDir << std::endl;
    }
    
    // Ustalanie nazw plików
    string outputFile;
    if (frameIndex == 0) {
        outputFile = outputDir + "/wip/temperatures_before.txt";
    } else if (frameIndex == 1) {
        outputFile = outputDir + "/wip/temperatures_after.txt";
    } else {
        return; // Jeśli klatka nie jest pierwsza ani druga, nic nie zapisuj
    }

    // Zapis do pliku
    ofstream outFile(outputFile);
    
    if (!outFile) {
        cerr << "Error: Cannot open output file " << outputFile << endl;
        return;
    }

    // Zapisujemy temperatury do pliku
    for (const auto& row : temperatures) {
        for (size_t i = 0; i < row.size(); ++i) {
            outFile << row[i];
            if (i < row.size() - 1) {
                outFile << " ";
            }
        }
        outFile << endl;
    }
}

// Funkcja do zapisywania temperatur (jednej macierzy) do pliku
void saveTemperaturesSingle(const vector<vector<double>>& temperatures, const string& outputFile) {
    // Zapis do pliku
    ofstream outFile(outputFile);

    if (!outFile) {
        cerr << "Error: Cannot open output file " << outputFile << endl;
        return;
    }

    // Zapisujemy temperatury do pliku
    for (const auto& row : temperatures) {
        for (size_t i = 0; i < row.size(); ++i) {
            outFile << row[i];
            if (i < row.size() - 1) {
                outFile << " ";  // Dodanie spacji między wartościami w wierszu
            }
        }
        outFile << endl;  // Nowa linia po każdym wierszu
    }
}

// Funkcja do zapisywania numerów klatek do pliku frames.txt
void saveFrameNumbersToFile(const vector<size_t>& frameNumbers, const string& outputDir) {
    std::filesystem::path wipDir = std::filesystem::path(outputDir) / "wip";
    if (!std::filesystem::exists(wipDir)) {
        std::cerr << "Katalog wip nie istnieje, tworzenie katalogu..." << std::endl;
        std::filesystem::create_directory(wipDir);
    } else {
        //std::cout << wipDir << std::endl;
    }
    
    string framesFile = outputDir + "/wip/frames.txt";
    ofstream frameOutFile(framesFile);
    
    if (!frameOutFile) {
        cerr << "Error: Cannot open output file " << framesFile << endl;
        return;
    }

    for (size_t i = 0; i < frameNumbers.size(); ++i) {
        frameOutFile << frameNumbers[i];
        if (i < frameNumbers.size() - 1) {
            frameOutFile << " ";
        }
    }

    frameOutFile << endl;
}

// Funkcja do przetwarzania plików zip i obliczania temperatur
vector<vector<vector<double>>> processZipFiles(const string& zipFilePath) {
    int err = 0;
    zip* z = zip_open(zipFilePath.c_str(), 0, &err);
    if (z == nullptr) {
        cerr << "Error: Cannot open zip file " << zipFilePath << endl;
        return {};
    }else{
        cout << zipFilePath << endl;
    }

    vector<string> filenames;
    zip_int64_t numFiles = zip_get_num_entries(z, 0);

    // Iteracja po plikach w archiwum i sortowanie ich alfabetycznie
    for (zip_int64_t i = 0; i < numFiles; ++i) {
        const char* filename = zip_get_name(z, i, 0);
        if (strstr(filename, "frame_") != nullptr) {
            filenames.push_back(filename);  // Dodajemy pliki z prefiksem "frame_"
        }
    }

    sort(filenames.begin(), filenames.end());  // Sortowanie alfabetyczne

    vector<vector<vector<double>>> allTemperatures;

    // Odczyt plików i obliczanie temperatur
    for (const auto& filename : filenames) {
        // Odczytanie zawartości pliku
        struct zip_stat st;
        zip_stat_init(&st);
        zip_stat(z, filename.c_str(), 0, &st);

        zip_file* zf = zip_fopen(z, filename.c_str(), 0);
        if (zf == nullptr) {
            cerr << "Error: Cannot open file " << filename << " in zip" << endl;
            continue;
        }

        vector<uint8_t> rawData(st.size);
        zip_fread(zf, rawData.data(), rawData.size());
        zip_fclose(zf);

        // Rozmiar obrazu to 256x384
        int fullRows = 384;  // 384 wiersze
        int fullCols = 256;  // 256 kolumn
        int roiRows = 192;   // Dolna połowa to 192 wiersze (od 192 do 383)
        int cols = fullCols;

        // Przetwarzamy cały obraz, aby uzyskać temperatury
        vector<vector<double>> fullTemperatures = processRawData(rawData, fullRows, fullCols);

        // Teraz, musimy tylko wybrać dolną połowę obrazu
        vector<vector<double>> temperatures(roiRows, vector<double>(cols));

        for (int i = 192; i < fullRows; ++i) {  // Przetwarzamy tylko wiersze 192-383
            for (int j = 0; j < cols; ++j) {
                temperatures[i - 192][j] = fullTemperatures[i][j];
            }
        }

        // Dodajemy temperatury z dolnej połowy do wektora
        allTemperatures.push_back(temperatures);
    }



    zip_close(z);

    return allTemperatures;
}

// Funkcja przekształcająca temperatury na zakres 0-255 (skalowanie do 8-bitowej skali szarości)
cv::Mat scaleTemperaturesToImage(const vector<vector<double>>& temperatures) {
    int rows = temperatures.size();
    int cols = temperatures[0].size();
    
    // Znajdowanie minimum i maksimum w temperaturach
    double minTemp = DBL_MAX, maxTemp = -DBL_MAX;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            minTemp = min(minTemp, temperatures[i][j]);
            maxTemp = max(maxTemp, temperatures[i][j]);
        }
    }

    // Tworzenie matrycy obrazu (skalowanie do zakresu 0-255)
    cv::Mat image(rows, cols, CV_8U);  // Używamy typu CV_8U, żeby obraz miał wartości 0-255 (8-bitowy obraz szarości)
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Przeskalowanie wartości temperatury na zakres 0-255
            uint8_t pixelValue = static_cast<uint8_t>(255 * (temperatures[i][j] - minTemp) / (maxTemp - minTemp));
            image.at<uint8_t>(i, j) = pixelValue;
        }
    }

    return image;
}

// Funkcja zapisująca obraz do pliku
void saveImageToFile(const cv::Mat& image, const string& outputDir, const string& fileName) {
    std::filesystem::path wipDir = std::filesystem::path(outputDir) / "wip";
    if (!std::filesystem::exists(wipDir)) {
        std::cerr << "Katalog wip nie istnieje, tworzenie katalogu..." << std::endl;
        std::filesystem::create_directory(wipDir);
    } else {
        //std::cout << wipDir << std::endl;
    }
        
    string outputPath = outputDir + "/wip/" + fileName;
    if (!cv::imwrite(outputPath, image)) {
        cerr << "Error: Cannot save image to " << outputPath << endl;
    }
}

std::string toStringWithLeadingZeros(int number, int width) {
    std::stringstream ss;
    ss << std::setw(width) << std::setfill('0') << number;
    return ss.str();
}

// Funkcja zapisująca obrazy w katalogu "wip", zaznaczając ROI
void saveTemperaturesToWipDirectory(const std::vector<std::vector<std::vector<double>>>& allTemperatures, const std::string& zipFilePath, const cv::Rect& roi) {
    // Uzyskujemy ścieżkę do katalogu, usuwając nazwę pliku zip
    size_t lastSlashPos = zipFilePath.find_last_of('/');
    std::string outputDir = (lastSlashPos != std::string::npos) ? zipFilePath.substr(0, lastSlashPos) : ".";

    // Tworzymy katalog "wip" jeśli jeszcze nie istnieje
    outputDir += "/wip";
    std::filesystem::create_directory(outputDir);  // Tworzenie katalogu, jeśli nie istnieje

    // Zapisujemy obrazy
    for (size_t i = 0; i < allTemperatures.size(); ++i) {
        // Przekształcamy temperatury na obraz (czarno-biały)
        const std::vector<std::vector<double>>& temperatureData = allTemperatures[i];
        cv::Mat image = scaleTemperaturesToImage(temperatureData);

        // Zaznaczamy ROI na obrazie
        cv::rectangle(image, roi, cv::Scalar(255, 255, 255), 1);  // Biały prostokąt o grubości 2 pikseli

        // Generujemy nazwę pliku (frame_001.png, frame_002.png, ...)
        std::string fileNameTxt = "frame_" + toStringWithLeadingZeros(i + 1, 3) + ".txt";
        std::string filePathTxt = outputDir + "/" + fileNameTxt;
        
        //void saveTemperaturesToFile(const vector<vector<double>>& temperatures, size_t frameIndex, const string& outputDir) {
        //saveTemperaturesToFile(temperatureData, i, outputDir + "/");
        saveTemperaturesSingle(temperatureData, filePathTxt);

        // Generujemy nazwę pliku (frame_001.png, frame_002.png, ...)
        std::string fileNamePng = "frame_" + toStringWithLeadingZeros(i + 1, 3) + ".png";
        std::string filePathPng = outputDir + "/" + fileNamePng;

        // Zapisujemy obraz jako PNG
        if (!cv::imwrite(filePathPng, image)) {
            std::cerr << "Error: Failed to save image " << filePathPng << std::endl;
        } else {
            std::cout << "Saved image: " << filePathPng << std::endl;
        }
    }
}

// Funkcja rysująca wykres średnich temperatur z zaznaczonymi lokalnymi maksimami
void plotTemperaturesWithLocalMaxima(const vector<double>& avgTemperatures, const vector<size_t>& localMaxima, const string& outputDir) {
    // Parametry obrazu
    int width = 800;
    int height = 400;

    // Tworzymy pusty obraz na wykres (czarne tło)
    Mat plotImage = Mat::zeros(height, width, CV_8UC3);

    // Normalizujemy temperatury, aby zmieściły się w zakresie wysokości obrazu
    double minTemp, maxTemp;
    minMaxLoc(avgTemperatures, &minTemp, &maxTemp);  // Znajdujemy minimalną i maksymalną temperaturę

    // Skala temperatury do wysokości obrazu
    double scale = (height - 20) / (maxTemp - minTemp);  // Przeskalowanie temperatury do wysokości wykresu

    // Rysowanie wykresu średnich temperatur
    for (size_t i = 0; i < avgTemperatures.size() - 1; ++i) {
        int x1 = i * (width / avgTemperatures.size());
        int y1 = height - (avgTemperatures[i] - minTemp) * scale;
        int x2 = (i + 1) * (width / avgTemperatures.size());
        int y2 = height - (avgTemperatures[i + 1] - minTemp) * scale;

        // Rysowanie linii łączącej punkty na wykresie
        line(plotImage, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 0), 2);
    }

    // Zaznaczanie lokalnych maksimów pionowymi liniami
    for (size_t idx : localMaxima) {
        int x = idx * (width / avgTemperatures.size());
        // Pionowa linia, która wskazuje na lokalne maksimum
        line(plotImage, Point(x, 0), Point(x, height), Scalar(0, 0, 255), 2);
    }

    // Wyświetlanie wykresu
    //imshow("Temperature Plot with Local Maxima", plotImage);
    //waitKey(0);  // Czeka na naciśnięcie dowolnego klawisza, by zamknąć okno

    string outputFile = outputDir + "/wip/hotspotPlot.png";

    // Zapisz obraz do pliku
    if (!imwrite(outputFile, plotImage)) {
        cerr << "Błąd podczas zapisywania obrazu do " << outputFile << endl;
    } else {
        cout << "Obraz zapisany do " << outputFile << endl;
    }
}

// Funkcja do przetwarzania plików raw z archiwum zip i analizy
void processZipFile(const string& zipFilePath, const Rect& roi) {
    vector<vector<vector<double>>> allTemperatures = processZipFiles(zipFilePath);
    
    // Zapisujemy obrazy do katalogu "wip"
    saveTemperaturesToWipDirectory(allTemperatures, zipFilePath, roi);
    
    if (allTemperatures.empty()) {
        cerr << "No frames processed." << endl;
        return;
    }

    // Wektor na średnie temperatury w ROI
    vector<double> avgTemperatures;
    
    // Przesunięcie ROI o 192 piksele w dół
    //cv::Rect shiftedRoi = roi;  // Tworzymy kopię istniejącego ROI
    //shiftedRoi.y += 192;        // Przesuwamy prostokąt o 192 piksele w dół

    // Obliczanie średnich temperatur w ROI
    for (const auto& temperatures : allTemperatures) {
        double avgTemp = calculateAverageTemperature(temperatures, roi);
        avgTemperatures.push_back(avgTemp);
    
        // Wyświetlanie wartości średniej temperatury
        std::cout << "Średnia temperatura w ROI: " << avgTemp << std::endl;
    }

    // Znajdowanie dwóch największych lokalnych maksimów w średnich temperaturach
    vector<size_t> localMaxima = findTwoLargestLocalMaxima(avgTemperatures);
    
    size_t lastSlashPosPlot = zipFilePath.find_last_of('/');
    string outputDirPlot = (lastSlashPosPlot != string::npos) ? zipFilePath.substr(0, lastSlashPosPlot) : ".";
    plotTemperaturesWithLocalMaxima(avgTemperatures, localMaxima, outputDirPlot);
    
    // Sprawdzamy, czy znaleźliśmy co najmniej dwa maksima
    if (localMaxima.size() >= 2) {
        // Zapisujemy temperatury z pierwszego maksimum
        size_t firstMaxIndex = localMaxima[0];
        const vector<vector<double>>& selectedTemperatures = allTemperatures[firstMaxIndex];

        // Ustalamy ścieżkę do katalogu zip
        size_t lastSlashPos = zipFilePath.find_last_of('/');
        string outputDir = (lastSlashPos != string::npos) ? zipFilePath.substr(0, lastSlashPos) : ".";

        // Zapisz temperatury do pliku
        //saveTemperaturesToFile(selectedTemperatures, 0, outputDir);  // Pierwsza klatka do temperatures_before.txt
        //saveTemperaturesToFile(allTemperatures[localMaxima[1]], 1, outputDir);  // Druga klatka do temperatures_after.txt
        saveTemperaturesSingle(allTemperatures[localMaxima[0]], outputDir + "/wip/frame_before.txt");
        saveTemperaturesSingle(allTemperatures[localMaxima[1]], outputDir + "/wip/frame_after.txt");

        // Zapisujemy numery klatek do pliku frames.txt
        saveFrameNumbersToFile(localMaxima, outputDir);
        
        // Tworzymy obrazy z temperaturami i zapisujemy je
        cv::Mat beforeImage = scaleTemperaturesToImage(selectedTemperatures);  // Pierwsza klatka
        cv::Mat afterImage = scaleTemperaturesToImage(allTemperatures[localMaxima[1]]);  // Druga klatka
        
        saveImageToFile(beforeImage, outputDir, "frame_before.png");  // Zapisz jako frame_before.png
        saveImageToFile(afterImage, outputDir, "frame_after.png");  // Zapisz jako frame_after.png
    } else {
        cout << "Can't find proper local maximas." << endl;
    }
}



int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: ./temperature_analysis <roi> <directory1> [<directory2> ...]" << endl;
        return -1;
    }

    // Parsowanie obszaru ROI
    int x, y, width, height;
    if (!parseArea(argv[1], x, y, width, height)) {
        cout << "Error: Invalid area format!" << endl;
        return -1;
    }
    cv::Rect roi(x, y, width, height);

        // Przetwarzanie katalogów
    for (int dirIndex = 2; dirIndex < argc; ++dirIndex) {
        string dirPath = argv[dirIndex];

        // Tworzenie pełnej nazwy pliku zip: <nazwa_katalogu>/ir_frames.zip
        string zipFile = dirPath + "/ir_frames.zip";

        // Sprawdzamy, czy plik ZIP istnieje
        if (std::filesystem::exists(zipFile)) {
            cout << "Przetwarzanie pliku ZIP: " << zipFile << endl;
            processZipFile(zipFile, roi);
        } else {
            cout << "Błąd: Plik ZIP " << zipFile << " nie istnieje!" << endl;
        }
    }

    return 0;
}
