/*
 * compile: g++ -std=c++17 -o cut cut.cpp `pkg-config --cflags --libs opencv4` -lexiv2
 * run: ./cut 1680,300,1750,1750 0002/
 * 
 * sudo apt-get install libexiv2-dev

 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <exiv2/exiv2.hpp>

namespace fs = std::filesystem;

using namespace cv;
using namespace std;

std::string formatFilename(const std::string& filename) {
    std::string formattedName = filename;
    std::size_t pos = formattedName.find_last_of("0123456789");
    
    // Sprawdź, czy znaleziono cyfrę
    if (pos != std::string::npos && pos > 0) {
        std::string prefix = formattedName.substr(0, pos);
        std::string suffix = formattedName.substr(pos);

        // Sprawdź, czy ostatnia cyfra jest pojedyncza
        if (suffix.length() == 1) {
            // Sprawdź, czy znak przed cyfrą jest cyfrą
            if (pos > 0 && !isdigit(formattedName[pos - 1])) {
                suffix.insert(0, "0"); // Dodaj zero na początku
            }
        }

        formattedName = prefix + suffix; // Połącz nazwy
    }
    return formattedName;
}

// Funkcja przycina obraz zgodnie z zadanym ROI i zapisuje wynik w katalogu 'wip'
void processImage(const std::string& filePath, const cv::Rect& roi) {
    // Wczytaj obraz
    cv::Mat image = cv::imread(filePath);
    if (image.empty()) {
        std::cerr << "Błąd: Nie można wczytać obrazu " << filePath << std::endl;
        return;
    }
    
        // Odczyt EXIF z pliku
    Exiv2::Image::AutoPtr imageFile = Exiv2::ImageFactory::open(filePath);
    if (!imageFile.get()) {
        std::cerr << "Błąd: Nie udało się otworzyć pliku EXIF: " << filePath << std::endl;
        return;
    }

    // Próbujemy wczytać dane EXIF
    imageFile->readMetadata();
    Exiv2::ExifData& exifData = imageFile->exifData();

    // Sprawdzamy, czy dane EXIF zostały wczytane
    if (exifData.empty()) {
        std::cerr << "Błąd: Brak danych EXIF w pliku: " << filePath << std::endl;
        return;
    }

    // Sprawdzenie orientacji EXIF
    int orientation = 1;  // Domyślnie ustawiamy na 1 (brak obrotu)
    Exiv2::ExifKey key("Exif.Image.Orientation");

    // Sprawdzenie, czy istnieje klucz orientacji w EXIF
    Exiv2::ExifData::const_iterator pos = exifData.findKey(key);
    if (pos != exifData.end()) {
        // Jeśli klucz istnieje, odczytujemy jego wartość
        orientation = pos->value().toLong();
    }

    // Obracanie obrazu na podstawie orientacji EXIF
    if (orientation == 3) {
        // Obrót o 180 stopni
        rotate(image, image, ROTATE_180);
    } else if (orientation == 6) {
        // Obrót o 90 stopni w prawo
        rotate(image, image, ROTATE_90_COUNTERCLOCKWISE);
    } else if (orientation == 8) {
        // Obrót o 90 stopni w lewo
        rotate(image, image, ROTATE_90_CLOCKWISE);
    }

    // Sprawdź, czy ROI jest prawidłowe dla obrazu
    if (roi.x + roi.width > image.cols || roi.y + roi.height > image.rows) {
        std::cerr << "Błąd: ROI wykracza poza granice obrazu: " << filePath << std::endl;
        return;
    }

    // Przytnij obraz
    cv::Mat croppedImage = image(roi);

    // Stwórz katalog 'wip' w katalogu wejściowym, jeśli nie istnieje
    fs::path inputPath(filePath);
    fs::path wipDir = inputPath.parent_path() / "wip";
    if (!fs::exists(wipDir)) {
        fs::create_directory(wipDir);
    }

    // Nazwa pliku wynikowego: 'nazwa_bez_rozszerzenia.cut.png'
    std::string outputFileName = formatFilename(inputPath.stem().string()) + ".cut.png";
    //std::string outputFileName = inputPath.stem().string() + ".cut.png";
    fs::path outputFilePath = wipDir / outputFileName;

    // Zapisz przycięty obraz
    cv::imwrite(outputFilePath.string(), croppedImage);
    std::cout << "Zapisano przycięty obraz: " << outputFilePath << std::endl;
}

// Funkcja przetwarza obrazy w podanym katalogu
void processDirectory(const fs::path& dirPath, const cv::Rect& roi) {
    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        std::cerr << "Błąd: Nieprawidłowy katalog: " << dirPath << std::endl;
        return;
    }

    // Iteracja po plikach w katalogu
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg") {
                processImage(entry.path().string(), roi);
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Użycie: " << argv[0] << " x,y,w,h katalog1 [katalog2 ...]" << std::endl;
        return 1;
    }

    // Parsowanie ROI (x,y,w,h)
    std::string roiStr = argv[1];
    int x, y, w, h;
    if (sscanf(roiStr.c_str(), "%d,%d,%d,%d", &x, &y, &w, &h) != 4) {
        std::cerr << "Błąd: Nieprawidłowy format ROI. Oczekiwano formatu x,y,w,h" << std::endl;
        return 1;
    }

    cv::Rect roi(x, y, w, h);

    // Przetwarzanie każdego podanego katalogu
    for (int i = 2; i < argc; ++i) {
        fs::path dirPath(argv[i]);
        processDirectory(dirPath, roi);
    }

    return 0;
}
