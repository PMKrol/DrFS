/*
 * DrFS all in one (cut, align, stack)
 * g++ -std=c++17 -o drfs drfs.cpp `pkg-config --cflags --libs opencv4` -lexiv2
./drfs --align --stack 2025.03.31_15.23*
 * 
 * 03-04.2025
 * 
 * added MINIMUM_MATCHES check to remove completely blurred photos
    if(matches.size() > MINIMUM_MATCHES){
        std::cout << "Matches size: " << matches.size() << std::endl;
    }else{
        scale = -1;
        return;
    }
    
 * changed estimateAffinePartial2D to use all point instead of best 15%:
        const int numGoodMatches = matches.size() * GOOD_MATCHES_RATIO;
        
 * hugely increased Blur kernel on canny images (50 -> 250)
 
*/
    

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <exiv2/exiv2.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <algorithm>

#include <atomic>
#include <fstream>
#include <thread>
#include <unordered_map>

namespace fs = std::filesystem;

using namespace cv;
using namespace std;

#define CANNY_THRESH1 100   // Dolny próg Canny'ego
#define CANNY_THRESH2 150   // Górny próg Canny'ego
#define CANNY_KERNEL 64
#define CANNY_TRESHOLD 64

#define MINIMUM_MATCHES 499

#define GOOD_MATCHES_RATIO 1
#define BLUR_KERNEL_SIZE 150

void alignImageAffine(const cv::Mat& baseImage, const cv::Mat& srcImage, cv::Mat& result, cv::Point2f& shift, float& scale) {
    // Convert images to grayscale
    cv::Mat baseGray, srcGray;
    cv::cvtColor(baseImage, baseGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(srcImage, srcGray, cv::COLOR_BGR2GRAY);
    
    
     //std::cout << "1" << std::endl;

    // ORB feature detection and descriptor computation
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypointsBase, keypointsSrc;
    cv::Mat descriptorsBase, descriptorsSrc;
    orb->detectAndCompute(baseGray, cv::noArray(), keypointsBase, descriptorsBase);
    orb->detectAndCompute(srcGray, cv::noArray(), keypointsSrc, descriptorsSrc);

    // Matching descriptors using BFMatcher
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsBase, descriptorsSrc, matches);
    std::sort(matches.begin(), matches.end());
     
    if(matches.size() > MINIMUM_MATCHES){
        //std::cout << "Matches size: " << matches.size() << std::endl;
    }else{
        scale = -1;
        return;
    }

    // Keep only the best matches
    const int numGoodMatches = matches.size() * GOOD_MATCHES_RATIO;
    matches.erase(matches.begin() + numGoodMatches, matches.end());

    // Extract point locations from matches
    std::vector<cv::Point2f> pointsBase, pointsSrc;
    for (size_t i = 0; i < matches.size(); ++i) {
        pointsBase.push_back(keypointsBase[matches[i].queryIdx].pt);
        pointsSrc.push_back(keypointsSrc[matches[i].trainIdx].pt);
    }
     //std::cout << "3" << std::endl;

    // Compute affine transformation matrix
    cv::Mat affineMatrix = cv::estimateAffinePartial2D(pointsSrc, pointsBase);

    // Warp the source image
    cv::warpAffine(srcImage, result, affineMatrix, baseImage.size());

    // Calculate the translation shift from affine matrix
    shift.x = affineMatrix.at<double>(0, 2);
    shift.y = affineMatrix.at<double>(1, 2);
     //std::cout << "4" << std::endl;
    
    scale = std::sqrt(std::pow(affineMatrix.at<double>(0, 0), 2) + std::pow(affineMatrix.at<double>(1, 0), 2));
}

bool copyFile(const std::string& srcPath, const std::string& destPath) {
    try {
        std::filesystem::copy(srcPath, destPath, std::filesystem::copy_options::overwrite_existing);
        return true;
    } catch (std::filesystem::filesystem_error& e) {
        std::cerr << "Error copying file: " << e.what() << std::endl;
        return false;
    }
}

void align_images(const std::string& directory) {
    std::filesystem::path wipDir = std::filesystem::path(directory) / "wip";
    if (!std::filesystem::exists(wipDir)) {
        std::cerr << "Katalog wip nie istnieje, tworzenie katalogu..." << std::endl;
        std::filesystem::create_directory(wipDir);
    } else {
        std::cout << wipDir << std::endl;
    }

    // Znajdowanie plików *.cut.png
    std::vector<std::filesystem::path> cutFiles;
    for (const auto& entry : std::filesystem::directory_iterator(wipDir)) {
        std::string filePath = entry.path().string();

        if (filePath.find(".cut.png") != std::string::npos &&
            filePath.find("Aligned") == std::string::npos &&
            filePath.find(".txt") == std::string::npos) {
            std::cout << "Plik znaleziony: " << filePath << std::endl;
            cutFiles.push_back(entry.path());
        }
    }

    if (cutFiles.empty()) {
        std::cerr << "Nie znaleziono plików *.cut.png w katalogu " << wipDir << std::endl;
        return;
    }

    std::sort(cutFiles.begin(), cutFiles.end()); // Sortowanie naturalne

    size_t middleIndex = cutFiles.size() / 2;
    cv::Mat baseImage = cv::imread(cutFiles[middleIndex].string());
    if (baseImage.empty()) {
        std::cerr << "Błąd przy wczytywaniu obrazu bazowego: " << cutFiles[middleIndex] << std::endl;
        return;
    }

    // Zapisanie obrazu bazowego z oryginalnym numerem
    std::string baseImageFile = (wipDir / ("Aligned_" + cutFiles[middleIndex].filename().string())).string();
    cv::imwrite(baseImageFile, baseImage);
    std::cout << "Base image saved with original number: " << baseImageFile << std::endl;

    cv::Mat result;
    cv::Point2f shift;
    float scale = 1.0f;

    // Wyrównanie w dół (niższe indeksy)
    for (int i = static_cast<int>(middleIndex) - 1; i >= 0; --i) {
        cv::Mat srcImage = cv::imread(cutFiles[i].string());
        if (srcImage.empty()) {
            std::cerr << "Błąd przy wczytywaniu obrazu: " << cutFiles[i] << std::endl;
            continue;
        }

        std::cout << "Processing: " << cutFiles[i].string() << std::endl;
        alignImageAffine(baseImage, srcImage, result, shift, scale);

        if (scale == -1) {
            std::cerr << "Alignment failed for: " << cutFiles[i].string() << std::endl;
            break;
        }

        // Zapisz wynik
        std::string alignedFile = (wipDir / ("Aligned_" + cutFiles[i].filename().string())).string();
        cv::imwrite(alignedFile, result);

        // Zapisz plik logu
        std::ofstream logFile(alignedFile + ".txt");
        logFile << "Shift: " << shift << "\nScale: " << scale << std::endl;

        baseImage = result; // Ustaw bieżący obraz jako bazowy dla kolejnego wyrównania
    }

    // Reset bazy do środkowego obrazu
    baseImage = cv::imread(cutFiles[middleIndex].string());

    // Wyrównanie w górę (wyższe indeksy)
    for (size_t i = middleIndex + 1; i < cutFiles.size(); ++i) {
        cv::Mat srcImage = cv::imread(cutFiles[i].string());
        if (srcImage.empty()) {
            std::cerr << "Błąd przy wczytywaniu obrazu: " << cutFiles[i] << std::endl;
            continue;
        }

        std::cout << "Processing: " << cutFiles[i].string() << std::endl;
        alignImageAffine(baseImage, srcImage, result, shift, scale);

        if (scale == -1) {
            std::cerr << "Alignment failed for: " << cutFiles[i].string() << std::endl;
            break;
        }

        // Zapisz wynik
        std::string alignedFile = (wipDir / ("Aligned_" + cutFiles[i].filename().string())).string();
        cv::imwrite(alignedFile, result);

        // Zapisz plik logu
        std::ofstream logFile(alignedFile + ".txt");
        logFile << "Shift: " << shift << "\nScale: " << scale << std::endl;

        baseImage = result; // Ustaw bieżący obraz jako bazowy dla kolejnego wyrównania
    }
}



// Funkcja dummy, która nic nie robi (dla plików już przyciętych)
void cut_dummyProcessImage(const std::string& filePath, const cv::Rect& roi) {
    std::cout << "Plik " << filePath << " już istnieje, pomijam przycinanie." << std::endl;
}

std::string cut_formatFilename(const std::string& filename) {
    std::string formattedName = filename;
    std::size_t pos = formattedName.find_last_of("0123456789");
    
    if (pos != std::string::npos && pos > 0) {
        std::string prefix = formattedName.substr(0, pos);
        std::string suffix = formattedName.substr(pos);

        if (suffix.length() == 1) {
            if (pos > 0 && !isdigit(formattedName[pos - 1])) {
                suffix.insert(0, "0"); 
            }
        }

        formattedName = prefix + suffix;
    }
    return formattedName;
}

void cut_processImage(const std::string& filePath, const cv::Rect& roi, bool cutFlag) {
    fs::path inputPath(filePath);
    fs::path wipDir = inputPath.parent_path() / "wip";
    std::string outputFileName = cut_formatFilename(inputPath.stem().string()) + ".cut.png";
    fs::path outputFilePath = wipDir / outputFileName;

    // Sprawdzenie, czy plik przycięty już istnieje
    if (!cutFlag && fs::exists(outputFilePath)) {
        cut_dummyProcessImage(filePath, roi);
        return; // Jeśli plik przycięty już istnieje i --cut nie zostało podane, nic nie robimy
    }

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

    imageFile->readMetadata();
    Exiv2::ExifData& exifData = imageFile->exifData();

    if (exifData.empty()) {
        std::cerr << "Błąd: Brak danych EXIF w pliku: " << filePath << std::endl;
        return;
    }

    int orientation = 1;
    Exiv2::ExifKey key("Exif.Image.Orientation");

    Exiv2::ExifData::const_iterator pos = exifData.findKey(key);
    if (pos != exifData.end()) {
        orientation = pos->value().toLong();
    }

    if (orientation == 3) {
        rotate(image, image, ROTATE_180);
    } else if (orientation == 6) {
        rotate(image, image, ROTATE_90_COUNTERCLOCKWISE);
    } else if (orientation == 8) {
        rotate(image, image, ROTATE_90_CLOCKWISE);
    }

    // Sprawdzenie, czy ROI jest prawidłowe
    if (roi.x + roi.width > image.cols || roi.y + roi.height > image.rows) {
        std::cerr << "Błąd: ROI wykracza poza granice obrazu: " << filePath << std::endl;
        return;
    }

    // Przycinanie obrazu
    cv::Mat croppedImage = image(roi);

    // Tworzenie katalogu 'wip', jeśli nie istnieje
    if (!fs::exists(wipDir)) {
        fs::create_directory(wipDir);
    }

    // Zapisanie przyciętego obrazu
    cv::imwrite(outputFilePath.string(), croppedImage);
    std::cout << "Zapisano przycięty obraz: " << outputFilePath << std::endl;
}

void cut_processDirectory(const fs::path& dirPath, const cv::Rect& roi, bool cutFlag) {
    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        std::cerr << "Błąd: Nieprawidłowy katalog: " << dirPath << std::endl;
        return;
    }

    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg") {
                cut_processImage(entry.path().string(), roi, cutFlag);
            }
        }
    }
}

// Funkcja do znalezienia wspólnej części nazw plików
std::string findCommonPrefix(const std::vector<std::string>& filenames) {
    if (filenames.empty()) return "";

    std::string prefix = filenames[0];
    for (size_t i = 1; i < filenames.size(); ++i) {
        size_t j = 0;
        while (j < prefix.length() && j < filenames[i].length() && prefix[j] == filenames[i][j]) {
            ++j;
        }
        prefix = prefix.substr(0, j); // skracamy prefiks do długości pasującej części
        if (prefix.empty()) break; // brak wspólnego prefiksu
    }
    
    // Sprawdź, czy ostatni znak to '-'
    if (!prefix.empty() && prefix.back() == '-') {
        prefix.pop_back(); // Usuń ostatni znak
    }
    
    return prefix;
}

// Funkcja do znalezienia plików do stackowania w podkatalogu 'wip'
std::vector<std::string> findAlignFiles(const std::string& baseDir) {
    std::vector<std::string> alignFiles;
    std::filesystem::path basePath(baseDir);

    // Sprawdź, czy katalog istnieje
    if (!std::filesystem::exists(basePath)) {
        std::cerr << "Directory does not exist: " << baseDir << std::endl;
        return alignFiles;
    }

    // Znajdź podkatalog 'wip'
    std::filesystem::path wipDir = basePath / "wip";
    if (!std::filesystem::exists(wipDir)) {
        std::cerr << "Subdirectory 'wip' not found in " << baseDir << std::endl;
        return alignFiles;
    }
    
    // Szukaj plików .png, których nazwa zaczyna się od 'Align'
    std::string prefix = "Aligned";
    //std::cout << "Looking for " + prefix + " files..." << std::endl;
    
    for (const auto& entry : std::filesystem::directory_iterator(wipDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            std::string filename = entry.path().filename().string();
            // Sprawdź, czy plik zaczyna się od prefiksu
            if (filename.rfind(prefix, 0) == 0) {  // rfind z 0 sprawdza, czy ciąg zaczyna się od 'prefix'
                alignFiles.push_back(entry.path().string());
            }
        }else{
            //std::cout << entry.path().filename().string() << std::endl;
        }
    }

    return alignFiles;
}

// Funkcja do zapisywania wyniku
void saveResult(const cv::Mat& result, const std::vector<std::string>& filenames, const std::string& method) {
    // Znajdź wspólny prefiks nazw plików wejściowych
    std::string commonPrefix = findCommonPrefix(filenames);
    
    // Usuń ścieżkę z commonPrefix
    std::filesystem::path commonPath = commonPrefix; 
    commonPrefix = commonPath.stem().string(); // Użyj tylko nazwy pliku bez rozszerzenia

    if (commonPrefix.empty()) commonPrefix = "result";

    // Ustal ścieżkę do katalogu wip
    std::filesystem::path firstFilePath = filenames[0];
    std::filesystem::path wipDir = firstFilePath.parent_path(); // katalog wip

    // Sprawdź, czy katalog wip istnieje
    if (!std::filesystem::exists(wipDir)) {
        std::cerr << "Directory wip does not exist in " << wipDir << std::endl;
        return;
    }

    // Zapisz wynik w katalogu wip
    std::string outputFileName = "Stack_" + commonPrefix + ".png"; // Użyj metody i wspólnego prefiksu
    std::filesystem::path outputPath = wipDir / outputFileName;

    cv::imwrite(outputPath.string(), result);
    std::cout << "Stacked image saved as " << outputPath.string() << std::endl;
}


// Funkcja do wczytywania obrazów z katalogu
std::vector<cv::Mat> loadImagesFromDirectory(const std::string& dirPath, std::vector<std::string>& filenames) {
    std::vector<cv::Mat> images;
    std::vector<std::string> alignFiles = findAlignFiles(dirPath);

    for (const std::string& filepath : alignFiles) {
        cv::Mat img = cv::imread(filepath, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Error loading image: " << filepath << std::endl;
            continue;
        }
        images.push_back(img);
        filenames.push_back(filepath); // Save filename
        std::cout << "Loaded image: " << filepath << std::endl;  // Display loaded image
    }

    // Combine images with filenames in pairs
    std::vector<std::pair<std::string, cv::Mat>> imagePairs;
    for (size_t i = 0; i < images.size(); ++i) {
        imagePairs.emplace_back(filenames[i], images[i]);
    }

    // Sort the pairs based on the filename
    std::sort(imagePairs.begin(), imagePairs.end(),
              [](const auto& a, const auto& b) {
                  return a.first < b.first;
              });

    // Clear and refill the images and filenames vectors with sorted data
    images.clear();
    filenames.clear();
    for (const auto& pair : imagePairs) {
        filenames.push_back(pair.first);
        images.push_back(pair.second);
    }

    return images;
}

// Function to fill 0-value pixels with the value of the nearest neighbor
void fillZeroValues(cv::Mat& matrix) {
    cv::Mat filledMatrix = matrix.clone();  // Make a copy for the output

    // Define a queue to hold pixels to process, storing pixel coordinates (y, x)
    std::queue<std::pair<int, int>> processingQueue;

    // Initialize queue with all non-zero pixels
    for (int y = 0; y < matrix.rows; ++y) {
        for (int x = 0; x < matrix.cols; ++x) {
            if (matrix.at<uchar>(y, x) != 0) {
                processingQueue.push({y, x});
            }
        }
    }

    // Directions for 4-neighbor connectivity
    const std::vector<std::pair<int, int>> directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    // Perform BFS to fill in zero-value pixels
    while (!processingQueue.empty()) {
        auto [y, x] = processingQueue.front();
        processingQueue.pop();

        uchar fillValue = matrix.at<uchar>(y, x);

        // Check all 4 neighbors
        for (const auto& [dy, dx] : directions) {
            int ny = y + dy;
            int nx = x + dx;

            // If within bounds and the neighbor is zero, fill it
            if (ny >= 0 && ny < matrix.rows && nx >= 0 && nx < matrix.cols) {
                if (matrix.at<uchar>(ny, nx) == 0) {
                    filledMatrix.at<uchar>(ny, nx) = fillValue;
                    matrix.at<uchar>(ny, nx) = fillValue;  // Update the original matrix to avoid reprocessing
                    processingQueue.push({ny, nx});
                }
            }
        }
    }

    // Copy back filled data to the original matrix
    filledMatrix.copyTo(matrix);
}

// Function to normalize brightness of a group of images
void normalizeImages(std::vector<cv::Mat>& images) {
    double totalMeanBrightness = 0.0;
    std::vector<double> imageBrightness(images.size());

    // Calculate the mean brightness for each image and the total mean
    for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat grayImage;
        if (images[i].channels() == 3) {
            cv::cvtColor(images[i], grayImage, cv::COLOR_BGR2GRAY);
        } else {
            grayImage = images[i];
        }
        imageBrightness[i] = cv::mean(grayImage)[0];
        totalMeanBrightness += imageBrightness[i];
    }

    // Calculate the target brightness as the average brightness across all images
    double targetBrightness = totalMeanBrightness / images.size();

    // Adjust the brightness of each image to match the target brightness
    for (size_t i = 0; i < images.size(); ++i) {
        double brightnessFactor = targetBrightness / imageBrightness[i];

        // Scale image intensities by the brightness factor
        images[i].convertTo(images[i], -1, brightnessFactor, 0);
    }
}


//Metoda 11: jak 10, tylko na podstawie canny
cv::Mat calculateMostFrequentNeighborhood(const cv::Mat& indexMatrix, int radius = 5) {
    cv::Mat resultMatrix = indexMatrix.clone();  // Create a copy of the input matrix

    // Define the kernel size as 2*radius + 1 to cover the neighborhood ± radius
    int kernelSize = 2 * radius + 1;

    // Iterate over each pixel in the matrix (skipping the borders)
    for (int y = radius; y < indexMatrix.rows - radius; ++y) {
        for (int x = radius; x < indexMatrix.cols - radius; ++x) {
            // Create a map to store the frequency of each value in the neighborhood
            std::unordered_map<int, int> frequencyMap;

            // Traverse the neighborhood within the kernel size
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int value = indexMatrix.at<uchar>(y + dy, x + dx);  // Get the pixel value
                    frequencyMap[value]++;  // Increment the frequency of this value
                }
            }

            // Find the most frequent value in the map
            int mostFrequentValue = -1;
            int maxFrequency = 0;

            for (const auto& pair : frequencyMap) {
                if (pair.second > maxFrequency) {
                    mostFrequentValue = pair.first;
                    maxFrequency = pair.second;
                }
            }

            // Set the result matrix pixel to the most frequent value
            resultMatrix.at<uchar>(y, x) = static_cast<uchar>(mostFrequentValue);
        }
    }

    return resultMatrix;
}

void normalizeAndThreshold(cv::Mat& edgeSumImage) {
    // Sprawdzenie, czy macierz jest odpowiedniego typu
    if (edgeSumImage.type() != CV_8U) {
        std::cerr << "Macierz musi być typu CV_8U!" << std::endl;
        return;
    }

    // Znalezienie minimalnej i maksymalnej wartości w obrazie
    double minVal, maxVal;
    cv::minMaxLoc(edgeSumImage, &minVal, &maxVal);  // minVal i maxVal to wartości minimalne i maksymalne w obrazie

    // Rozciągamy wartości na zakres 0-255
    cv::Mat normalizedImage = edgeSumImage.clone();
    normalizedImage.convertTo(normalizedImage, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

    // Przycinamy wartości mniejsze niż próg THRESHOLD_VALUE
    for (int y = 0; y < normalizedImage.rows; ++y) {
        for (int x = 0; x < normalizedImage.cols; ++x) {
            if (normalizedImage.at<uchar>(y, x) < CANNY_TRESHOLD) {
                normalizedImage.at<uchar>(y, x) = 0;  // Ustawiamy piksel na 0, jeśli jest poniżej progu
            }
        }
    }

    // Przypisanie wyniku do oryginalnego obrazu
    edgeSumImage = normalizedImage.clone();
}

void computeImageMetricsParallelCanny(
    const std::vector<cv::Mat>& normalizedImages,
    int kernelSize,
    std::vector<cv::Mat>& cannyWeight) {  // Use cv::Mat for image storage

    // Przygotowanie wektorów wynikowych
    std::vector<cv::Mat> grayImages(normalizedImages.size());
    std::vector<cv::Mat> cannyImages(normalizedImages.size());
    std::vector<cv::Mat> edgeSumImages(normalizedImages.size());

    // Użycie zmiennej do synchronizacji wątków
    std::vector<std::thread> threads;
    std::atomic<int> activeThreads(0);

    // Inicjalizacja cannyWeight dla każdego obrazu (wektor obrazów, nie zmiennych)
    cannyWeight.resize(normalizedImages.size());  // Resize to hold cv::Mat, not float

    //way faster than for-for.
    for (size_t i = 0; i < normalizedImages.size(); ++i) {
        // Krok 1: Konwersja na obraz w skali szarości
        cv::cvtColor(normalizedImages[i], grayImages[i], cv::COLOR_BGR2GRAY);

        // Krok 2: Wykonanie Canny'ego
        cv::Canny(grayImages[i], cannyImages[i], CANNY_THRESH1, CANNY_THRESH2);

        // Krok 3: Obliczanie sumy krawędzi w otoczeniu piksela
        edgeSumImages[i] = cv::Mat::zeros(cannyImages[i].size(), CV_32F);  // Obraz na wynik sumy krawędzi

        // TODO sprawdzić GaussianBlur
        // Zastosowanie blura (rozmycia) do obrazu krawędziowego
        cv::blur(cannyImages[i], edgeSumImages[i], cv::Size(BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE));

        normalizeAndThreshold(edgeSumImages[i]);
        
        // Zapisujemy wynik w wektorze cannyWeight, gdzie każdy element jest obrazem Canny'ego
        cannyWeight[i] = edgeSumImages[i];  // Assign the sum of edges (Mat)

        std::cout << "Processing image " << i + 1 << " of " << normalizedImages.size() << std::endl;
    }
    
    std::cout << "Typ danych norm: " << normalizedImages[0].type() << std::endl;
    std::cout << "Typ danych norm: " << grayImages[0].type() << std::endl;
    std::cout << "Typ danych norm: " << cannyImages[0].type() << std::endl;
    std::cout << "Typ danych norm: " << edgeSumImages[0].type() << std::endl;

}


void saveMatToTextFile(const cv::Mat& mat, const std::string& filename) {
    // Sprawdzamy, czy macierz jest jednowymiarowa (np. 1xN lub Nx1)
    if (mat.empty()) {
        std::cerr << "Mat is empty!" << std::endl;
        return;
    }

    // Otwieramy plik do zapisu
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    // Iterujemy po wszystkich elementach macierzy i zapisujemy je do pliku
    for (int i = 0; i < mat.total(); ++i) {
        outFile << mat.at<float>(i);  // Zapisujemy wartość (można dostosować typ do swoich potrzeb, np. uchar)
        
        // Dodajemy separator (spacja lub tabulator)
        if (i < mat.total() - 1) {
            outFile << " ";  // Spacja jako separator
        }
    }

    outFile.close();
    std::cout << "Mat was saved to " << filename << std::endl;
}

void replaceZerosWithMostFrequentValue(cv::Mat& indexMatrix, int maxIterations = 100) {
    // Sprawdzenie, czy macierz jest odpowiedniego typu
    if (indexMatrix.type() != CV_8U) {
        std::cerr << "Macierz musi być typu CV_8U!" << std::endl;
        return;
    }else{
        std::cout << "Macierz ok." << std::endl;
    }

    int rows = indexMatrix.rows;
    int cols = indexMatrix.cols;

    // Macierz pomocnicza do przechowywania nowych wartości
    cv::Mat newIndexMatrix = indexMatrix.clone();

    // Iteracje przetwarzania
    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        bool changed = false;
        std::cout << "It1." << std::endl;

        // Przechodzimy przez każdy piksel
        for (int y = 1; y < rows - 1; ++y) {  // Od 1 do rows-1, żeby uniknąć brzegów
            for (int x = 1; x < cols - 1; ++x) {  // Od 1 do cols-1, żeby uniknąć brzegów
                //std::cout << std::to_string(indexMatrix.at<uchar>(y, x)) << std::endl;
            
                if (indexMatrix.at<uchar>(y, x) == 0) {
                    
                    //std::cout << "0" << std::endl;
                    std::map<int, int> valueCount;  // Mapa do zliczania wartości w sąsiedztwie

                    // Zliczanie wartości w sąsiedztwie 3x3 (okno wokół piksela)
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            int neighborValue = indexMatrix.at<uchar>(y + dy, x + dx);
                            if (neighborValue != 0) {  // Pomijamy wartość 0, bo to jest aktualizowany piksel
                                valueCount[neighborValue]++;
                            }
                        }
                    }

                    // Szukamy wartości, która występuje najczęściej
                    int mostFrequentValue = -1;
                    int maxCount = -1;
                    for (const auto& entry : valueCount) {
                        if (entry.second > maxCount) {
                            maxCount = entry.second;
                            mostFrequentValue = entry.first;
                        }
                    }

                    // Jeśli znaleziono najczęściej występującą wartość, zapisujemy ją do macierzy pomocniczej
                    if (mostFrequentValue != -1) {
                        newIndexMatrix.at<uchar>(y, x) = mostFrequentValue;
                        changed = true;
                    }
                }
            }
        }

        // Jeśli nic się nie zmieniło, kończymy iterację
        if (!changed) {
            break;
        }else{
            std::cout << "Found zeros..." << std::endl;
        }

        // Zastosowanie nowych wartości z macierzy pomocniczej do głównej macierzy
        indexMatrix = newIndexMatrix.clone();
    }
}


cv::Mat stackWithCanny(const std::vector<cv::Mat>& images, const std::string& output_dir = "./", int kernelSize = 5){
    if (images.empty()) {
        std::cerr << "Błąd: Brak obrazów do przetworzenia." << std::endl;
        return cv::Mat();
    }

    std::cout << "Normalizing images." << std::endl;
    std::vector<cv::Mat> normalizedImages = images;
    normalizeImages(normalizedImages);
    std::cout << "Done" << std::endl;

    int rows = normalizedImages[0].rows;
    int cols = normalizedImages[0].cols;

    std::vector<cv::Mat> cannyWeight;
    std::cout << "Calculating cannyWeight." << std::endl;

    computeImageMetricsParallelCanny(normalizedImages, kernelSize, cannyWeight);

    for (int i = 0; i < cannyWeight.size(); ++i) {
        //std::cout << normalizedImages[i].size() << " vs " << cannyWeight[i].size() << std::endl;

        //std::cout << "Saving " << i << "dbg txt." << std::endl;
        //saveMatToTextFile(cannyWeight[i], output_dir + "/dbg_canny-" + std::to_string(i) + ".txt");

        //std::cout << "Saving " << i << "dbg image." << std::endl;
        cv::imwrite(output_dir + "/dbg_matrixCanny" + std::to_string(i) + ".png", cannyWeight[i]);
    }

    std::cout << "Done." << std::endl;

    cv::Mat indexMatrix = cv::Mat::zeros(rows, cols, CV_8U);

    // Tworzenie indexMatrix na podstawie wartości Canny
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int max_idx = 0;

            //          !!! TODO: if all zeros!     !!
            double max_value = 0.0;

            for (int i = 0; i < cannyWeight.size(); ++i) {
                double canny_value = cannyWeight[i].at<uchar>(y, x);
                if (canny_value > max_value) {
                    max_value = canny_value;
                    max_idx = i;
                }
            }

            if (max_value == 0.0) {
                // Obsługuje przypadek, gdy wszystkie wartości w cannyWeight[i] są zerowe
                indexMatrix.ptr<uchar>(y)[x] = 0;  // Możesz ustawić 0, jeśli chcesz oznaczyć brak krawędzi
            } else {
                indexMatrix.ptr<uchar>(y)[x] = max_idx + 1;  // Indeksowanie 1-based
            }
        }
    }
    
    // Zapis do pliku tekstowego (debugging)
    //saveMatToTextFile(indexMatrix, output_dir + "/dbg_indexMatrix.txt");
    //cv::imwrite(output_dir + "/dbg_indexMatrixCanny.png", indexMatrix);

    //std::cout << "Done1." << std::endl;
    //solve zeros:
    //indexMatrix = calculateAverageNeighborhood(indexMatrix.clone(), 50);
    //indexMatrix = calculateMostFrequentNeighborhood(indexMatrix.clone(), CANNY_KERNEL);
    
    //replaceZerosWithMostFrequentValue(indexMatrix);
    fillZeroValues(indexMatrix);
    //cv::imwrite(output_dir + "/dbg_indexMatrixFilled.png", indexMatrix);

    // Naprawiona normalizacja i zapis
    cv::Mat scaledIndexMatrix;
    cv::normalize(indexMatrix, scaledIndexMatrix, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::imwrite(output_dir + "/dbg_indexMatrixNorm.png", scaledIndexMatrix);

    // Debugowanie: zapis nieskalowanej macierzy
    //cv::imwrite("indexMatrixCannyUnscaled.png", indexMatrix);
    //std::cout << "Done2." << std::endl;

    //std::cout << "Done3." << std::endl;
    // Zapis macierzy znormalizowanej
    //cv::imwrite(output_dir + "/dbg_indexMatrixCannyNorm.png", scaledIndexMatrix);
    
    //std::cout << "Done3." << std::endl;

    // Tworzenie obrazu wynikowego
    cv::Mat result = cv::Mat::zeros(rows, cols, normalizedImages[0].type());
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int idx = indexMatrix.at<uchar>(y, x);
            if (idx > 0) {
                result.at<cv::Vec3b>(y, x) = normalizedImages[idx - 1].at<cv::Vec3b>(y, x);
            } else {
                cv::Vec3d averagePixel(0, 0, 0);
                for (const auto& img : normalizedImages) {
                    averagePixel += img.at<cv::Vec3b>(y, x);
                }
                result.at<cv::Vec3b>(y, x) = averagePixel / static_cast<double>(images.size());
                //TODO
                //std::cout << "Should be none." <<std::endl;
            }
        }
    }

    return result;
}

void stack_images(const std::string& dirPath) {
    std::cout << "Processing directory: " << dirPath << std::endl;

    // Wczytaj obrazy z podkatalogu "wip"
    std::vector<std::string> filenames;
    std::vector<cv::Mat> images = loadImagesFromDirectory(dirPath, filenames); // "11" jako aX

    if (images.empty()) {
        std::cerr << "No valid images found in directory: " << dirPath << std::endl;
        return;
    }

    // Wyświetl listę plików wybranych do stakowania
    std::cout << "Images selected for stacking:" << std::endl;
    for (const auto& filepath : filenames) {
        std::cout << " - " << filepath << std::endl;
    }

    // Przetwarzanie obrazów metodą "-m11"
    cv::Mat result = stackWithCanny(images, dirPath + "/wip", CANNY_KERNEL);

    if (result.empty()) {
        std::cerr << "Failed to stack images in directory: " << dirPath << std::endl;
        return;
    }

    // Zapisz wynik w katalogu "wip"
    saveResult(result, filenames, "-m11");
}



void printUsage() {
    std::cerr << "Użycie: ./program [--cut] [--align] [--stack] [x,y,w,h] katalog1 [katalog2 ...]" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    // Obsługa --cut
    bool cut = false;
    cv::Rect roi;

    if (std::string(argv[1]) == "--cut") {
        cut = true;
        argc--; argv++;

        // Parsowanie ROI, jeśli zostało podane
        if (argc > 1) {
            int roiX, roiY, roiW, roiH;
            if (sscanf(argv[1], "%d,%d,%d,%d", &roiX, &roiY, &roiW, &roiH) == 4) {
                roi = cv::Rect(roiX, roiY, roiW, roiH);
                argc--; argv++;
            } else {
                std::cerr << "Invalid ROI format. Expected format: x,y,w,h" << std::endl;
                printUsage();
                return 1;
            }
        } else {
            std::cerr << "Missing ROI parameters after --cut." << std::endl;
            printUsage();
            return 1;
        }
        
        cout << "Will cut." << std::endl;
    }

    // Obsługa --align
    bool align = false;
    if (argc > 1 && std::string(argv[1]) == "--align") {
        align = true;
        argc--; argv++;
        cout << "Will align." << std::endl;
    }

    // Obsługa --stack
    bool stack = false;
    if (argc > 1 && std::string(argv[1]) == "--stack") {
        stack = true;
        argc--; argv++;
        cout << "Will stack." << std::endl;
    }

    // Obsługa --edges_cut
    /*bool edgesCut = false;
    if (argc > 1 && std::string(argv[1]) == "--edges_cut") {
        edgesCut = true;
        argc--; argv++;
    }*/

    // Obsługuje każdy katalog
    for (int i = 1; i < argc; ++i) {
        std::filesystem::path dirPath(argv[i]);

        if (cut) {
            cout << "Starting cut stage." << std::endl;
            // Funkcja przetwarzająca przycinanie obrazów
            cut_processDirectory(dirPath, roi, cut);
        }
        
        if (align) {
            cout << "Starting align stage." << std::endl;
            // Funkcja wyrównania obrazów, metoda przekazywana jako "-a12"
            align_images(argv[i]);
        }
        
        if (stack) {
            cout << "Starting stack stage." << std::endl;
            // Funkcja obsługująca tworzenie stosu obrazów
            stack_images(dirPath);
        }// else if (edgesCut) {
            // Funkcja obsługująca przycinanie krawędzi obrazów
            //edges_cut(dirPath);
        //}
    }

    return 0;
}


