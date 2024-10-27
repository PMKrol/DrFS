#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#define RECT_SIZE 50 // Rozmiar pola do analizy ostrości
#define DILATION_ITER 5 // Ilość iteracji rozszerzenia maski

/* TODO:
 * color normalisation! (https://github.com/PetteriAimonen/focus-stack/blob/master/src/task_align.cc)
 */

/*
 * Based or inspired on:
 * https://github.com/bznick98/Focus_Stacking      1/2
 * https://github.com/cmcguinness/focusstack       1  
 * https://github.com/PetteriAimonen/focus-stack   1
 * https://github.com/abadams/ImageStack           0 (there is no)
 * https://github.com/maitek/image_stacking        1
 */


// Metoda 1: Stacking za pomocą Laplacian (piksel po pikselu)
// as in https://github.com/cmcguinness/focusstack
// as in https://github.com/PetteriAimonen/focus-stack
cv::Mat stackLaplacian(const std::vector<cv::Mat>& images) {
    if (images.empty()) {
        std::cerr << "No images to stack!" << std::endl;
        return cv::Mat();
    }

    // Wstępne przetwarzanie
    std::vector<cv::Mat> laplacians(images.size());
    std::vector<cv::Mat> grayImages(images.size());

    for (size_t i = 0; i < images.size(); ++i) {
        cv::cvtColor(images[i], grayImages[i], cv::COLOR_BGR2GRAY);
        cv::Laplacian(grayImages[i], laplacians[i], CV_64F);

        // Uzyskanie wartości absolutnej wyników (aby uniknąć wartości ujemnych)
        cv::Mat absLaplacian;
        cv::convertScaleAbs(laplacians[i], absLaplacian);
        laplacians[i] = absLaplacian; // Zastąpienie oryginalnego Laplace'a przetworzonym
    }

    cv::Mat result = images[0].clone();
    const int rows = result.rows;
    const int cols = result.cols;

    // Wskazówki do wskaźników
    for (int y = 0; y < rows; ++y) {
        // Wyświetl postęp co 10 wierszy
        if (y % 10 == 0) {
            std::cout << "Processing row " << y << " of " << rows << std::endl;
        }

        // Odczytanie wiersza z wyniku
        auto resultRow = result.ptr<cv::Vec3b>(y);

        for (int x = 0; x < cols; ++x) {
            double maxSharpness = 0.0;
            int bestImageIndex = 0;

            for (size_t i = 0; i < images.size(); ++i) {
                double currentSharpness = laplacians[i].at<unsigned char>(y, x); // Używamy unsigned char, ponieważ mamy wartości 0-255
                if (currentSharpness > maxSharpness) {
                    maxSharpness = currentSharpness;
                    bestImageIndex = i;
                }
            }

            resultRow[x] = images[bestImageIndex].at<cv::Vec3b>(y, x); // Ustawianie wyniku
        }
    }

    std::cout << "Processing complete!" << std::endl;
    return result;
}

// Metoda 2: Stacking z użyciem map ostrości i maski
cv::Mat stackWithMask(const std::vector<cv::Mat>& images) {
    if (images.empty()) {
        std::cerr << "No images to stack!" << std::endl;
        return cv::Mat();
    }

    std::vector<cv::Mat> laplacianImages;
    for (const auto& image : images) {
        cv::Mat gray, laplacian;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
        cv::Laplacian(gray, laplacian, CV_64F);
        cv::convertScaleAbs(laplacian, laplacian);
        laplacianImages.push_back(laplacian);
    }

    cv::Mat result = images[0].clone();

    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {
            double maxSharpness = 0.0;
            int bestImageIndex = 0;

            for (size_t i = 0; i < laplacianImages.size(); ++i) {
                double currentSharpness = laplacianImages[i].at<uchar>(y, x);
                if (currentSharpness > maxSharpness) {
                    maxSharpness = currentSharpness;
                    bestImageIndex = i;
                }
            }

            result.at<cv::Vec3b>(y, x) = images[bestImageIndex].at<cv::Vec3b>(y, x);
        }
    }

    return result;
}

// Metoda 3: Funkcja do stakowania obrazów z wykorzystaniem maski ostrości
cv::Mat stackWithSharpnessMask(const std::vector<cv::Mat>& images) {
    if (images.empty()) {
        std::cerr << "No images to stack!" << std::endl;
        return cv::Mat();
    }

    // Wstępne przetwarzanie
    std::vector<cv::Mat> laplacians(images.size());
    std::vector<cv::Mat> grayImages(images.size());

    for (size_t i = 0; i < images.size(); ++i) {
        cv::cvtColor(images[i], grayImages[i], cv::COLOR_BGR2GRAY);
        cv::Laplacian(grayImages[i], laplacians[i], CV_64F);
    }

    // Obliczanie maski ostrości
    cv::Mat sharpnessMask = cv::Mat::zeros(images[0].size(), CV_64F);
    for (size_t i = 0; i < images.size(); ++i) {
        sharpnessMask += laplacians[i]; // Sumujemy ostrość dla każdej warstwy
    }

    // Normalizacja maski ostrości
    cv::normalize(sharpnessMask, sharpnessMask, 0, 1, cv::NORM_MINMAX);

    cv::Mat result = cv::Mat::zeros(images[0].size(), images[0].type());

    const int rows = result.rows;
    const int cols = result.cols;

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            // Odczytanie wartości ostrości dla danego piksela
            double maxSharpness = 0.0;
            int bestImageIndex = 0;

            for (size_t i = 0; i < images.size(); ++i) {
                double currentSharpness = laplacians[i].at<double>(y, x);
                if (currentSharpness > maxSharpness) {
                    maxSharpness = currentSharpness;
                    bestImageIndex = i;
                }
            }

            // Ustawianie wartości w wyniku na podstawie maksymalnej ostrości
            result.at<cv::Vec3b>(y, x) = images[bestImageIndex].at<cv::Vec3b>(y, x);
        }
    }

    std::cout << "Processing complete!" << std::endl;
    return result;
}



// Metoda 4: Funkcja do stakowania obrazów z prostokątami ostrości
// Funkcja do stakowania obrazów z prostokątami ostrości
cv::Mat stackWithRectangles(const std::vector<cv::Mat>& images) {
    if (images.empty()) {
        std::cerr << "No images to stack!" << std::endl;
        return cv::Mat();
    }

    // Upewnij się, że wszystkie obrazy mają ten sam rozmiar
    for (const auto& img : images) {
        if (img.size() != images[0].size()) {
            std::cerr << "All images must have the same size!" << std::endl;
            return cv::Mat();
        }
    }

    const int rows = images[0].rows;
    const int cols = images[0].cols;
    cv::Mat result = cv::Mat::zeros(images[0].size(), images[0].type());

    // Oblicz Laplacian dla każdego obrazu i przechowaj w wektorze
    std::vector<cv::Mat> laplacians(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat gray;
        cv::cvtColor(images[i], gray, cv::COLOR_BGR2GRAY);
        cv::Laplacian(gray, laplacians[i], CV_64F);
    }

    const int rectHalfSize = RECT_SIZE / 2;

    for (int y = 0; y < rows; ++y) {
        // Wyświetl postęp co 10 wierszy
        if (y % 10 == 0) {
            std::cout << "Processing row " << y << " of " << rows << std::endl;
        }

        for (int x = 0; x < cols; ++x) {
            // Wyświetl postęp co 10 kolumn
//             if (x % 10 == 0) {
//                 std::cout << "Processing column " << x << " of " << cols << std::endl;
//             }

            double maxSharpness = 0.0;
            int bestImageIndex = 0;

            // Sprawdzenie granic dla prostokąta
            int startY = std::max(0, y - rectHalfSize);
            int endY = std::min(rows, y + rectHalfSize + 1);
            int startX = std::max(0, x - rectHalfSize);
            int endX = std::min(cols, x + rectHalfSize + 1);

            // Sprawdź każdy obraz i wybierz najlepszy
            for (size_t i = 0; i < images.size(); ++i) {
                double currentSharpness = 0.0;

                // Oblicz ostrość w obrębie prostokąta
                for (int py = startY; py < endY; ++py) {
                    for (int px = startX; px < endX; ++px) {
                        currentSharpness += laplacians[i].at<double>(py, px);
                    }
                }

                // Ustal maksymalną ostrość
                if (currentSharpness > maxSharpness) {
                    maxSharpness = currentSharpness;
                    bestImageIndex = i;
                }
            }

            // Ustaw piksel w wyniku na podstawie najostrzejszego prostokąta
            result.at<cv::Vec3b>(y, x) = images[bestImageIndex].at<cv::Vec3b>(y, x);
        }
    }

    std::cout << "Processing complete!" << std::endl;
    return result;
}

//Metoda 5:
cv::Mat stackWithFloatingMasks(const std::vector<cv::Mat>& images) {
    if (images.empty()) {
        std::cerr << "No images to stack!" << std::endl;
        return cv::Mat();
    }

    // Upewnij się, że wszystkie obrazy mają ten sam rozmiar
    for (const auto& img : images) {
        if (img.size() != images[0].size()) {
            std::cerr << "All images must have the same size!" << std::endl;
            return cv::Mat();
        }
    }

    const int rows = images[0].rows;
    const int cols = images[0].cols;

    // Wynikowy obraz
    cv::Mat result = cv::Mat::zeros(images[0].size(), images[0].type());

    // Tworzymy Laplacian (ostrość) dla każdego obrazu
    std::vector<cv::Mat> sharpnessMaps(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat gray, laplacian;
        cv::cvtColor(images[i], gray, cv::COLOR_BGR2GRAY);
        cv::Laplacian(gray, laplacian, CV_64F);

        // Konwertujemy wynik Laplaciana na wartość absolutną
        cv::convertScaleAbs(laplacian, sharpnessMaps[i]);
    }

    // Tworzymy pustą maskę o rozmiarze obrazów
    cv::Mat combinedMask = cv::Mat::zeros(rows, cols, CV_8U);

    // Tworzymy maski dla każdego obrazu, które oznaczają miejsca o wysokiej ostrości
    for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat mask = cv::Mat::zeros(rows, cols, CV_8U);

        // Przechodzimy przez każdy obraz, sprawdzając ostrość w prostokątach
        for (int y = 0; y < rows - RECT_SIZE; y += RECT_SIZE) {
            for (int x = 0; x < cols - RECT_SIZE; x += RECT_SIZE) {
                // Wyciągamy prostokąt
                cv::Rect roi(x, y, RECT_SIZE, RECT_SIZE);
                cv::Mat region = sharpnessMaps[i](roi);

                // Obliczamy sumaryczną ostrość w regionie
                double sharpnessSum = cv::sum(region)[0];

                // Ustal próg ostrości, aby zadecydować, czy maskować ten obszar
                if (sharpnessSum > 500) { // Próg można regulować
                    mask(roi).setTo(255); // Oznaczamy ten obszar jako ostry
                }
            }
        }

        // Rozszerzamy maski, aby się łączyły
        cv::Mat dilatedMask;
        cv::dilate(mask, dilatedMask, cv::Mat(), cv::Point(-1, -1), DILATION_ITER);

        // Łączymy maski z poprzednimi
        combinedMask = combinedMask | dilatedMask;
    }

    // Tworzymy wynikowy obraz, biorąc z obrazów te fragmenty, które są oznaczone w maskach
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (combinedMask.at<uchar>(y, x) > 0) {
                // Wybierz pierwszy obraz, który zawiera maskę
                for (size_t i = 0; i < images.size(); ++i) {
                    if (sharpnessMaps[i].at<uchar>(y, x) > 0) {
                        result.at<cv::Vec3b>(y, x) = images[i].at<cv::Vec3b>(y, x);
                        break;
                    }
                }
            }
        }
    }

    std::cout << "Processing complete!" << std::endl;
    return result;
}

//Metoda 6: odpowiednik naive_focus_stacking z Focus_Stacking (python)
// as in https://github.com/bznick98/Focus_Stacking (naive_focus_stacking)
cv::Mat stackWithFloatingMasks2(const std::vector<cv::Mat>& images, bool debug = false) {
    if (images.empty()) {
        throw std::runtime_error("No images provided for stacking.");
    }

    // Check if images are in color or grayscale
    bool isColor = (images[0].channels() == 3);

    // Align images - placeholder for align_images equivalent in C++
    std::vector<cv::Mat> aligned_images = images; // Assume already aligned for now

    // Convert to grayscale if the images are colored
    std::vector<cv::Mat> aligned_gray;
    if (isColor) {
        for (const auto& img : aligned_images) {
            cv::Mat gray_img;
            cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
            aligned_gray.push_back(gray_img);
        }
    } else {
        aligned_gray = aligned_images;
    }

    // Apply Gaussian blur on all images
    for (auto& gray_img : aligned_gray) {
        cv::GaussianBlur(gray_img, gray_img, cv::Size(3, 3), 3);
    }

    // Apply Laplacian filter (Laplacian of Gaussian)
    for (auto& gray_img : aligned_gray) {
        cv::Laplacian(gray_img, gray_img, CV_64F, 3);
    }

    if (debug) {
        // Display Laplacian images
        for (size_t i = 0; i < aligned_gray.size(); ++i) {
            cv::imshow("Laplacian Image " + std::to_string(i), aligned_gray[i]);
            cv::waitKey(0);
        }
    }

    // Prepare the canvas for the output image
    cv::Mat canvas = cv::Mat::zeros(images[0].size(), images[0].type());

    // Get the maximum value of the LoG across all images
    cv::Mat max_LoG = cv::Mat::zeros(aligned_gray[0].size(), aligned_gray[0].type());
    for (const auto& gray_img : aligned_gray) {
        cv::max(max_LoG, cv::abs(gray_img), max_LoG);
    }

    // Find masks where the LoG achieves the maximum
    std::vector<cv::Mat> masks;
    for (const auto& gray_img : aligned_gray) {
        cv::Mat mask = (cv::abs(gray_img) == max_LoG);
        masks.push_back(mask);
    }

    if (debug) {
        // Display masks
        for (size_t i = 0; i < masks.size(); ++i) {
            cv::imshow("Mask " + std::to_string(i), masks[i]);
            cv::waitKey(0);
        }
    }

    // Apply masks to blend the images
    for (size_t i = 0; i < aligned_images.size(); ++i) {
        aligned_images[i].copyTo(canvas, masks[i]);
    }

    return canvas;
}

//Metoda 7: na inspirowane (?) stack.py i utils.py, ale nie działa 
// as in https://github.com/bznick98/Focus_Stacking (lap_focus_stacking) - NOT WORKING YET
cv::Mat stackWithLaplacianPyramid(const std::vector<cv::Mat>& images, int N = 3) {
    // Przechowuje obrazy Laplace'a
    std::vector<cv::Mat> LP_f;

    // Iteracja przez każdy obraz
    for (const auto& img : images) {
        // Zmiana formatu obrazu na float, aby zachować precyzję
        cv::Mat imgFloat;
        img.convertTo(imgFloat, CV_32F);

        std::vector<cv::Mat> gaussianPyramid;
        gaussianPyramid.push_back(imgFloat);

        // Tworzenie piramidy Gaussa
        for (int i = 0; i < N; ++i) {
            cv::Mat down;
            cv::pyrDown(gaussianPyramid[i], down);
            gaussianPyramid.push_back(down);
        }

        // Tworzenie piramidy Laplace'a
        for (int i = N; i > 0; --i) {
            cv::Mat up;
            cv::pyrUp(gaussianPyramid[i], up, gaussianPyramid[i - 1].size());
            LP_f.push_back(gaussianPyramid[i - 1] - up);
        }
        // Ostatni poziom piramidy Gaussa
        LP_f.push_back(gaussianPyramid.back());
    }

    // Fuzja obrazów Laplace'a
    cv::Mat fused_img = cv::Mat::zeros(images[0].size(), images[0].type());

    // Dodaj wszystkie obrazy Laplace'a
    for (const auto& laplacian : LP_f) {
        // Sprawdź, czy rozmiar jest zgodny z fused_img
        if (laplacian.size() == fused_img.size()) {
            fused_img += laplacian;
        } else {
            std::cerr << "Rozmiary obrazów nie pasują: " 
                      << laplacian.size() << " != " 
                      << fused_img.size() << std::endl;
        }
    }

    // Normalizacja do zakresu 0-255
    cv::normalize(fused_img, fused_img, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Ulepszenie jasności
    double alpha = 1.5; // Współczynnik kontrastu
    int beta = 30;      // Wartość jasności
    cv::Mat enhanced_img;
    fused_img.convertTo(enhanced_img, CV_8U, alpha, beta); // Zastosowanie kontrastu i jasności

    return enhanced_img;
}


//Metoda 8:  
// inspired by https://github.com/bznick98/Focus_Stacking
// it creates something alike shadowed wimage - it does not look like drill, but might be useful while 
// measuring tool wear.
cv::Mat stackWithLaplacianPyramidShadow(const std::vector<cv::Mat>& images, int N = 3) {
    if (images.empty()) {
        throw std::invalid_argument("List of images is empty.");
    }

    // Sprawdzenie, czy wszystkie obrazy mają ten sam rozmiar
    for (const auto& img : images) {
        if (img.size() != images[0].size()) {
            throw std::invalid_argument("All images must have the same size.");
        }
    }

    // Przechowuje obrazy Laplace'a
    std::vector<cv::Mat> LP_f;
    int numImages = images.size();

    // Iteracja przez każdy obraz
    for (const auto& img : images) {
        // Zmiana formatu obrazu na float, aby zachować precyzję
        cv::Mat imgFloat;
        img.convertTo(imgFloat, CV_32F);

        std::vector<cv::Mat> gaussianPyramid;
        gaussianPyramid.push_back(imgFloat);

        // Tworzenie piramidy Gaussa
        for (int i = 0; i < N; ++i) {
            cv::Mat down;
            cv::pyrDown(gaussianPyramid[i], down);
            gaussianPyramid.push_back(down);
        }

        // Tworzenie piramidy Laplace'a
        for (int i = N; i > 0; --i) {
            cv::Mat up;
            cv::pyrUp(gaussianPyramid[i], up, gaussianPyramid[i - 1].size());
            LP_f.push_back(gaussianPyramid[i - 1] - up);
        }
        // Ostatni poziom piramidy Gaussa
        LP_f.push_back(gaussianPyramid.back());
    }

    // Fuzja obrazów Laplace'a
    cv::Mat fused_img = cv::Mat::zeros(images[0].size(), images[0].type()); // Inicjalizacja jako czarny obraz

    // Dodaj wszystkie obrazy Laplace'a
    for (const auto& laplacian : LP_f) {
        // Upewnij się, że rozmiary są zgodne
        if (laplacian.size() == fused_img.size()) {
            fused_img += laplacian; // Dodaj tylko, jeśli rozmiary są zgodne
        }
    }

    // Normalizacja do zakresu 0-255
    cv::normalize(fused_img, fused_img, 0, 255, cv::NORM_MINMAX);

    // Ulepszenie jasności
    double alpha = 5; // Współczynnik kontrastu
    int beta = 1;      // Wartość jasności
    fused_img.convertTo(fused_img, CV_8U, alpha, beta); // Zastosowanie kontrastu i jasności

    return fused_img;
}

//Metoda 9: na podstawie: https://github.com/maitek/image_stacking/blob/master/auto_stack.py
// wyliczanie średniej z kolorów z wszystkich obrazów.
// as in https://github.com/maitek/image_stacking (both stackImagesKeypointMatching and stackImagesECC)
// as in https://github.com/bznick98/Focus_Stacking
cv::Mat stackByAverage(const std::vector<cv::Mat>& images) {
    if (images.empty()) {
        throw std::invalid_argument("The input vector is empty.");
    }

    // Zakładamy, że wszystkie obrazy mają ten sam rozmiar i typ
    cv::Mat sum = cv::Mat::zeros(images[0].size(), images[0].type());

    // Suma wszystkich obrazów
    for (const auto& image : images) {
        sum += image; // Dodaj każdy obraz do sumy
    }

    // Oblicz średnią
    cv::Mat average = sum / static_cast<double>(images.size());
    // Konwertuj do typu uint8 (zakres [0, 255])
    average.convertTo(average, CV_8U);

    return average;
}

//Metoda 10: 


//Koniec METOD

// Funkcja do znalezienia plików do stackowania w podkatalogu 'wip'
std::vector<std::string> findAlignFiles(const std::string& baseDir, const std::string& aX) {
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
    
    // Szukaj plików .png, których nazwa zaczyna się od 'Align{-aX}'
    std::string prefix = "Aligned-" + aX;
    //std::cout << "Looking for " + prefix + " files..." << std::endl;
    
    for (const auto& entry : std::filesystem::directory_iterator(wipDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            std::string filename = entry.path().filename().string();
            // Sprawdź, czy plik zaczyna się od prefiksu 'Align{-aX}'
            if (filename.rfind(prefix, 0) == 0) {  // rfind z 0 sprawdza, czy ciąg zaczyna się od 'prefix'
                alignFiles.push_back(entry.path().string());
            }
        }else{
            //std::cout << entry.path().filename().string() << std::endl;
        }
    }

    return alignFiles;
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
    return prefix;
}



// Funkcja do wczytywania obrazów z katalogu
std::vector<cv::Mat> loadImagesFromDirectory(const std::string& dirPath, const std::string& aX, std::vector<std::string>& filenames) {
    std::vector<cv::Mat> images;
    std::vector<std::string> alignFiles = findAlignFiles(dirPath, aX);

    for (const std::string& filepath : alignFiles) {
        cv::Mat img = cv::imread(filepath, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Error loading image: " << filepath << std::endl;
            continue;
        }
        images.push_back(img);
        filenames.push_back(filepath); // Zapisz nazwę pliku
        std::cout << "Loaded image: " << filepath << std::endl;  // Wyświetl załadowany obraz
    }

    return images;
}

// Funkcja do przetwarzania obrazów
cv::Mat processImages(const std::vector<cv::Mat>& images, const std::string& method) {
    cv::Mat result;

    if (method == "-m1") {
        result = stackLaplacian(images);
    } else if (method == "-m2") {
        result = stackWithMask(images);
    } else if (method == "-m3") {
        result = stackWithSharpnessMask(images);
    } else if (method == "-m4") {
        result = stackWithRectangles(images);
    } else if (method == "-m5") {
        result = stackWithFloatingMasks(images);
    } else if (method == "-m6") {
        result = stackWithFloatingMasks2(images);
    } else if (method == "-m7") {
        result = stackWithLaplacianPyramid(images);
    } else if (method == "-m8") {
        result = stackWithLaplacianPyramidShadow(images);
    } else if (method == "-m9") {
        result = stackByAverage(images);
    } else {
        std::cerr << "Unknown method: " << method << std::endl;
        return cv::Mat();  // Zwróć pusty obraz w przypadku błędu
    }

    return result;
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
    std::string outputFileName = "Stack" + method + "." + commonPrefix + ".png"; // Użyj metody i wspólnego prefiksu
    std::filesystem::path outputPath = wipDir / outputFileName;

    cv::imwrite(outputPath.string(), result);
    std::cout << "Stacked image saved as " << outputPath.string() << std::endl;
}


int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./stack -m1|-m2 -aX directory1 [directory2 ...]" << std::endl;
        return -1;
    }

    std::string method = argv[1];
    std::string aX = argv[2];  // Przykład: "-aX"

    if (aX.length() != 3 || aX[0] != '-' || aX[1] != 'a') {
        std::cerr << "Invalid format for -aX switch" << std::endl;
        return -1;
    }

    // Przetwarzanie każdego katalogu osobno
    for (int i = 3; i < argc; ++i) {
        std::string dirPath = argv[i];
        std::cout << "Processing directory: " << dirPath << std::endl;

        std::vector<std::string> filenames; // Dodano do przechowywania nazw plików
        // Wczytaj obrazy z podkatalogu "wip" w bieżącym katalogu
        std::vector<cv::Mat> images = loadImagesFromDirectory(dirPath, aX.substr(1), filenames);

        if (images.empty()) {
            std::cerr << "No valid images found in directory: " << dirPath << std::endl;
            continue;  // Przejdź do kolejnego katalogu
        }

        // Wyświetl listę plików wybranych do stakowania
        std::cout << "Images selected for stacking:" << std::endl;
        for (const auto& filepath : filenames) {
            std::cout << " - " << filepath << std::endl;  // Wyświetl nazwę pliku
        }

        // Przetwarzanie obrazów
        cv::Mat result = processImages(images, method);

        if (result.empty()) {
            std::cerr << "Failed to stack images in directory: " << dirPath << std::endl;
            continue;
        }

        // Zapisz wynik
        saveResult(result, filenames, method);  // Przekazano filenames do saveResult
    }

    return 0;
}
