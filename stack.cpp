#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <atomic>
#include <algorithm>
#include <fstream>
#include <thread>


#define RECT_SIZE 50 // Rozmiar pola do analizy ostrości
#define DILATION_ITER 5 // Ilość iteracji rozszerzenia maski

#define NUM_LEVELS 5  // Liczba poziomów piramidy (met. 7)

cv::Mat precomputedKernel;
cv::Mat deviationResult;// = cv::Mat::zeros(rows, cols, CV_64F);  // Używamy CV_64F dla odchylenia
cv::Mat entropyResult;// = cv::Mat::zeros(rows, cols, CV_64F);  // Używamy CV_64F dla odchylenia
    

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

void calculateDeviationAndEntropy(const cv::Mat& image, cv::Mat& deviations, cv::Mat& entropies, int kernel_size) {
    // Obliczanie ilości piksli, które będą dodane jako padding
    int pad_amount = (kernel_size - 1) / 2;

    // Sprawdzamy, czy obraz jest pusty
    if (image.empty()) {
        std::cerr << "Error: Image is empty.\n";
        return;  // Zwracamy, jeśli obraz jest pusty
    }

    // Tworzenie obrazu z paddingiem (dodanie obramowania)
    cv::Mat paddedImage;
    cv::copyMakeBorder(image, paddedImage, pad_amount, pad_amount, pad_amount, pad_amount, cv::BORDER_REFLECT101);

    // Inicjalizowanie macierzy do przechowywania wyników odchyleń i entropii
    //deviations.create(image.rows, image.cols, CV_64F); // Inicjalizujemy macierz dla odchyleń
    //entropies.create(image.rows, image.cols, CV_64F);   // Inicjalizujemy macierz dla entropii

    // Przechodzimy po każdym pikselu w oryginalnym obrazie
    for (int row = pad_amount; row < paddedImage.rows - pad_amount; ++row) {
        for (int col = pad_amount; col < paddedImage.cols - pad_amount; ++col) {
            // Wybieramy region w paddingowanej macierzy, który odpowiada bieżącemu pikselowi
            cv::Rect region(col - pad_amount, row - pad_amount, kernel_size, kernel_size);
            cv::Mat area = paddedImage(region);  // Wyciągamy region z paddingowanego obrazu

            // Obliczanie odchylenia standardowego
            cv::Scalar mean, stddev;
            cv::meanStdDev(area, mean, stddev);
            //std::cout << "Deviations size: " << deviations.size() << ". Trying to write at " << row - pad_amount << "x" << col - pad_amount << std::endl;
            deviations.at<double>(row - pad_amount, col - pad_amount) = stddev[0];
            //std::cout << "ok1" << std::endl;

            // Obliczanie histogramu dla regionu w celu obliczenia entropii
            std::vector<int> hist(256, 0);  // Histogram dla 256 poziomów szarości
            for (int i = 0; i < area.rows; ++i) {
                for (int j = 0; j < area.cols; ++j) {
                    int pixelValue = static_cast<int>(area.at<uchar>(i, j));
                    hist[pixelValue]++;
                }
            }

            // Obliczanie entropii dla danego regionu
            double entropy_value = 0.0;
            int areaSize = area.rows * area.cols;  // Liczba pikseli w regionie
            for (int value : hist) {
                if (value > 0) {
                    double p = static_cast<double>(value) / areaSize;  // Prawdopodobieństwo danego poziomu szarości
                    entropy_value -= p * std::log2(p);  // Obliczamy entropię
                }
            }

            // Zapisujemy wynik entropii w odpowiedniej komórce macierzy entropii
            entropies.at<double>(row - pad_amount, col - pad_amount) = entropy_value;
        }
    }
}

void saveMatToTxt(const cv::Mat& mat, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Błąd przy otwieraniu pliku do zapisu: " << filename << std::endl;
        return;
    }

    // Iteracja przez wszystkie elementy macierzy
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            // Zapisz element do pliku, rozdzielając spacje między wartościami
            file << mat.at<float>(i, j);
            if (j < mat.cols - 1) file << " ";
        }
        file << std::endl; // Nowa linia po każdym wierszu
    }

    file.close();
    std::cout << "Zapisano macierz do pliku " << filename << std::endl;
}

// Funkcja do tworzenia fuzji na poziomie Laplace’a na podstawie lokalnego odchylenia i entropii
cv::Mat computeLaplacianLayerFusion(const std::vector<cv::Mat>& laplacianLevelImages, int kernel_size) {
    int rows = laplacianLevelImages[0].rows;
    int cols = laplacianLevelImages[0].cols;
    std::cout << "Making fusedLayer var... " << std::endl;

    cv::Mat fusedLayer = cv::Mat::zeros(rows, cols, laplacianLevelImages[0].type());

    std::cout << "Fusion: Image is " << rows << "x" << cols << std::endl;

    std::cout << "ent " << std::endl;
    std::vector<cv::Mat> entropies;
    std::cout << "dev " << std::endl;
    std::vector<cv::Mat> deviations;


    // Inicjalizujemy macierze dla odchylenia i entropii z rozmiarami rows i cols
    std::cout << "mk " << std::endl;
    //deviationResult.release();
    //deviationResult = cv::Mat::zeros(rows, cols, CV_64F);  // Używamy CV_64F dla odchylenia
    //deviationResult = cv::Mat(rows, cols, CV_64F);
    //cv::Mat deviationResult;// = cv::Mat::zeros(rows, cols, CV_64F, cv::Scalar(0));  // Używamy CV_64F dla odchylenia
    //deviationResult.create(rows, cols, float);
    std::cout << "mk " << std::endl;
    //entropyResult.release();
    //entropyResult = cv::Mat::zeros(rows, cols, CV_64F);  // Używamy CV_64F dla entropii
    //entropyResult = cv::Mat(rows, cols, CV_64F, cv::Scalar(1));
    std::cout << "ok" << std::endl;

    // Obliczanie lokalnego odchylenia i entropii dla każdego obrazu na poziomie
    int imNo = 0;
    for (const auto& img : laplacianLevelImages) {
        std::cout << "Dev & Ent." << std::endl;

        //czyszczenie tych macierzy
        std::cout << rows << "x" << cols << std::endl;
        deviationResult = cv::Mat::zeros(rows, cols, CV_64F);  // Używamy CV_64F dla odchylenia
        std::cout << "ok" << std::endl;
        entropyResult = cv::Mat::zeros(rows, cols, CV_64F);  // Używamy CV_64F dla entropii
        // Obliczamy odchylenie i entropię w jednym przebiegu
        calculateDeviationAndEntropy(img, deviationResult, entropyResult, kernel_size);
        std::cout << "ok" << std::endl;

        // Dodajemy wyniki do wektorów
        deviations.push_back(deviationResult.clone());
        std::cout << "ok" << std::endl;
        entropies.push_back(entropyResult.clone());

        std::cout << "ok" << std::endl;
        // Uzyskujemy rozmiary obrazów: oryginalnego obrazu, odchylenia i entropii
        cv::Size imageSize = img.size();
        cv::Size deviationSize = deviationResult.size();
        cv::Size entropySize = entropyResult.size();

        // Wyświetlamy rozmiary w jednym wierszu
        std::cout << "Image " << imNo << ": "
                << "Original Image Size = " << imageSize.width << "x" << imageSize.height << ", "
                << "Deviation Size = " << deviationSize.width << "x" << deviationSize.height << ", "
                << "Entropy Size = " << entropySize.width << "x" << entropySize.height
                << std::endl;

        // Dodajemy obliczone odchylenie i entropię do wyników
//         deviations.push_back(deviationResult);
//         entropies.push_back(entropyResult);

        imNo++;  // Zwiększamy numer obrazu
    }

    // Iterujemy po wszystkich wierszach obrazu
    int max_rows = rows;  // Użyj rozmiaru pierwszego obrazu jako odniesienia
    int max_cols = cols;

    for (int i = 0; i < max_rows; ++i) {
        std::cout << i << " of " << max_rows << " rows." << std::endl;

        for (int j = 0; j < max_cols; ++j) {
            int D_max_idx = 0, E_max_idx = 0, D_min_idx = 0, E_min_idx = 0;
            std::cout << "Ne";
            double D_max = deviations[0].at<double>(i, j), E_max = entropies[0].at<double>(i, j);
            std::cout <<"xt. ";
            double D_min = D_max, E_min = E_max;

            for (int k = 1; k < laplacianLevelImages.size(); ++k) {
                if (deviations[k].rows <= i || deviations[k].cols <= j ||
                    entropies[k].rows <= i || entropies[k].cols <= j ||
                    laplacianLevelImages[k].rows <= i || laplacianLevelImages[k].cols <= j) {
                    std::cerr << "Error: Index out of bounds at (i=" << i << ", j=" << j << ", k=" << k << ").\n";
                    return fusedLayer;  // Wyjście z funkcji, aby zapobiec dalszym błędom
                }

                double D_val = deviations[k].at<double>(i, j);
                double E_val = entropies[k].at<double>(i, j);

                if (D_val > D_max) { D_max = D_val; D_max_idx = k; }
                if (E_val > E_max) { E_max = E_val; E_max_idx = k; }
                if (D_val < D_min) { D_min = D_val; D_min_idx = k; }
                if (E_val < E_min) { E_min = E_val; E_min_idx = k; }

            }

            std::cout << "Dev: " << deviations[0].rows << "x" << deviations[0].cols << ", Entr: " << entropies[0].rows << "x" << entropies[0].cols
                    << ", Image is " << rows << "x" << cols << ", calculating pixel: " << i << "x" << j
                    << ": D_max_idx: " << D_max_idx << ", E_max_idx: " << E_max_idx
                    << ", D_min_idx: " << D_min_idx << ", E_min_idx: " << E_min_idx;

            if (fusedLayer.rows <= i || fusedLayer.cols <= j) {
                std::cerr << "Error: `fusedLayer` index out of bounds at (i=" << i << ", j=" << j << ").\n";
                return fusedLayer;
            }

            std::cout << " => option ";

            if (D_max_idx == E_max_idx) {
                fusedLayer.at<double>(i, j) = laplacianLevelImages[D_max_idx].at<double>(i, j);
                std::cout << "1." << std::endl;
            } else if (D_min_idx == E_min_idx) {
                fusedLayer.at<double>(i, j) = laplacianLevelImages[D_min_idx].at<double>(i, j);
                std::cout << "2." << std::endl;
            } else {
                double sum = 0.0;
                for (int k = 0; k < laplacianLevelImages.size(); ++k) {
                    sum += laplacianLevelImages[k].at<double>(i, j);
                }
                fusedLayer.at<double>(i, j) = sum / laplacianLevelImages.size();
                std::cout << "3." << std::endl;
            }
        }
    }

//     deviations.clear() ;
//     entropies.clear();
//     deviationResult.deallocate();
    std::cout << "Saving fusedLayer. Size: " << fusedLayer.size() << std::endl;
    
    saveMatToTxt(fusedLayer, "fus.txt");
    
    if (fusedLayer.type() != CV_8U) {
        fusedLayer.convertTo(fusedLayer, CV_8U, 255.0);  // Skaluje wartości z 0-1 na 0-255
    }
    
    std::cout << "ok1" << std::endl;

    cv::imwrite("fus.png", fusedLayer.clone());
    
    std::cout << "ok2" << std::endl;

    
    return fusedLayer;
}


// // Funkcja, która oblicza nową warstwę Laplace’a dla danego poziomu (średnia)
// cv::Mat computeLaplacianLayer(const std::vector<cv::Mat>& laplacianLevelImages) {
//     // Zakładamy, że wszystkie obrazy mają ten sam rozmiar na danym poziomie
//     cv::Mat result = cv::Mat::zeros(laplacianLevelImages[0].size(), laplacianLevelImages[0].type());
//
//     // Sumujemy wartości dla każdego obrazu na danym poziomie
//     for (const auto& laplacianImage : laplacianLevelImages) {
//         result += laplacianImage;
//     }
//
//     // Średnia wartości, aby zachować odpowiednią intensywność
//     result /= static_cast<double>(laplacianLevelImages.size());
//     return result;
// }

// Funkcja do składania obrazów z piramidą Laplace’a
cv::Mat stackWithLaplacianPyramidMono(const std::vector<cv::Mat>& monoImages) {
    int factor = std::pow(2, NUM_LEVELS);

    // Rozszerzamy każdy obraz do najbliższego rozmiaru podzielnego przez 2^{NUM_LEVELS}
    std::vector<cv::Mat> extendedImages;
    std::vector<cv::Size> originalSizes;
    for (const auto& img : monoImages) {
        originalSizes.push_back(img.size()); // Zapisujemy oryginalny rozmiar
        int newRows = std::ceil(img.rows / static_cast<double>(factor)) * factor;
        int newCols = std::ceil(img.cols / static_cast<double>(factor)) * factor;
        cv::Mat extended;
        cv::copyMakeBorder(img, extended, 0, newRows - img.rows, 0, newCols - img.cols, cv::BORDER_REFLECT);
        extendedImages.push_back(extended);
    }

    // Piramidy Gaussa dla każdego obrazu
    std::vector<std::vector<cv::Mat>> gaussianPyramids;
    for (const auto& img : extendedImages) {
        std::vector<cv::Mat> gaussianPyramid;
        cv::Mat current = img;
        gaussianPyramid.push_back(current);

        // Tworzymy piramidę Gaussa o zdefiniowanej liczbie poziomów
        for (int i = 0; i < NUM_LEVELS; ++i) {
            cv::Mat down;
            cv::pyrDown(current, down);
            gaussianPyramid.push_back(down);
            current = down;
        }
        gaussianPyramids.push_back(gaussianPyramid);
    }

    // Piramidy Laplace’a dla każdego obrazu
    std::vector<std::vector<cv::Mat>> laplacianPyramids;
    for (const auto& gaussianPyramid : gaussianPyramids) {
        std::vector<cv::Mat> laplacianPyramid;
        for (size_t i = 0; i < gaussianPyramid.size() - 1; ++i) {
            cv::Mat up;
            cv::pyrUp(gaussianPyramid[i + 1], up, gaussianPyramid[i].size());
            cv::Mat laplacian = gaussianPyramid[i] - up;
            laplacianPyramid.push_back(laplacian);
        }
        laplacianPyramid.push_back(gaussianPyramid.back());
        laplacianPyramids.push_back(laplacianPyramid);
    }

    // Tworzymy nową piramidę Laplace’a przez średnią wartości na każdym poziomie
    std::vector<cv::Mat> fusedLaplacianPyramid;
    int numLevels = laplacianPyramids[0].size();
    for (int level = 0; level < numLevels; ++level) {
        std::cout << "Fusion level: " << level << " of " << numLevels << std::endl;
        std::vector<cv::Mat> laplacianLevelImages;
        for (const auto& laplacianPyramid : laplacianPyramids) {
            laplacianLevelImages.push_back(laplacianPyramid[level]);
        }
        fusedLaplacianPyramid.push_back(computeLaplacianLayerFusion(laplacianLevelImages, 5).clone());
    }

    // Rekonstrukcja obrazu z piramidy Laplace’a
    cv::Mat result = fusedLaplacianPyramid.back();
    for (int level = fusedLaplacianPyramid.size() - 2; level >= 0; --level) {
        cv::Mat up;
        cv::pyrUp(result, up, fusedLaplacianPyramid[level].size());
        result = up + fusedLaplacianPyramid[level];
    }

    // Przycinanie obrazu do pierwotnego rozmiaru
    cv::Size originalSize = originalSizes[0];  // Zakładamy, że wszystkie obrazy mają ten sam pierwotny rozmiar
    result = result(cv::Rect(0, 0, originalSize.width, originalSize.height));

    return result;
}


// Funkcja stackWithLaplacianPyramid, która przyjmuje wektor obrazów wielokanałowych (kolorowych)
cv::Mat stackWithLaplacianPyramid(const std::vector<cv::Mat>& images) {
    // Sprawdzamy, czy lista obrazów nie jest pusta
    if (images.empty()) {
        throw std::invalid_argument("The input image list is empty.");
    }
    
//     int cols = images[0].cols;
//     int rows = images[0].rows;
//     entropyResult = cv::Mat(rows, cols, CV_64F);
//     deviationResult = cv::Mat(rows, cols, CV_64F);

    // Wektor, który będzie przechowywał jednokanałowe wersje obrazów
    std::vector<cv::Mat> monoImages;

    // Konwertujemy każdy obraz na skalę szarości (jednokanałowy)
    for (const auto& img : images) {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        monoImages.push_back(gray);
    }

    // Wywołujemy stackWithLaplacianPyramidMono i zwracamy wynik
    return stackWithLaplacianPyramidMono(monoImages);
}

///////////////////////////
// cv::Mat stackWithLaplacianPyramid2(const std::vector<cv::Mat>& images) {
//     int numImages = images.size();
//
//     if (numImages == 0) {
//         throw std::invalid_argument("Input image vector is empty.");
//     }
//
//     // Tworzymy wektor do przechowywania piramid Laplace'a
//     std::vector<std::vector<cv::Mat>> laplacianPyramids(numImages);
//
//     // Dla każdego obrazu tworzymy piramidę Laplace'a
//     for (int i = 0; i < numImages; ++i) {
//         cv::Mat img = images[i];
//
//         // Tworzymy piramidę Gaussa
//         std::vector<cv::Mat> gaussianPyramid;
//         gaussianPyramid.push_back(img);
//         cv::Mat currentImg = img;
//
//         for (int j = 0; j < 5; ++j) { // Możesz zmienić liczbę poziomów piramidy
//             cv::Mat downsampled;
//             cv::pyrDown(currentImg, downsampled);  // Próbkowanie w dół
//             gaussianPyramid.push_back(downsampled);
//             currentImg = downsampled;
//         }
//
//         // Tworzymy piramidę Laplace'a dla tego obrazu
//         std::vector<cv::Mat> laplacianPyramid;
//         for (int j = 0; j < gaussianPyramid.size() - 1; ++j) {
//             cv::Mat expanded;
//             cv::pyrUp(gaussianPyramid[j + 1], expanded, gaussianPyramid[j].size());  // Rozciąganie
//             cv::Mat laplacian = gaussianPyramid[j] - expanded;  // Laplacian
//             laplacianPyramid.push_back(laplacian);
//         }
//
//         // Ostatni poziom piramidy Gaussa jest samodzielny
//         laplacianPyramid.push_back(gaussianPyramid.back());
//
//         // Dodajemy piramidę Laplace'a tego obrazu do wektora piramid
//         laplacianPyramids[i] = laplacianPyramid;
//     }
//
//     // Fuzja: dla każdego poziomu piramidy, wybieramy najlepszy obraz (najostrzejszy) z kanałów
//     std::vector<cv::Mat> fusedPyramid;
//     int numLevels = laplacianPyramids[0].size();  // Zakładamy, że wszystkie obrazy mają tę samą liczbę poziomów
//
//     for (int level = 0; level < numLevels; ++level) {
//         cv::Mat fusedLevel = cv::Mat::zeros(laplacianPyramids[0][level].size(), laplacianPyramids[0][level].type());
//
//         // Łączymy obrazy na danym poziomie piramidy
//         for (int i = 0; i < numImages; ++i) {
//             fusedLevel += laplacianPyramids[i][level];
//         }
//
//         // Normalizujemy wynik na poziomie (aby uzyskać poprawne jasności)
//         fusedLevel /= numImages;
//         fusedPyramid.push_back(fusedLevel);
//     }
//
//     // Rekonstrukcja obrazu: budujemy obraz na podstawie piramidy Laplace'a dla każdego poziomu
//     cv::Mat result = fusedPyramid.back();
//     for (int level = fusedPyramid.size() - 2; level >= 0; --level) {
//         cv::Mat expanded;
//         cv::pyrUp(result, expanded, fusedPyramid[level].size());
//         result = expanded + fusedPyramid[level];
//     }
//
//     return result;
// }
//
// // Nowa funkcja stackWithLaplacianPyramid
// cv::Mat stackWithLaplacianPyramid(const std::vector<cv::Mat>& images) {
//     if (images.empty()) {
//         throw std::invalid_argument("Input image vector is empty.");
//     }
//
//     std::vector<cv::Mat> blueChannelImages, greenChannelImages, redChannelImages;
//
//     // Rozdzielamy kanały dla każdego obrazu
//     for (const auto& img : images) {
//         std::vector<cv::Mat> channels(3);
//         cv::split(img, channels); // Rozdzielamy na B, G, R
//         blueChannelImages.push_back(channels[0]);
//         greenChannelImages.push_back(channels[1]);
//         redChannelImages.push_back(channels[2]);
//     }
//
//     // Stosujemy stackWithLaplacianPyramid2 dla każdego kanału
//     cv::Mat blueResult = stackWithLaplacianPyramid2(blueChannelImages);
//     cv::Mat greenResult = stackWithLaplacianPyramid2(greenChannelImages);
//     cv::Mat redResult = stackWithLaplacianPyramid2(redChannelImages);
//
//     // Łączymy kanały w jeden obraz kolorowy
//     std::vector<cv::Mat> mergedChannels = {blueResult, greenResult, redResult};
//     cv::Mat result;
//     cv::merge(mergedChannels, result);
//
//     return result;
// }


//////////////////// koniec met 7.////////////////



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

// //niedoskonałe
// //void normalizeImagesUsingMedian(std::vector<cv::Mat>& images) {
// void normalizeImages(std::vector<cv::Mat>& images) {
//     // Compute global median for each channel
//     cv::Scalar globalMedian = cv::Scalar(0, 0, 0);
//
//     // Collect pixel values for each channel
//     std::vector<std::vector<float>> channelValues(3);
//
//     for (const auto& image : images) {
//         cv::Mat img;
//         if (image.channels() == 3) {
//             img = image;
//         } else {
//             cv::cvtColor(image, img, cv::COLOR_GRAY2BGR); // Ensure 3-channel for uniformity
//         }
//
//         // Collect pixel values for each channel
//         std::vector<cv::Mat> channels(3);
//         cv::split(img, channels);
//         for (int c = 0; c < 3; ++c) {
//             channels[c].reshape(1, 1).convertTo(channels[c], CV_32F);
//             std::vector<float> values(channels[c].begin<float>(), channels[c].end<float>());
//             channelValues[c].insert(channelValues[c].end(), values.begin(), values.end());
//         }
//     }
//
//     // Compute global median
//     for (int c = 0; c < 3; ++c) {
//         auto& values = channelValues[c];
//         std::sort(values.begin(), values.end());
//         size_t mid = values.size() / 2;
//         globalMedian[c] = (values.size() % 2 == 0) ?
//                            (values[mid - 1] + values[mid]) / 2.0 :
//                            values[mid];
//     }
//
//     // Normalize each image based on global median
//     for (auto& image : images) {
//         cv::Mat img;
//         if (image.channels() == 3) {
//             img = image;
//         } else {
//             cv::cvtColor(image, img, cv::COLOR_GRAY2BGR);
//         }
//
//         // Normalize each channel
//         std::vector<cv::Mat> channels(3);
//         cv::split(img, channels);
//         for (int c = 0; c < 3; ++c) {
//             channels[c].convertTo(channels[c], CV_32F);
//             channels[c] -= globalMedian[c];  // Shift by global median
//             cv::normalize(channels[c], channels[c], 0, 255, cv::NORM_MINMAX); // Normalize to [0, 255]
//         }
//
//         // Merge channels back and clip values to valid range
//         cv::Mat normalized;
//         cv::merge(channels, normalized);
//         normalized.convertTo(image, CV_8U);
//     }
// }


// //szare
// //void normalizeImagesUsingMedian(std::vector<cv::Mat>& images) {
// void normalizeImages(std::vector<cv::Mat>& images) {
//     // Compute global median and median absolute deviation (MAD) for each channel
//     cv::Scalar globalMedian = cv::Scalar(0, 0, 0);
//     cv::Scalar globalMAD = cv::Scalar(0, 0, 0);
//
//     // Collect pixel values for each channel
//     std::vector<std::vector<float>> channelValues(3);
//
//     for (const auto& image : images) {
//         cv::Mat img;
//         if (image.channels() == 3) {
//             img = image;
//         } else {
//             cv::cvtColor(image, img, cv::COLOR_GRAY2BGR); // Ensure 3-channel for uniformity
//         }
//
//         // Collect pixel values for each channel
//         std::vector<cv::Mat> channels(3);
//         cv::split(img, channels);
//         for (int c = 0; c < 3; ++c) {
//             channels[c].reshape(1, 1).copyTo(channelValues[c]);
//         }
//     }
//
//     // Compute global median and MAD
//     for (int c = 0; c < 3; ++c) {
//         auto& values = channelValues[c];
//         std::sort(values.begin(), values.end());
//         size_t mid = values.size() / 2;
//         globalMedian[c] = (values.size() % 2 == 0) ?
//                            (values[mid - 1] + values[mid]) / 2.0 :
//                            values[mid];
//
//         // Compute MAD (Median Absolute Deviation)
//         std::vector<float> deviations;
//         deviations.reserve(values.size());
//         for (float value : values) {
//             deviations.push_back(std::abs(value - globalMedian[c]));
//         }
//
//         std::sort(deviations.begin(), deviations.end());
//         globalMAD[c] = (deviations.size() % 2 == 0) ?
//                        (deviations[mid - 1] + deviations[mid]) / 2.0 :
//                        deviations[mid];
//     }
//
//     // Normalize each image to the global median and MAD
//     for (auto& image : images) {
//         cv::Mat img;
//         if (image.channels() == 3) {
//             img = image;
//         } else {
//             cv::cvtColor(image, img, cv::COLOR_GRAY2BGR);
//         }
//
//         // Normalize each channel
//         std::vector<cv::Mat> channels(3);
//         cv::split(img, channels);
//         for (int c = 0; c < 3; ++c) {
//             channels[c].convertTo(channels[c], CV_32F);
//             channels[c] -= globalMedian[c];
//             if (globalMAD[c] > 0.001) { // Avoid division by zero
//                 channels[c] /= globalMAD[c];
//             }
//             channels[c] += globalMedian[c];
//         }
//
//         // Merge channels back and clip values to valid range
//         cv::Mat normalized;
//         cv::merge(channels, normalized);
//         normalized.convertTo(image, CV_8U);
//     }
// }

//czerwone
// //void normalizeImagesUsingMedian(std::vector<cv::Mat>& images) {
// void normalizeImages(std::vector<cv::Mat>& images) {
//     std::cout << "Normalizing images using median." << std::endl;
//
//     // Compute global median and median absolute deviation (MAD) for each channel
//     cv::Scalar globalMedian = cv::Scalar(0, 0, 0);
//     cv::Scalar globalMAD = cv::Scalar(0, 0, 0);
//     int totalPixels = 0;
//
//     // Collect pixel values for each channel
//     std::vector<std::vector<float>> channelValues(3);
//
//     for (const auto& image : images) {
//         cv::Mat img;
//         if (image.channels() == 3) {
//             img = image;
//         } else {
//             cv::cvtColor(image, img, cv::COLOR_GRAY2BGR); // Ensure 3-channel for uniformity
//         }
//
//         // Collect pixel values for each channel
//         std::vector<cv::Mat> channels(3);
//         cv::split(img, channels);
//         for (int c = 0; c < 3; ++c) {
//             channelValues[c].insert(channelValues[c].end(),
//                                     channels[c].begin<float>(), channels[c].end<float>());
//         }
//     }
//
//     // Compute global median and MAD
//     for (int c = 0; c < 3; ++c) {
//         std::vector<float>& values = channelValues[c];
//
//         // Sort values to compute the median
//         std::sort(values.begin(), values.end());
//         size_t mid = values.size() / 2;
//         globalMedian[c] = (values.size() % 2 == 0) ?
//                            (values[mid - 1] + values[mid]) / 2.0 :
//                            values[mid];
//
//         // Compute MAD (Median Absolute Deviation)
//         std::vector<float> deviations;
//         deviations.reserve(values.size());
//         for (float value : values) {
//             deviations.push_back(std::abs(value - globalMedian[c]));
//         }
//
//         std::sort(deviations.begin(), deviations.end());
//         globalMAD[c] = (deviations.size() % 2 == 0) ?
//                        (deviations[mid - 1] + deviations[mid]) / 2.0 :
//                        deviations[mid];
//     }
//
//     // Normalize each image to the global median and MAD
//     for (auto& image : images) {
//         cv::Mat img;
//         if (image.channels() == 3) {
//             img = image;
//         } else {
//             cv::cvtColor(image, img, cv::COLOR_GRAY2BGR);
//         }
//
//         // Normalize each channel
//         std::vector<cv::Mat> channels(3);
//         cv::split(img, channels);
//         for (int c = 0; c < 3; ++c) {
//             channels[c].convertTo(channels[c], CV_32F);
//             channels[c] -= globalMedian[c];
//             if (globalMAD[c] > 0.001) { // Avoid division by zero
//                 channels[c] /= globalMAD[c];
//             }
//             channels[c] *= 128.0; // Scale to standard 0-255 range (optional)
//             channels[c] += 128.0;
//         }
//
//         // Merge channels back and clip values to valid range
//         cv::Mat normalized;
//         cv::merge(channels, normalized);
//         normalized.convertTo(image, CV_8U, 1.0, 0.0); // Clip to valid range
//     }
//
//     std::cout << "Done." << std::endl;
// }


//void normalizeImagesToGlobalMean(std::vector<cv::Mat>& images) {
// void normalizeImages(std::vector<cv::Mat>& images) {
//     // Compute global mean and standard deviation for each channel
//     cv::Scalar globalMean = cv::Scalar(0, 0, 0);
//     cv::Scalar globalStdDev = cv::Scalar(0, 0, 0);
//     int totalPixels = 0;
//
//     for (const auto& image : images) {
//         cv::Mat img;
//         if (image.channels() == 3) {
//             img = image;
//         } else {
//             cv::cvtColor(image, img, cv::COLOR_GRAY2BGR); // Ensure 3-channel for uniformity
//         }
//         cv::Scalar mean, stddev;
//         cv::meanStdDev(img, mean, stddev);
//
//         globalMean += mean * static_cast<double>(img.total());
//         globalStdDev += stddev.mul(stddev) * static_cast<double>(img.total()); // Accumulate variance
//         totalPixels += img.total();
//     }
//
//     globalMean /= totalPixels; // Average mean
//     globalStdDev = cv::Scalar(std::sqrt(globalStdDev[0] / totalPixels),
//                               std::sqrt(globalStdDev[1] / totalPixels),
//                               std::sqrt(globalStdDev[2] / totalPixels)); // Std deviation
//
//     // Normalize each image to the global mean and standard deviation
//     for (auto& image : images) {
//         cv::Mat img;
//         if (image.channels() == 3) {
//             img = image;
//         } else {
//             cv::cvtColor(image, img, cv::COLOR_GRAY2BGR);
//         }
//
//         cv::Mat normalized;
//         cv::Scalar mean, stddev;
//         cv::meanStdDev(img, mean, stddev);
//
//         // Normalize each channel
//         for (int i = 0; i < 3; ++i) {
//             img.convertTo(normalized, CV_32F);
//             normalized -= mean[i];
//             if (stddev[i] > 0.001) { // Avoid division by zero
//                 normalized /= stddev[i];
//             }
//             normalized *= globalStdDev[i];
//             normalized += globalMean[i];
//         }
//
//         // Clip values to valid range and convert back to the original format
//         cv::Mat clipped;
//         cv::normalize(normalized, clipped, 0, 255, cv::NORM_MINMAX, CV_8U);
//         image = clipped;
//     }
// }


//dużo szybsze niż void normalizeImagesUsingMedian, a równie niedoskonałe ;)
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

// // Funkcja normalizująca jasność i kontrast każdego obrazu
// void normalizeImages(std::vector<cv::Mat>& images) {
//     for (auto& image : images) {
//         cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);
//
//         if (image.channels() == 3) {
//             cv::Mat imgYCrCb;
//             cv::cvtColor(image, imgYCrCb, cv::COLOR_BGR2YCrCb);
//             std::vector<cv::Mat> channels;
//             cv::split(imgYCrCb, channels);
//             cv::equalizeHist(channels[0], channels[0]);
//             cv::merge(channels, imgYCrCb);
//             cv::cvtColor(imgYCrCb, image, cv::COLOR_YCrCb2BGR);
//         } else if (image.channels() == 1) {
//             cv::equalizeHist(image, image);
//         }
//     }
// }

cv::Mat computeEntropy(const cv::Mat& image, int kernelSize) {
    cv::Mat entropy = cv::Mat::zeros(image.size(), CV_64F);
    int pad = kernelSize / 2;
    cv::Mat paddedImage;
    cv::copyMakeBorder(image, paddedImage, pad, pad, pad, pad, cv::BORDER_REFLECT101);

    for (int y = pad; y < image.rows + pad; ++y) {
        for (int x = pad; x < image.cols + pad; ++x) {
            cv::Rect roi(x - pad, y - pad, kernelSize, kernelSize);
            cv::Mat area = paddedImage(roi);
            cv::Mat hist;
            int histSize = 256;
            float range[] = {0, 256};
            const float* histRange = {range};
            cv::calcHist(&area, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

            hist /= kernelSize * kernelSize;
            double pixelEntropy = 0.0;
            for (int i = 0; i < histSize; ++i) {
                float p = hist.at<float>(i);
                if (p > 0) pixelEntropy -= p * std::log2(p);
            }
            entropy.at<double>(y - pad, x - pad) = pixelEntropy;
        }
    }
    return entropy;
}

cv::Mat computeDeviation(const cv::Mat& image, int kernelSize) {
    cv::Mat deviation = cv::Mat::zeros(image.size(), CV_64F);
    int pad = kernelSize / 2;
    cv::Mat paddedImage;
    cv::copyMakeBorder(image, paddedImage, pad, pad, pad, pad, cv::BORDER_REFLECT101);

    for (int y = pad; y < image.rows + pad; ++y) {
        for (int x = pad; x < image.cols + pad; ++x) {
            cv::Rect roi(x - pad, y - pad, kernelSize, kernelSize);
            cv::Mat area = paddedImage(roi);
            double avg = cv::mean(area)[0];
            double pixelDeviation = cv::sum((area - avg).mul(area - avg))[0] / area.total();
            deviation.at<double>(y - pad, x - pad) = pixelDeviation;
        }
    }
    return deviation;
}

cv::Mat calculateAverageNeighborhood(const cv::Mat& indexMatrix, int radius = 5) {
    cv::Mat averagedMatrix;
    // Define the kernel size as 2*radius + 1 to cover the neighborhood ± radius
    int kernelSize = 2 * radius + 1;

    // Use OpenCV's blur function to compute the mean over the neighborhood
    cv::blur(indexMatrix, averagedMatrix, cv::Size(kernelSize, kernelSize));

    // Convert the result to integer type if needed, rounding to the nearest integer
    averagedMatrix.convertTo(averagedMatrix, CV_8U);  // Assumes indexMatrix is also 8-bit

    return averagedMatrix;
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

void computeImageMetricsThread(
    const cv::Mat& image,
    cv::Mat& gray,
    cv::Mat& entropy,
    cv::Mat& deviation,
    int kernelSize) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    entropy = computeEntropy(gray, kernelSize);
    deviation = computeDeviation(gray, kernelSize);
}

void computeImageMetricsParallel(
    const std::vector<cv::Mat>& images,
    std::vector<cv::Mat>& grayImages,
    std::vector<cv::Mat>& entropyImages,
    std::vector<cv::Mat>& deviationImages,
    int kernelSize) {

    grayImages.resize(images.size());
    entropyImages.resize(images.size());
    deviationImages.resize(images.size());

    std::vector<std::thread> threads;
    std::atomic<int> activeThreads(0);

    for (size_t i = 0; i < images.size(); ++i) {
        activeThreads.fetch_add(1, std::memory_order_relaxed); // Increment active thread count
        threads.emplace_back([&, i]() {
            computeImageMetricsThread(
                std::cref(images[i]),
                std::ref(grayImages[i]),
                std::ref(entropyImages[i]),
                std::ref(deviationImages[i]),
                kernelSize);

            // Decrement active thread count and display the remaining count
            int remaining = activeThreads.fetch_sub(1, std::memory_order_relaxed) - 1;
            std::cout << "Images left: " << remaining+1 << std::endl << std::flush;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

cv::Mat stackWithDeviationAndEntropy(const std::vector<cv::Mat>& images, int kernelSize = 5) {
    if (images.empty()) {
        std::cerr << "Błąd: Brak obrazów do przetworzenia." << std::endl;
        return cv::Mat();
    }

    std::cout << "Normalizing images." << std::endl;
    std::vector<cv::Mat> normalizedImages = images;
    //its quite important since entropy and deviation depends on it.
    normalizeImages(normalizedImages);
    std::cout << "Done" << std::endl;


    std::vector<cv::Mat> grayImages, entropyImages, deviationImages;

    std::cout << "Calculating deviation and entropy." << std::endl;

    //replacing image calculations one-by-one with parallel computing:
    computeImageMetricsParallel(
        normalizedImages,
        grayImages,
        entropyImages,
        deviationImages,
        kernelSize);

//     int imNo = 1;
//     std::vector<cv::Mat> grayImages, entropyImages, deviationImages;
//     for (const auto& img : normalizedImages) {
//         std::cout << "Calculating deviation and entropy for " << imNo << "/" << images.size() << std::endl;
//         cv::Mat gray, entropy, deviation;
//         cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
//         grayImages.push_back(gray);
//         entropyImages.push_back(computeEntropy(gray, kernelSize));
//         deviationImages.push_back(computeDeviation(gray, kernelSize));
//         imNo++;
//     }

    std::cout << "Done." << std::endl;

    int rows = normalizedImages[0].rows;
    int cols = normalizedImages[0].cols;
    cv::Mat indexMatrix = cv::Mat::zeros(rows, cols, CV_8U);  // Matrix to store 1-based indices
    cv::Mat result = cv::Mat::zeros(rows, cols, normalizedImages[0].type());

    // Step 1: Populate indexMatrix with the selected image indices
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int D_max_idx = 0, E_max_idx = 0, D_min_idx = 0, E_min_idx = 0;
            double D_max = -1, E_max = -1;
            double D_min = std::numeric_limits<double>::max();
            double E_min = std::numeric_limits<double>::max();

            // Loop through each image to find max and min deviation and entropy
            for (int i = 0; i < images.size(); ++i) {
                double d = deviationImages[i].at<double>(y, x);
                double e = entropyImages[i].at<double>(y, x);

                if (d > D_max) { D_max = d; D_max_idx = i; }
                if (e > E_max) { E_max = e; E_max_idx = i; }
                if (d < D_min) { D_min = d; D_min_idx = i; }
                if (e < E_min) { E_min = e; E_min_idx = i; }
            }

            // Determine the index for the selected image based on deviation and entropy criteria
            if (D_max_idx == E_max_idx) {
                indexMatrix.at<uchar>(y, x) = D_max_idx + 1;  // 1-based index
            } else if (D_min_idx == E_min_idx) {
                indexMatrix.at<uchar>(y, x) = D_min_idx + 1;  // 1-based index
            } else {
                indexMatrix.at<uchar>(y, x) = 0;  // No clear winner, set to 0
            }
        }
    }

    indexMatrix = calculateAverageNeighborhood(indexMatrix.clone(), 50);
    fillZeroValues(indexMatrix);

    // Step 2: Scale indexMatrix to fill the 0-255 range for better visualization
    cv::Mat scaledIndexMatrix;
    cv::normalize(indexMatrix, scaledIndexMatrix, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Step 3: Save scaledIndexMatrix as a grayscale image
    cv::imwrite("indexMatrix.png", scaledIndexMatrix);

    // Step 4: Create the result image based on indexMatrix
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int idx = indexMatrix.at<uchar>(y, x);

            if (idx > 0) {
                // Use the pixel from the chosen image (idx - 1 to convert back to 0-based)
                result.at<cv::Vec3b>(y, x) = normalizedImages[idx - 1].at<cv::Vec3b>(y, x);
            } else {
                // No clear winner; average all pixels at this position
                cv::Vec3d averagePixel(0, 0, 0);
                for (const auto& img : normalizedImages) {
                    averagePixel += img.at<cv::Vec3b>(y, x);
                }
                result.at<cv::Vec3b>(y, x) = averagePixel / static_cast<double>(images.size());
                std::cout << "Should be none!" << std::endl;
            }
        }
    }

    return result;
}



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
        std::cout << "Removed." << std::endl;
        //result = stackWithRectangles(images);
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
    } else if (method == "-m10") {
        result = stackWithDeviationAndEntropy(images);
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
        std::cerr << "Usage: ./stack -mX -aX directory1 [directory2 ...]" << std::endl;
        return -1;
    }

    std::string method = argv[1];
    std::string aX = argv[2];  // Przykład: "-aX"

    if (aX.length() < 3 || aX[0] != '-' || aX[1] != 'a') {
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
