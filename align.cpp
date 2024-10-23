/*
 * compile: g++  -std=c++17 -o align align.cpp `pkg-config --cflags --libs opencv4`
 * Ubuntu 22.04.4 LTS: libopencv-features2d-dev:amd64 version 4.5.4+dfsg-9ubuntu4 works 4.2.0 does not
 */

/*
 * Based or inspired on:
 * https://github.com/bznick98/Focus_Stacking       2
 * https://github.com/cmcguinness/focusstack        2
 * https://github.com/PetteriAimonen/focus-stack        
 * https://github.com/abadams/ImageStack            
 * https://github.com/maitek/image_stacking         2
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>
#include <vector>
#include <filesystem>
#include <string>

using namespace cv;
using namespace std;

// Stałe do przetwarzania Canny
const int CANNY_LOW = 50; // Próg dolny Canny
const int CANNY_HIGH = 100; // Próg górny Canny
const int MAX_OFFSET = 64; // Maksymalne przesunięcie w pikselach
const int SMALL_OFFSET = 10;

// Method 1: ORB-based feature matching alignment
//as in https://github.com/maitek/image_stacking/blob/master/auto_stack.py (stackImagesKeypointMatching)
//as in https://github.com/cmcguinness/focusstack/blob/master/FocusStack.py (align_images)
//as in https://github.com/bznick98/Focus_Stacking/blob/master/src/utils.py (align_images)
void alignImageORB(const cv::Mat& baseImage, const cv::Mat& srcImage, cv::Mat& result, cv::Point2f& shift) {
    // Convert images to grayscale
    cv::Mat baseGray, srcGray;
    cv::cvtColor(baseImage, baseGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(srcImage, srcGray, cv::COLOR_BGR2GRAY);

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

    // Keep only the best matches
    const int numGoodMatches = matches.size() * 0.1;
    matches.erase(matches.begin() + numGoodMatches, matches.end());

    // Extract point locations from matches
    std::vector<cv::Point2f> pointsBase, pointsSrc;
    for (size_t i = 0; i < matches.size(); ++i) {
        pointsBase.push_back(keypointsBase[matches[i].queryIdx].pt);
        pointsSrc.push_back(keypointsSrc[matches[i].trainIdx].pt);
    }

    // Compute homography matrix
    cv::Mat H = cv::findHomography(pointsSrc, pointsBase, cv::RANSAC);

    // Warp the source image
    cv::warpPerspective(srcImage, result, H, baseImage.size());

    // Calculate the translation shift from homography matrix
    shift.x = H.at<double>(0, 2);
    shift.y = H.at<double>(1, 2);
}


// Method 2: Phase correlation alignment
void alignImagePhaseCorrelation(const cv::Mat& baseImage, const cv::Mat& srcImage, cv::Mat& result, cv::Point2f& shift) {
    // Convert to grayscale
    cv::Mat baseGray, srcGray;
    cv::cvtColor(baseImage, baseGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(srcImage, srcGray, cv::COLOR_BGR2GRAY);

    // Convert to float type (CV_32F)
    baseGray.convertTo(baseGray, CV_32F);
    srcGray.convertTo(srcGray, CV_32F);

    // Phase correlation to find the shift
    cv::Point2d pcShift = cv::phaseCorrelate(baseGray, srcGray);

    // Store the calculated shift in the provided cv::Point2f
    shift.x = static_cast<float>(pcShift.x);
    shift.y = static_cast<float>(pcShift.y);

    // Apply translation to align the images
    cv::Mat translationMatrix = (cv::Mat_<double>(2, 3) << 1, 0, shift.x, 0, 1, shift.y);
    cv::warpAffine(srcImage, result, translationMatrix, baseImage.size());
}

// Method 3: FFT-based alignment
void alignImageFFT(const cv::Mat& baseImage, const cv::Mat& srcImage, cv::Mat& result, cv::Point2f& shift) {
    // Convert images to grayscale
    cv::Mat baseGray, srcGray;
    cv::cvtColor(baseImage, baseGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(srcImage, srcGray, cv::COLOR_BGR2GRAY);

    // Convert to float type (CV_32F)
    baseGray.convertTo(baseGray, CV_32F);
    srcGray.convertTo(srcGray, CV_32F);

    // Perform FFT (Discrete Fourier Transform)
    cv::Mat baseDFT, srcDFT;
    cv::dft(baseGray, baseDFT, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(srcGray, srcDFT, cv::DFT_COMPLEX_OUTPUT);

    // Multiply spectrums
    cv::Mat crossPower;
    cv::mulSpectrums(baseDFT, srcDFT, crossPower, 0, true);

    // Perform inverse DFT
    cv::Mat invDFT;
    cv::idft(crossPower, invDFT);

    // Split the complex result into real and imaginary parts
    std::vector<cv::Mat> channels;
    cv::split(invDFT, channels);  // Splits the channels into real and imaginary parts

    // Calculate the magnitude of the complex output to find the peak location
    cv::Mat magnitude;
    cv::magnitude(channels[0], channels[1], magnitude);  // Calculate magnitude

    // Find the peak location in the magnitude to get the translation shift
    cv::Point maxLoc;
    cv::minMaxLoc(magnitude, NULL, NULL, NULL, &maxLoc);  // Use magnitude for finding max location

    // Store the shift
    shift.x = static_cast<float>(maxLoc.x);
    shift.y = static_cast<float>(maxLoc.y);

    // Apply translation to align the images
    cv::Mat translationMatrix = (cv::Mat_<double>(2, 3) << 1, 0, shift.x, 0, 1, shift.y);
    cv::warpAffine(srcImage, result, translationMatrix, baseImage.size());
}


// Method 4: SIFT-based alignment
// as in https://github.com/cmcguinness/focusstack/blob/master/FocusStack.py (align_images)
// as in https://github.com/bznick98/Focus_Stacking/blob/master/src/utils.py (align_images)
void alignImageSIFT(const cv::Mat& baseImage, const cv::Mat& imgToAlign, cv::Mat& result, cv::Point2f& shift) {

    Ptr<Feature2D> detector = SIFT::create();
    
    // Convert to grayscale
    Mat baseGray, imgGray;
    cvtColor(baseImage, baseGray, COLOR_BGR2GRAY);
    cvtColor(imgToAlign, imgGray, COLOR_BGR2GRAY);

    // Detect keypoints and descriptors
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(baseGray, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(imgGray, noArray(), keypoints2, descriptors2);

    // Match descriptors
    BFMatcher matcher;
    vector<vector<DMatch>> knnMatches;
    matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);
    vector<DMatch> goodMatches;
    for (const auto& knnMatch : knnMatches) {
        if (knnMatch[0].distance < 0.75 * knnMatch[1].distance) {
            goodMatches.push_back(knnMatch[0]);
        }
    }

    // Extract matched points
    vector<Point2f> pts1, pts2;
    for (size_t i = 0; i < goodMatches.size(); i++) {
        pts1.push_back(keypoints1[goodMatches[i].queryIdx].pt);
        pts2.push_back(keypoints2[goodMatches[i].trainIdx].pt);
    }

    // Find homography and warp
    Mat H = findHomography(pts2, pts1, RANSAC);
    warpPerspective(imgToAlign, result, H, baseImage.size());

    // Calculate shift (only translation component)
    shift = Point2f(H.at<double>(0, 2), H.at<double>(1, 2));
}

// Method 5: Template matching-based alignment (simple implementation)
void alignImageByTemplateMatching(const cv::Mat& baseImg, const cv::Mat& imgToAlign, cv::Mat& result, cv::Point2f& shift) {
    // Convert both images to grayscale
    cv::Mat img1Gray, img2Gray;
    cv::cvtColor(baseImg, img1Gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgToAlign, img2Gray, cv::COLOR_BGR2GRAY);

    // Perform template matching
    cv::Mat matchResult;
    cv::matchTemplate(img1Gray, img2Gray, matchResult, cv::TM_CCOEFF_NORMED);

    // Find the best match location
    cv::Point maxLoc;
    cv::minMaxLoc(matchResult, nullptr, nullptr, nullptr, &maxLoc);

    // Store the shift
    shift.x = static_cast<float>(maxLoc.x);
    shift.y = static_cast<float>(maxLoc.y);

    // Create translation matrix based on the match location
    cv::Mat translationMat = (cv::Mat_<double>(2, 3) << 1, 0, shift.x, 0, 1, shift.y);

    // Apply the transformation (translation) to align the image
    cv::warpAffine(imgToAlign, result, translationMat, baseImg.size());
}

// Method 6: Funkcja do wyrównywania obrazu na podstawie Canny
void alignImageByCanny(const cv::Mat& baseImg, const cv::Mat& imgToAlign, cv::Mat& result, cv::Point2f& shift, int margin = MAX_OFFSET, cv::Point2f startPoint = cv::Point2f(0, 0)) {
    // Przetwarzanie obrazu bazowego na krawędzie (Canny)
    Mat baseGray, baseCanny;
    cvtColor(baseImg, baseGray, COLOR_BGR2GRAY);
    Canny(baseGray, baseCanny, CANNY_LOW, CANNY_HIGH);

    // Przetwarzanie obrazu do wyrównania na krawędzie (Canny)
    Mat currentGray, currentCanny;
    cvtColor(imgToAlign, currentGray, COLOR_BGR2GRAY);
    Canny(currentGray, currentCanny, CANNY_LOW, CANNY_HIGH);

    // Wyszukiwanie najlepszego przesunięcia
    int maxOffset = margin;
    Point bestOffset(0, 0);
    int maxOverlap = 0;
    
    int xInt = static_cast<int>(startPoint.x);
    int yInt = static_cast<int>(startPoint.y);

    // Sprawdzanie przesunięcia w zakresie
    for (int dx = xInt - maxOffset; dx <= xInt + maxOffset; ++dx) {
        for (int dy = yInt - maxOffset; dy <= yInt + maxOffset; ++dy) {
            // Przesuwanie obrazu
            Mat shiftedImage;
            Mat translationMatrix = (Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
            warpAffine(currentCanny, shiftedImage, translationMatrix, currentCanny.size());

            // Liczenie nakładania krawędzi
            int overlap = countNonZero(baseCanny & shiftedImage); // Liczymy nakładające się piksele

            // Jeśli więcej punktów się pokrywa, zapisujemy to przesunięcie
            if (overlap > maxOverlap) {
                maxOverlap = overlap;
                bestOffset = Point(dx, dy);
            }
        }
    }

    // Zapisanie przesunięcia
    shift = Point2f(bestOffset.x, bestOffset.y);

    // Przesunięcie obrazu do wyrównania
    Mat translationMatrix = (Mat_<double>(2, 3) << 1, 0, bestOffset.x, 0, 1, bestOffset.y);
    warpAffine(imgToAlign, result, translationMatrix, imgToAlign.size());
}

//Method 7: method 3 and 6 combined
void alignImageByFFTAndCanny(const cv::Mat& baseImg, const cv::Mat& imgToAlign, cv::Mat& result, cv::Point2f& shift) {
    alignImageFFT(baseImg, imgToAlign, result, shift);
    cv::Point2f startPointByFFT = shift;
    
    if(std::abs(startPointByFFT.x) > MAX_OFFSET || std::abs(startPointByFFT.y) > MAX_OFFSET){
        //fallback to standard values
        alignImageByCanny(baseImg, imgToAlign, result, shift);
    }else{
        alignImageByCanny(baseImg, imgToAlign, result, shift, SMALL_OFFSET, startPointByFFT); // Canny overlap-based alignment
    }
}

// Method 8: SIFT-based translation-only alignment
void alignImageSIFTTranslationOnly(const cv::Mat& baseImage, const cv::Mat& imgToAlign, cv::Mat& result, cv::Point2f& shift) {
    Ptr<Feature2D> detector = SIFT::create();
    
    // Konwersja obrazów do skali szarości
    Mat baseGray, imgGray;
    cvtColor(baseImage, baseGray, COLOR_BGR2GRAY);
    cvtColor(imgToAlign, imgGray, COLOR_BGR2GRAY);

    // Detekcja kluczowych punktów i deskryptorów
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(baseGray, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(imgGray, noArray(), keypoints2, descriptors2);

    // Dopasowywanie deskryptorów
    BFMatcher matcher;
    vector<vector<DMatch>> knnMatches;
    matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);
    vector<DMatch> goodMatches;
    for (const auto& knnMatch : knnMatches) {
        if (knnMatch[0].distance < 0.75 * knnMatch[1].distance) {
            goodMatches.push_back(knnMatch[0]);
        }
    }

    // Wyciąganie punktów dopasowanych
    vector<Point2f> pts1, pts2;
    for (size_t i = 0; i < goodMatches.size(); i++) {
        pts1.push_back(keypoints1[goodMatches[i].queryIdx].pt);
        pts2.push_back(keypoints2[goodMatches[i].trainIdx].pt);
    }

    // Znajdź transformację ograniczoną do przesunięcia (ignorowanie rotacji i skalowania)
    Mat H = findHomography(pts2, pts1, RANSAC);

    // Ekstrakcja komponentów przesunięcia (H[0,2] i H[1,2])
    shift = Point2f(H.at<double>(0, 2), H.at<double>(1, 2));

    // Zbudowanie macierzy translacji (bez skalowania i rotacji)
    Mat translationMat = (Mat_<double>(2, 3) << 1, 0, shift.x, 0, 1, shift.y);

    // Zastosowanie przesunięcia obrazu
    warpAffine(imgToAlign, result, translationMat, baseImage.size());
}

// Metoda 9: Dopasowanie obrazów z użyciem SIFT bez filtrowania dopasowań i tylko przesunięcie
void alignImageSIFTNoFilter(const cv::Mat& baseImage, const cv::Mat& imgToAlign, cv::Mat& result, cv::Point2f& shift) {
    Ptr<Feature2D> detector = SIFT::create();

    // Konwersja obrazów na skale szarości
    Mat baseGray, imgGray;
    cvtColor(baseImage, baseGray, COLOR_BGR2GRAY);
    cvtColor(imgToAlign, imgGray, COLOR_BGR2GRAY);

    // Detekcja kluczowych punktów i deskryptorów
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(baseGray, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(imgGray, noArray(), keypoints2, descriptors2);

    // Dopasowywanie deskryptorów bez filtrowania
    BFMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Wyciągnięcie punktów z dopasowań
    vector<Point2f> pts1, pts2;
    for (size_t i = 0; i < matches.size(); i++) {
        pts1.push_back(keypoints1[matches[i].queryIdx].pt);
        pts2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    // Znajdź transformację przy użyciu metody RANSAC (ale ograniczamy do translacji)
    Mat H = findHomography(pts2, pts1, RANSAC);

    // Ekstrakcja komponentu translacji (H[0,2] i H[1,2] to przesunięcia)
    shift = Point2f(H.at<double>(0, 2), H.at<double>(1, 2));

    // Zbudowanie macierzy translacji (bez skalowania, rotacji czy innych zniekształceń)
    Mat translationMat = (Mat_<double>(2, 3) << 1, 0, shift.x, 0, 1, shift.y);

    // Zastosowanie przesunięcia (bez zniekształceń)
    warpAffine(imgToAlign, result, translationMat, baseImage.size());
}

//Metoda 10: 
//https://github.com/maitek/image_stacking/blob/master/auto_stack.py (stackImagesECC)
void alignImagesECC(const cv::Mat& baseImage, const cv::Mat& imgToAlign, cv::Mat& result, cv::Point2f& shift) {
    // Konwersja obrazów do skali szarości
    cv::Mat baseGray, alignGray;
    cv::cvtColor(baseImage, baseGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgToAlign, alignGray, cv::COLOR_BGR2GRAY);

    // Inicjalizacja macierzy transformacji
    cv::Mat M = cv::Mat::eye(3, 3, CV_32F);
    
    // Wykonanie transformacji ECC
    cv::findTransformECC(alignGray, baseGray, M, cv::MOTION_HOMOGRAPHY);

    // Wydobycie wartości przesunięcia
    shift = cv::Point2f(M.at<float>(0, 2), M.at<float>(1, 2));

    // Wyrównanie obrazu do podstawowego
    cv::warpPerspective(imgToAlign, result, M, baseImage.size());
}

// Funkcja do kopiowania pliku
bool copyFile(const std::string& srcPath, const std::string& destPath) {
    try {
        std::filesystem::copy(srcPath, destPath, std::filesystem::copy_options::overwrite_existing);
        return true;
    } catch (std::filesystem::filesystem_error& e) {
        std::cerr << "Error copying file: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./align [-a{1-7}] base_image.png image1.png image2.png ..." << std::endl;
        return -1;
    }

    // Determine the alignment method based on the switch
    std::string alignmentMethod = argv[1];
    cv::Mat baseImage = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (baseImage.empty()) {
        std::cerr << "Error loading base image: " << argv[2] << std::endl;
        return -1;
    }

    // Przetwarzanie każdego obrazu
    for (int i = 3; i < argc; ++i) {
        std::string srcFilePath = argv[i];
        cv::Mat srcImage = cv::imread(srcFilePath, cv::IMREAD_COLOR);
        if (srcImage.empty()) {
            std::cerr << "Error loading image: " << srcFilePath << std::endl;
            continue;
        }

        // Sprawdzanie, czy oba pliki są takie same
        if (std::filesystem::equivalent(argv[2], argv[i])) {
            // Kopiowanie pliku bez analizy
            std::string outputFilename = std::filesystem::path(srcFilePath).parent_path().string() + "/Aligned" + alignmentMethod + "." + std::filesystem::path(srcFilePath).filename().string();
            if (copyFile(srcFilePath, outputFilename)) {
                //std::cout << "Files are identical, copied to " << outputFilename << std::endl;
            }
            continue;
        }

        cv::Mat result;
        cv::Point2f shift;
        if (alignmentMethod == "-a1") {
            alignImageORB(baseImage, srcImage, result, shift); // ORB feature-based alignment
        } else if (alignmentMethod == "-a2") {
            alignImagePhaseCorrelation(baseImage, srcImage, result, shift); // Phase correlation alignment
        } else if (alignmentMethod == "-a3") {
            alignImageFFT(baseImage, srcImage, result, shift); // FFT-based alignment
        } else if (alignmentMethod == "-a4") {
            alignImageSIFT(baseImage, srcImage, result, shift); // SIFT-based alignment
        } else if (alignmentMethod == "-a5") {
            alignImageByTemplateMatching(baseImage, srcImage, result, shift); // Template matching alignment
        } else if (alignmentMethod == "-a6") {
            alignImageByCanny(baseImage, srcImage, result, shift); // Canny overlap-based alignment
        } else if (alignmentMethod == "-a7") {
            alignImageByFFTAndCanny(baseImage, srcImage, result, shift);
        } else if (alignmentMethod == "-a8") {
            alignImageSIFTTranslationOnly(baseImage, srcImage, result, shift); 
        } else if (alignmentMethod == "-a9") {
            alignImageSIFTNoFilter(baseImage, srcImage, result, shift);
        } else if (alignmentMethod == "-a10") {
            alignImagesECC(baseImage, srcImage, result, shift); 
        } else {
            std::cerr << "Unknown alignment method!" << std::endl;
            return -1;
        }
        
        if(abs(shift.x) > MAX_OFFSET || abs(shift.y) > MAX_OFFSET){
            std::cout << "Shift is too big: " << shift << std::endl;
        }

        // Zapisanie obrazu w tym samym katalogu co oryginalny plik
        std::string outputFilename = std::filesystem::path(srcFilePath).parent_path().string() + "/Aligned" + alignmentMethod + "." + std::filesystem::path(srcFilePath).filename().string();
        cv::imwrite(outputFilename, result);
        //std::cout << "Aligned image saved as " << outputFilename << std::endl;
        std::cout << "Shift: " << shift << std::endl;
    }

    return 0;
}
