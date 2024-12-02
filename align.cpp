/*
 * compile: g++  -std=c++17 -o align align.cpp `pkg-config --cflags --libs opencv4`
 * Ubuntu 22.04.4 LTS: libopencv-features2d-dev:amd64 version 4.5.4+dfsg-9ubuntu4 works 4.2.0 does not
 */

/*
 * Based or inspired on:
 * https://github.com/bznick98/Focus_Stacking       2               (m1, m4)
 * https://github.com/cmcguinness/focusstack        2               (m1, m4)
 * https://github.com/PetteriAimonen/focus-stack    1               (m10)
 * https://github.com/abadams/ImageStack            4               (m12-15)
 * https://github.com/maitek/image_stacking         2               (m1, m10)
 * 
 * best stack: bznick98
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>
#include <filesystem>
#include <string>

using namespace cv;
using namespace std;

// Stałe do przetwarzania Canny
const int CANNY_LOW = 50; // Próg dolny Canny
const int CANNY_HIGH = 100; // Próg górny Canny
const int MAX_OFFSET = 512; // Maksymalne przesunięcie w pikselach
const int SMALL_OFFSET = 10;

// Method 1: ORB-based feature matching alignment
//as in https://github.com/maitek/image_stacking/blob/master/auto_stack.py (stackImagesKeypointMatching)
//as in https://github.com/cmcguinness/focusstack/blob/master/FocusStack.py (align_images)
//as in https://github.com/bznick98/Focus_Stacking/blob/master/src/utils.py (align_images)
void alignImageORB(const cv::Mat& baseImage, const cv::Mat& srcImage, cv::Mat& result, cv::Point2f& shift, float& scale) {
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
    
    // Oblicz skalowanie jako średnią z wartości diagonalnych macierzy homografii
    scale = std::sqrt(H.at<double>(0, 0) * H.at<double>(0, 0) + H.at<double>(1, 1) * H.at<double>(1, 1)) / std::sqrt(2);

    // Calculate the translation shift from homography matrix
    shift.x = H.at<double>(0, 2);
    shift.y = H.at<double>(1, 2);
}


// Method 2: Phase correlation alignment
void alignImagePhaseCorrelation(const cv::Mat& baseImage, const cv::Mat& srcImage, cv::Mat& result, cv::Point2f& shift, float& scale) {
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
    
    scale = 1.0f;
}

// Method 3: FFT-based alignment
void alignImageFFT(const cv::Mat& baseImage, const cv::Mat& srcImage, cv::Mat& result, cv::Point2f& shift, float& scale) {
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
    
    // Skalowanie ustalone na 1.0, ponieważ nie dokonujemy zmian rozmiaru w tej metodzie
    scale = 1.0f;
}


// Method 4: SIFT-based alignment
// as in https://github.com/cmcguinness/focusstack/blob/master/FocusStack.py (align_images)
// as in https://github.com/bznick98/Focus_Stacking/blob/master/src/utils.py (align_images)
void alignImageSIFT(const cv::Mat& baseImage, const cv::Mat& imgToAlign, cv::Mat& result, cv::Point2f& shift, float& scale) {

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
    
    // Oblicz skalowanie z macierzy homografii
    scale = std::sqrt(H.at<double>(0, 0) * H.at<double>(0, 0) + H.at<double>(1, 1) * H.at<double>(1, 1)) / std::sqrt(2);

    // Calculate shift (only translation component)
    shift = Point2f(H.at<double>(0, 2), H.at<double>(1, 2));
}

// Method 5: Template matching-based alignment (simple implementation)
void alignImageByTemplateMatching(const cv::Mat& baseImg, const cv::Mat& imgToAlign, cv::Mat& result, cv::Point2f& shift, float& scale) {
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
    
    // Skalowanie ustalone na 1.0
    scale = 1.0f;
}

// Method 6: Funkcja do wyrównywania obrazu na podstawie Canny
void alignImageByCanny(const cv::Mat& baseImg, const cv::Mat& imgToAlign, cv::Mat& result, cv::Point2f& shift, float& scale, int margin = MAX_OFFSET, cv::Point2f startPoint = cv::Point2f(0, 0)) {
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
    
    // Skalowanie ustalone na 1.0
    scale = 1.0f;
}

//Method 7: method 3 and 6 combined
void alignImageByFFTAndCanny(const cv::Mat& baseImg, const cv::Mat& imgToAlign, cv::Mat& result, cv::Point2f& shift, float& scale) {
    alignImageFFT(baseImg, imgToAlign, result, shift, scale);
    cv::Point2f startPointByFFT = shift;
    
    if(std::abs(startPointByFFT.x) > MAX_OFFSET || std::abs(startPointByFFT.y) > MAX_OFFSET){
        //fallback to standard values
        alignImageByCanny(baseImg, imgToAlign, result, shift, scale);
    }else{
        alignImageByCanny(baseImg, imgToAlign, result, shift, scale, SMALL_OFFSET, startPointByFFT); // Canny overlap-based alignment
    }
    
    // Skalowanie ustalone na 1.0
    scale = 1.0f;
}

// Method 8: SIFT-based translation-only alignment
void alignImageSIFTTranslationOnly(const cv::Mat& baseImage, const cv::Mat& imgToAlign, cv::Mat& result, cv::Point2f& shift, float& scale) {
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
    
    // Skala w translacji-only wynosi 1
    scale = 1.0f;
    
}

// Metoda 9: Dopasowanie obrazów z użyciem SIFT bez filtrowania dopasowań i tylko przesunięcie
void alignImageSIFTNoFilter(const cv::Mat& baseImage, const cv::Mat& imgToAlign, cv::Mat& result, cv::Point2f& shift, float& scale) {
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
    
    // Skala w translacji-only wynosi 1
    scale = 1.0f;
}

//Metoda 10:
//https://github.com/maitek/image_stacking/blob/master/auto_stack.py (stackImagesECC)
//https://github.com/PetteriAimonen/focus-stack
void alignImagesECC(const cv::Mat& baseImage, const cv::Mat& imgToAlign, cv::Mat& result, cv::Point2f& shift, float& scale) {
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
    
    // Obliczanie skali jako długości wektora (skalowanie w kierunku x)
    scale = std::sqrt(M.at<float>(0, 0) * M.at<float>(0, 0) + M.at<float>(1, 0) * M.at<float>(1, 0));

}

//Metoda 11: 
//as in https://github.com/bznick98/Focus_Stacking
// rezygnujemy z metody piramid, ponieważ wymaga ona wykorzystania matchTemplate, homography
// lub innego dopasowywania, a wszystkie są wykorzystywane w innych metodach
// możliwe, że metoda piramid by przyspieszyła proces, ale nie jest to kluczowe w tym zastosowaniu.
void alignImagesByPyramid(const cv::Mat& baseImage, const cv::Mat& imgToAlign, cv::Mat& result, cv::Point2f& shift, float& scale, int N = 3) {
    // Zmiana formatu obrazów na 8-bitowe
    cv::Mat baseGray, imgGray;
    cv::cvtColor(baseImage, baseGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgToAlign, imgGray, cv::COLOR_BGR2GRAY);

    // Tworzenie piramidy Gaussa dla obrazu bazowego
    std::vector<cv::Mat> baseGaussianPyramid;
    baseGaussianPyramid.push_back(baseGray);
    for (int i = 0; i < N; ++i) {
        cv::Mat down;
        cv::pyrDown(baseGaussianPyramid.back(), down);
        baseGaussianPyramid.push_back(down);
    }

    // Tworzenie piramidy Gaussa dla obrazu do wyrównania
    std::vector<cv::Mat> imgGaussianPyramid;
    imgGaussianPyramid.push_back(imgGray);
    for (int i = 0; i < N; ++i) {
        cv::Mat down;
        cv::pyrDown(imgGaussianPyramid.back(), down);
        imgGaussianPyramid.push_back(down);
    }

    // Inicjalizacja obrazu wyrównanego i przesunięcia
    cv::Mat alignedImage = imgGaussianPyramid[N].clone();
    shift = cv::Point2f(0.0f, 0.0f);

    // Wyrównywanie od najwyższego poziomu piramidy do najniższego
    for (int level = N; level >= 0; --level) {
        cv::Mat baseLevel = baseGaussianPyramid[level];
        cv::Mat imgLevel = imgGaussianPyramid[level];

        // Wykrywanie punktów kluczowych i opisów
        cv::Ptr<cv::ORB> detector = cv::ORB::create();
        std::vector<cv::KeyPoint> keypointsBase, keypointsImg;
        cv::Mat descriptorsBase, descriptorsImg;
        detector->detectAndCompute(baseLevel, cv::noArray(), keypointsBase, descriptorsBase);
        detector->detectAndCompute(imgLevel, cv::noArray(), keypointsImg, descriptorsImg);

        // Dopasowywanie punktów kluczowych
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<cv::DMatch> matches;
        matcher.match(descriptorsBase, descriptorsImg, matches);

        // Filtracja dopasowań
        std::vector<cv::DMatch> goodMatches;
        double max_dist = 0; 
        double min_dist = 100;

        for (const auto& match : matches) {
            double dist = match.distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }

        for (const auto& match : matches) {
            if (match.distance <= std::max(2 * min_dist, 30.0)) {
                goodMatches.push_back(match);
            }
        }

        // Oblicz homografię tylko jeśli są wystarczające dopasowania
        if (goodMatches.size() >= 4) {
            std::vector<cv::Point2f> pointsBase, pointsImg;
            for (const auto& match : goodMatches) {
                pointsBase.push_back(keypointsBase[match.queryIdx].pt);
                pointsImg.push_back(keypointsImg[match.trainIdx].pt);
            }

            cv::Mat homography = cv::findHomography(pointsImg, pointsBase, cv::RANSAC);
            if (!homography.empty()) {
                // Oblicz przesunięcie
                shift.x += homography.at<double>(0, 2); // x przesunięcie
                shift.y += homography.at<double>(1, 2); // y przesunięcie

                // Zastosuj homografię
                cv::warpPerspective(alignedImage, alignedImage, homography, baseLevel.size());
                
                scale = std::sqrt(homography.at<double>(0, 0) * homography.at<double>(0, 0) +
                                  homography.at<double>(1, 0) * homography.at<double>(1, 0));
            
            }
        }

        // Zwiększ obraz do następnego poziomu
        if (level > 0) {
            cv::Mat up;
            cv::pyrUp(alignedImage, up, baseGaussianPyramid[level - 1].size());
            alignedImage = up;
        }
    }

    // Ustalenie ostatecznego obrazu
    alignedImage.copyTo(result);

    // Skoryguj przesunięcie na podstawie rozmiaru obrazu
    shift.x /= (1 << N); // Podziel przez 2^N
    shift.y /= (1 << N); // Podziel przez 2^N
}

//Metoda 12: https://github.com/abadams/ImageStack -> Alignment.cpp (Digest::align)
//I'ts almost the same as Rigid, so we wont use Rigid.
void alignImageAffine(const cv::Mat& baseImage, const cv::Mat& srcImage, cv::Mat& result, cv::Point2f& shift, float& scale) {
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

    // Compute affine transformation matrix
    cv::Mat affineMatrix = cv::estimateAffinePartial2D(pointsSrc, pointsBase);

    // Warp the source image
    cv::warpAffine(srcImage, result, affineMatrix, baseImage.size());

    // Calculate the translation shift from affine matrix
    shift.x = affineMatrix.at<double>(0, 2);
    shift.y = affineMatrix.at<double>(1, 2);
    
    scale = std::sqrt(std::pow(affineMatrix.at<double>(0, 0), 2) + std::pow(affineMatrix.at<double>(1, 0), 2));

}

//Metoda 13: https://github.com/abadams/ImageStack -> Alignment.cpp (Digest::align)
void alignImageTranslation(const cv::Mat& baseImage, const cv::Mat& srcImage, cv::Mat& result, cv::Point2f& shift, float& scale) {
    // Konwersja obrazów do odcieni szarości
    cv::Mat baseGray, srcGray;
    cv::cvtColor(baseImage, baseGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(srcImage, srcGray, cv::COLOR_BGR2GRAY);

    // Wykrywanie cech i obliczanie deskryptorów ORB
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypointsBase, keypointsSrc;
    cv::Mat descriptorsBase, descriptorsSrc;
    orb->detectAndCompute(baseGray, cv::noArray(), keypointsBase, descriptorsBase);
    orb->detectAndCompute(srcGray, cv::noArray(), keypointsSrc, descriptorsSrc);

    // Dopasowywanie deskryptorów przy użyciu BFMatcher
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsBase, descriptorsSrc, matches);
    std::sort(matches.begin(), matches.end());

    // Zachowanie tylko najlepszych dopasowań
    const int numGoodMatches = matches.size() * 0.1;
    matches.erase(matches.begin() + numGoodMatches, matches.end());

    // Ekstrakcja lokalizacji punktów z dopasowań
    std::vector<cv::Point2f> pointsBase, pointsSrc;
    for (const auto& match : matches) {
        pointsBase.push_back(keypointsBase[match.queryIdx].pt);
        pointsSrc.push_back(keypointsSrc[match.trainIdx].pt);
    }

    // Obliczanie średnich punktów dla translacji
    if (pointsBase.empty() || pointsSrc.empty()) {
        std::cerr << "Brak wystarczających punktów do obliczenia translacji!" << std::endl;
        return;
    }

    // Obliczanie środka
    cv::Point2f centerBase(0, 0), centerSrc(0, 0);
    for (const auto& pt : pointsBase) {
        centerBase += pt;
    }
    for (const auto& pt : pointsSrc) {
        centerSrc += pt;
    }
    centerBase.x /= pointsBase.size();
    centerBase.y /= pointsBase.size();
    centerSrc.x /= pointsSrc.size();
    centerSrc.y /= pointsSrc.size();

    // Obliczanie przesunięcia
    shift = centerBase - centerSrc;

    // Ustalanie macierzy translacji
    cv::Mat translationMatrix = (cv::Mat_<double>(2, 3) << 1, 0, shift.x, 0, 1, shift.y);

    // Odkształcanie źródłowego obrazu
    cv::warpAffine(srcImage, result, translationMatrix, baseImage.size());
    
    
    scale = 1.0f;
}

//Metoda 14: https://github.com/abadams/ImageStack -> Alignment.cpp (Digest::align)
void alignImageSimilarity(const cv::Mat& baseImage, const cv::Mat& srcImage, cv::Mat& result, cv::Point2f& shift, float& scale) {
    // Konwersja obrazów do odcieni szarości
    cv::Mat baseGray, srcGray;
    cv::cvtColor(baseImage, baseGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(srcImage, srcGray, cv::COLOR_BGR2GRAY);

    // Wykrywanie cech i obliczanie deskryptorów ORB
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypointsBase, keypointsSrc;
    cv::Mat descriptorsBase, descriptorsSrc;
    orb->detectAndCompute(baseGray, cv::noArray(), keypointsBase, descriptorsBase);
    orb->detectAndCompute(srcGray, cv::noArray(), keypointsSrc, descriptorsSrc);

    // Dopasowywanie deskryptorów przy użyciu BFMatcher
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsBase, descriptorsSrc, matches);
    std::sort(matches.begin(), matches.end());

    // Zachowanie tylko najlepszych dopasowań
    const int numGoodMatches = matches.size() * 0.1;
    matches.erase(matches.begin() + numGoodMatches, matches.end());

    // Ekstrakcja lokalizacji punktów z dopasowań
    std::vector<cv::Point2f> pointsBase, pointsSrc;
    for (const auto& match : matches) {
        pointsBase.push_back(keypointsBase[match.queryIdx].pt);
        pointsSrc.push_back(keypointsSrc[match.trainIdx].pt);
    }

    // Obliczanie macierzy podobieństwa
    cv::Mat similarityMatrix = cv::estimateAffine2D(pointsSrc, pointsBase);

    // Sprawdzenie, czy macierz została poprawnie obliczona
    if (similarityMatrix.empty()) {
        std::cerr << "Nie udało się obliczyć macierzy podobieństwa!" << std::endl;
        return;
    }

    // Odkształcanie źródłowego obrazu
    cv::warpAffine(srcImage, result, similarityMatrix, baseImage.size());

    // Obliczanie przesunięcia z macierzy podobieństwa
    shift.x = similarityMatrix.at<double>(0, 2);
    shift.y = similarityMatrix.at<double>(1, 2);
    
    scale = std::sqrt(std::pow(similarityMatrix.at<double>(0, 0), 2) + std::pow(similarityMatrix.at<double>(1, 0), 2));
}

//Metoda 15: https://github.com/abadams/ImageStack -> Alignment.cpp (Digest::align)
void alignImageTransform(const cv::Mat& baseImage, const cv::Mat& srcImage, cv::Mat& result, cv::Point2f& shift, float& scale) {
    // Konwersja obrazów do odcieni szarości
    cv::Mat baseGray, srcGray;
    cv::cvtColor(baseImage, baseGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(srcImage, srcGray, cv::COLOR_BGR2GRAY);

    // Wykrywanie cech i obliczanie deskryptorów ORB
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypointsBase, keypointsSrc;
    cv::Mat descriptorsBase, descriptorsSrc;
    orb->detectAndCompute(baseGray, cv::noArray(), keypointsBase, descriptorsBase);
    orb->detectAndCompute(srcGray, cv::noArray(), keypointsSrc, descriptorsSrc);

    // Dopasowywanie deskryptorów przy użyciu BFMatcher
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsBase, descriptorsSrc, matches);
    std::sort(matches.begin(), matches.end());

    // Zachowanie tylko najlepszych dopasowań
    const int numGoodMatches = matches.size() * 0.1;
    matches.erase(matches.begin() + numGoodMatches, matches.end());

    // Ekstrakcja lokalizacji punktów z dopasowań
    std::vector<cv::Point2f> pointsBase, pointsSrc;
    for (const auto& match : matches) {
        pointsBase.push_back(keypointsBase[match.queryIdx].pt);
        pointsSrc.push_back(keypointsSrc[match.trainIdx].pt);
    }

    // Sprawdzanie, czy mamy wystarczającą liczbę punktów do obliczenia macierzy perspektywicznej
    if (pointsSrc.size() < 4 || pointsBase.size() < 4) {
        std::cerr << "Niewystarczająca liczba punktów do obliczenia transformacji!" << std::endl;
        return;
    }

    // Wybieramy dokładnie cztery najlepsze dopasowania punktów
    pointsBase.resize(4);
    pointsSrc.resize(4);

    // Obliczanie macierzy perspektywicznej
    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(pointsSrc, pointsBase);

    // Sprawdzenie, czy macierz została poprawnie obliczona
    if (perspectiveMatrix.empty()) {
        std::cerr << "Nie udało się obliczyć macierzy perspektywicznej!" << std::endl;
        return;
    }

    // Odkształcanie źródłowego obrazu
    cv::warpPerspective(srcImage, result, perspectiveMatrix, baseImage.size());

    // Określanie skali na podstawie przekształconych rogów obrazu
    std::vector<cv::Point2f> srcCorners = { {0, 0}, {static_cast<float>(srcImage.cols), 0},
                                            {static_cast<float>(srcImage.cols), static_cast<float>(srcImage.rows)}, {0, static_cast<float>(srcImage.rows)} };
    std::vector<cv::Point2f> transformedCorners(4);
    cv::perspectiveTransform(srcCorners, transformedCorners, perspectiveMatrix);

    // Przesunięcie na podstawie macierzy perspektywicznej (zakładając, że chodzi o translację)
    shift.x = perspectiveMatrix.at<double>(0, 2);
    shift.y = perspectiveMatrix.at<double>(1, 2);
    
    
     // Obliczanie przeskalowania wzdłuż osi X i Y na podstawie odległości między przekształconymi rogami
    float scaleX = std::sqrt(std::pow(transformedCorners[1].x - transformedCorners[0].x, 2) +
                             std::pow(transformedCorners[1].y - transformedCorners[0].y, 2)) / srcImage.cols;
    float scaleY = std::sqrt(std::pow(transformedCorners[2].x - transformedCorners[1].x, 2) +
                             std::pow(transformedCorners[2].y - transformedCorners[1].y, 2)) / srcImage.rows;

    // Przybliżona skala to średnia z obu osi
    scale = (scaleX + scaleY) / 2.0f;
}

// Metoda 16: metoda 12 bez rotacji
void alignImageTranslationScale(const cv::Mat& baseImage, const cv::Mat& srcImage, cv::Mat& result, cv::Point2f& shift, float& scale) {
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
    for (const auto& match : matches) {
        pointsBase.push_back(keypointsBase[match.queryIdx].pt);
        pointsSrc.push_back(keypointsSrc[match.trainIdx].pt);
    }

    // Estimate translation and scaling factors
    cv::Point2f sumShift(0, 0);
    float sumScale = 0.0f;

    for (size_t i = 0; i < pointsBase.size(); ++i) {
        sumShift += pointsBase[i] - pointsSrc[i];
        sumScale += cv::norm(pointsBase[i]) / cv::norm(pointsSrc[i]);
    }

    // Compute average shift and scaling
    shift = sumShift * (1.0f / pointsBase.size());
    scale = sumScale / pointsBase.size();

    // Construct transformation matrix
    cv::Mat translationScaleMatrix = (cv::Mat_<double>(2, 3) << scale, 0, shift.x, 0, scale, shift.y);

    // Warp the source image with translation and scaling only
    cv::warpAffine(srcImage, result, translationScaleMatrix, baseImage.size());
}

// Funkcja normalizująca jasność i kontrast każdego obrazu
void normalizeImage(cv::Mat& image) {
    //for (auto& image : images) {
        cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);

        if (image.channels() == 3) {
            cv::Mat imgYCrCb;
            cv::cvtColor(image, imgYCrCb, cv::COLOR_BGR2YCrCb);
            std::vector<cv::Mat> channels;
            cv::split(imgYCrCb, channels);
            cv::equalizeHist(channels[0], channels[0]);
            cv::merge(channels, imgYCrCb);
            cv::cvtColor(imgYCrCb, image, cv::COLOR_YCrCb2BGR);
        } else if (image.channels() == 1) {
            cv::equalizeHist(image, image);
        }
    //}
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
    
    //normalizeImage(baseImage);

    // Przetwarzanie każdego obrazu
    for (int i = 3; i < argc; ++i) {
        std::string srcFilePath = argv[i];
        cv::Mat srcImage = cv::imread(srcFilePath, cv::IMREAD_COLOR);
        if (srcImage.empty()) {
            std::cerr << "Error loading image: " << srcFilePath << std::endl;
            continue;
        }
        
        //normalizeImage(srcImage);

        // Sprawdzanie, czy oba pliki są takie same
        if (std::filesystem::equivalent(argv[2], argv[i])) {
            // Kopiowanie pliku bez analizy
            std::string outputFilename = std::filesystem::path(srcFilePath).parent_path().string() + "/Aligned" + alignmentMethod + "." + std::filesystem::path(srcFilePath).filename().string();
//             if (copyFile(srcFilePath, outputFilename)) {
//                 //std::cout << "Files are identical, copied to " << outputFilename << std::endl;
//             }
            
            cv::imwrite(outputFilename, srcImage);
            continue;
        }

        cv::Mat result;
        cv::Point2f shift;
        float scale = 1.0f;
        if (alignmentMethod == "-a1") {
            alignImageORB(baseImage, srcImage, result, shift, scale); // ORB feature-based alignment
        } else if (alignmentMethod == "-a2") {
            alignImagePhaseCorrelation(baseImage, srcImage, result, shift, scale); // Phase correlation alignment
        } else if (alignmentMethod == "-a3") {
            alignImageFFT(baseImage, srcImage, result, shift, scale); // FFT-based alignment
        } else if (alignmentMethod == "-a4") {
            alignImageSIFT(baseImage, srcImage, result, shift, scale); // SIFT-based alignment
        } else if (alignmentMethod == "-a5") {
            alignImageByTemplateMatching(baseImage, srcImage, result, shift, scale); // Template matching alignment
        } else if (alignmentMethod == "-a6") {
            alignImageByCanny(baseImage, srcImage, result, shift, scale); // Canny overlap-based alignment
        } else if (alignmentMethod == "-a7") {
            alignImageByFFTAndCanny(baseImage, srcImage, result, shift, scale);
        } else if (alignmentMethod == "-a8") {
            alignImageSIFTTranslationOnly(baseImage, srcImage, result, shift, scale); 
        } else if (alignmentMethod == "-a9") {
            alignImageSIFTNoFilter(baseImage, srcImage, result, shift, scale);
        } else if (alignmentMethod == "-a10") {
            alignImagesECC(baseImage, srcImage, result, shift, scale); 
        } else if (alignmentMethod == "-a11") {
            alignImagesByPyramid(baseImage, srcImage, result, shift, scale); 
        } else if (alignmentMethod == "-a12") {
            alignImageAffine(baseImage, srcImage, result, shift, scale); 
        } else if (alignmentMethod == "-a13") {
            alignImageTranslation(baseImage, srcImage, result, shift, scale); 
        } else if (alignmentMethod == "-a14") {
            alignImageSimilarity(baseImage, srcImage, result, shift, scale); 
        } else if (alignmentMethod == "-a15") {
            alignImageTransform(baseImage, srcImage, result, shift, scale); 
        } else if (alignmentMethod == "-a16") {
            alignImageTranslationScale(baseImage, srcImage, result, shift, scale);
        } else {
            std::cerr << "Unknown alignment method!" << std::endl;
            return -1;
        }
        
        if(abs(shift.x) > MAX_OFFSET || abs(shift.y) > MAX_OFFSET){
            std::cout << "Shift is too big. ";
        }

        // Zapisanie obrazu w tym samym katalogu co oryginalny plik
        std::string outputFilename = std::filesystem::path(srcFilePath).parent_path().string() + "/Aligned" + alignmentMethod + "." + std::filesystem::path(srcFilePath).filename().string();
        cv::imwrite(outputFilename, result);
        //std::cout << "Aligned image saved as " << outputFilename << std::endl;
        std::cout << "Shift: " << shift << ", Scale: " << scale << std::endl;
    }

    return 0;
}
