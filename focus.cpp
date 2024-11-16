#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>

namespace fs = std::filesystem;

// Struktura na przechowywanie opcji programu
struct Options {
    int canny_low = 50;
    int canny_high = 100;
    int method = 1;
    int sequence_length = -1;
    double canny_proc = -1.0;
    bool copy_files = false;
};

// Funkcja pomocnicza zastępująca starts_with w C++17
bool starts_with(const std::string& str, const std::string& prefix) {
    return str.find(prefix) == 0;
}

// Funkcja do parsowania argumentów
Options parseArguments(int argc, char* argv[], std::string& directory) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (starts_with(arg, "-f")) {
            opts.method = std::stoi(arg.substr(2));
        } else if (arg == "--canny" && i + 1 < argc) {
            std::istringstream iss(argv[++i]);
            std::string value;
            if (std::getline(iss, value, ',')) opts.canny_low = std::stoi(value);
            if (std::getline(iss, value)) opts.canny_high = std::stoi(value);
        } else if (starts_with(arg, "-n") && arg.size() > 2) {
            opts.sequence_length = std::stoi(arg.substr(2));
        } else if (arg == "--canny_proc" && i + 1 < argc) {
            opts.canny_proc = std::stod(argv[++i]);
        } else if (arg == "--copy") {
            opts.copy_files = true;
        } else {
            directory = arg;
        }
    }
    return opts;
}

// Funkcja do wykonania metody Canny na obrazie
cv::Mat processCanny(const cv::Mat& image, int low, int high, int& edge_points) {
    cv::Mat gray, edges;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, low, high);
    edge_points = cv::countNonZero(edges);
    return edges;
}

// Funkcja do obsługi metody -f1
void processMethod1(const fs::path& input_dir, const Options& opts) {
    std::vector<std::pair<fs::path, int>> edge_counts;
    std::vector<cv::Mat> images;

    // Wyszukiwanie plików
    for (const auto& entry : fs::directory_iterator(input_dir / "wip")) {
        if (entry.path().extension() == ".png" && entry.path().filename().string().find("cut.png") != std::string::npos) {
            images.push_back(cv::imread(entry.path().string()));
        }
    }

    // Sortowanie nazw plików
    std::sort(images.begin(), images.end());

    // Przetwarzanie obrazów
    for (const auto& entry : fs::directory_iterator(input_dir / "wip")) {
        if (entry.path().extension() == ".png" && entry.path().filename().string().find("cut.png") != std::string::npos) {
            cv::Mat image = cv::imread(entry.path().string());
            if (!image.empty()) {
                int edge_points = 0;
                cv::Mat edges = processCanny(image, opts.canny_low, opts.canny_high, edge_points);

                // Zapis wynikowego obrazu Canny
                std::string canny_filename = (entry.path().parent_path() / ("Canny." + entry.path().filename().string())).string();
                cv::imwrite(canny_filename, edges);

                // Zapis liczby punktów krawędziowych
                std::ofstream edge_count_file(entry.path().parent_path() / ("Canny." + entry.path().stem().string() + ".cut.txt"));
                edge_count_file << edge_points;
                edge_count_file.close();

                edge_counts.emplace_back(entry.path(), edge_points);
            }
        }
    }

    // Obsługa argumentu -nX (sekwencja z największą sumą punktów krawędziowych)
    if (opts.sequence_length > 0) {
        int max_sum = 0;
        size_t max_start_idx = 0;
        for (size_t i = 0; i + opts.sequence_length <= edge_counts.size(); ++i) {
            int current_sum = 0;
            for (size_t j = 0; j < opts.sequence_length; ++j) {
                current_sum += edge_counts[i + j].second;
            }
            if (current_sum > max_sum) {
                max_sum = current_sum;
                max_start_idx = i;
            }
        }

        // Tworzenie dowiązań symbolicznych lub kopiowanie plików
        for (size_t i = 0; i < opts.sequence_length; ++i) {
            const auto& file_info = edge_counts[max_start_idx + i];
            fs::path target_path = file_info.first.parent_path() / ("Focused." + file_info.first.filename().string());

            if (opts.copy_files) {
                fs::copy(file_info.first, target_path, fs::copy_options::overwrite_existing);
            } else {
                fs::create_symlink(file_info.first, target_path);
            }
        }
    }

    // Obsługa argumentu --canny_proc (udział punktów krawędziowych powyżej progu)
    if (opts.canny_proc > 0) {
        for (const auto& file_info : edge_counts) {
            double edge_ratio = static_cast<double>(file_info.second) / (images[0].cols * images[0].rows) * 100.0;
            if (edge_ratio > opts.canny_proc) {
                fs::path target_path = file_info.first.parent_path() / ("Focused." + file_info.first.filename().string());

                if (opts.copy_files) {
                    fs::copy(file_info.first, target_path, fs::copy_options::overwrite_existing);
                } else {
                    fs::create_symlink(file_info.first, target_path);
                }
            }
        }
    }
}

// Główna funkcja programu
int main(int argc, char* argv[]) {
    try {
        std::string directory;
        Options opts = parseArguments(argc, argv, directory);

        if (directory.empty() || !fs::exists(directory)) {
            std::cerr << "Podano nieprawidłowy katalog!" << std::endl;
            return 1;
        }

        if (opts.method == 1) {
            processMethod1(directory, opts);
        } else {
            std::cerr << "Nieobsługiwana metoda: -f" << opts.method << std::endl;
            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Wystąpił błąd: " << e.what() << std::endl;
        return 1;
    }
}
