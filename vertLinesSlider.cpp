#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Definicje progów
#define CANNY_THRESH1 50
#define CANNY_THRESH2 150
#define MIN_GROUP_SIZE 10

// Funkcja do grupowania pikseli
void groupPixels(const Mat& edges, vector<vector<Point>>& groups) {
    // Tworzymy matrycę, która śledzi, które piksele zostały już odwiedzone
    Mat visited = Mat::zeros(edges.size(), CV_8U);
    int rows = edges.rows;
    int cols = edges.cols;

    // Funkcja BFS do grupowania
    auto bfs = [&](int startX, int startY) {
        vector<Point> group;
        vector<Point> queue = {{startX, startY}};
        visited.at<uchar>(startY, startX) = 1;

        // Przeszukiwanie w szerz (BFS)
        while (!queue.empty()) {
            Point p = queue.back();
            queue.pop_back();
            group.push_back(p);

            // Sprawdzamy sąsiadów
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) continue; // Pomijamy aktualny piksel
                    int nx = p.x + dx;
                    int ny = p.y + dy;
                    if (nx >= 0 && ny >= 0 && nx < cols && ny < rows && !visited.at<uchar>(ny, nx) && edges.at<uchar>(ny, nx) > 0) {
                        visited.at<uchar>(ny, nx) = 1;
                        queue.push_back(Point(nx, ny));
                    }
                }
            }
        }
        return group;
    };

    // Grupowanie pikseli
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (edges.at<uchar>(y, x) > 0 && !visited.at<uchar>(y, x)) {
                // Wykonaj BFS, jeśli piksel nie był jeszcze odwiedzony
                vector<Point> group = bfs(x, y);
                if (group.size() >= MIN_GROUP_SIZE) {
                    groups.push_back(group); // Dodajemy grupę, jeśli jest wystarczająco duża
                }
            }
        }
    }
}

// Funkcja do rysowania grup na obrazie
void drawGroups(Mat& src, const vector<vector<Point>>& groups) {
    for (size_t i = 0; i < groups.size(); i++) {
        Scalar color = (groups[i].size() >= MIN_GROUP_SIZE) ? Scalar(255, 255, 255) : Scalar(0, 0, 255);
        for (const Point& p : groups[i]) {
            src.at<Vec3b>(p) = color; // Ustawiamy odpowiedni kolor dla grupy
        }
    }
}

int main(int argc, char** argv) {
    // Wczytanie obrazu
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    Mat srcImage = imread(argv[1], IMREAD_COLOR);
    if (srcImage.empty()) {
        cout << "Nie udało się wczytać obrazu!" << endl;
        return -1;
    }

    // Przekształcenie obrazu na odcienie szarości
    Mat grayImage;
    cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);

    // Wykonanie detekcji Canny'ego
    Mat edges;
    Canny(grayImage, edges, CANNY_THRESH1, CANNY_THRESH2);

    // Grupowanie pikseli Canny'ego
    vector<vector<Point>> groups;
    groupPixels(edges, groups);

    // Rysowanie grup na obrazie
    Mat resultImage = srcImage.clone();
    drawGroups(resultImage, groups);

    // Wyświetlanie wyników
    imshow("Original Image", srcImage);
    imshow("Edges with Grouping", resultImage);

    waitKey(0);
    return 0;
}
