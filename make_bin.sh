#sudo apt-get install libexiv2-dev libopencv-features2d-dev:amd64 libboost-filesystem-dev libopencv-imgcodecs-dev libopencv-highgui-dev libopencv-imgproc-dev

echo "Compiling cut"
g++ -std=c++17 -o cut cut.cpp `pkg-config --cflags --libs opencv4` -lexiv2

echo "Compiling align"
g++ -std=c++17 -o align align.cpp `pkg-config --cflags --libs opencv4`

echo "Compiling stack"
g++ -std=c++17 -o stack stack.cpp `pkg-config --cflags --libs opencv4` -pthread

echo "Compiling findEdges"
g++ -std=c++17 -o findEdges findEdges.cpp `pkg-config --cflags --libs opencv4` -lboost_filesystem -lboost_system

echo "Compiling measureWear"
g++ -std=c++17 -o measureWear measureWear.cpp `pkg-config --cflags --libs opencv4`
