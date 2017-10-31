#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    cv::VideoCapture capture("../1.mp4");
    while(true) {
        cv::Mat frame;
        capture >> frame;
        if (frame.empty() == true)
            break;
        cv::imshow("Video Player", frame);
        cv::waitKey(30);
    }
    return 0;
}
