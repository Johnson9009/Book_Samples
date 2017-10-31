#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    cv::VideoCapture capture(0);
    while(true) {
        cv::Mat frame;
        capture >> frame;
        cv::imshow("Camera", frame);
        if (cv::waitKey(30) >= 0)
            break;
    }
    return 0;
}
