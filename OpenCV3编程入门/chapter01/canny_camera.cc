#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    cv::VideoCapture capture(0);
    cv::Mat edges;
    while(true) {
        cv::Mat frame;
        capture >> frame;
        cv::cvtColor(frame, edges, cv::COLOR_BGR2GRAY);
        cv::blur(edges, edges, cv::Size(7, 7));
        cv::Canny(edges, edges, 0, 30, 3);
        cv::imshow("Camera After Canny", edges);
        if (cv::waitKey(30) >= 0)
            break;
    }
    return 0;
}
