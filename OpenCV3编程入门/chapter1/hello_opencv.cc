#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    cv::Mat img = cv::imread("../1.jpg");
    cv::imshow("［载入的图片］", img);
    cv::waitKey(6000);
    return 0;
}
