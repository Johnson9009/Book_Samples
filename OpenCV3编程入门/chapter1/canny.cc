#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    cv::Mat srcImage = cv::imread("../1.jpg");
    cv::imshow("Original", srcImage);
    cv::Mat edge, grayImage;
    cv::cvtColor(srcImage, grayImage, cv::COLOR_BGR2GRAY);
    cv::blur(grayImage, edge, cv::Size(3, 3));
    cv::Canny(edge, edge, 3, 9, 3);
    cv::imshow("Cannied", edge);
    cv::waitKey();
    return 0;
}
