#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char *argv[]) {
    cv::Mat srcImage = cv::imread("../1.jpg");
    cv::imshow("Original", srcImage);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::Mat dstImage;
    cv::erode(srcImage, dstImage, element);
    cv::imshow("Eroded", dstImage);
    cv::waitKey();
    return 0;
}
