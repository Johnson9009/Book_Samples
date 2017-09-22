#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char *argv[]) {
    cv::Mat srcImage = cv::imread("../1.jpg");
    cv::imshow("Original", srcImage);
    cv::Mat dstImage;
    cv::blur(srcImage, dstImage, cv::Size(7, 7));
    cv::imshow("Blured", dstImage);
    cv::waitKey();
    return 0;
}
