#include <cstdio>
#include <string>
#include <opencv4/opencv2/opencv.hpp>

#include "PatchMatch.h"

using namespace std;

void cvMat2ImageMat(cv::Mat &mat, ImageMat<int, 3> &out) {
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            for (int k = 0; k < 3; ++k) {
                out(i, j, k) = mat.at<cv::Vec3b>(i, j)[k];
            }
        }
    }
}

void imageMat2cvMat(ImageMat<int, 3> &mat, cv::Mat &out) {
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            for (int k = 0; k < 3; ++k) {
                out.at<cv::Vec3b>(i, j)[k] = mat(i, j, k);
            }
        }
    }
}

int main(int argc, char **argv) {
    string nameA = argv[1];
    string nameB = argv[2];

    cv::Mat imageA = cv::imread(nameA, 1);
    cv::Mat imageB = cv::imread(nameB, 1);
    cv::Mat imageC;
    imageC = imageA;

    ImageMat<int, 3> dataA(imageA.rows, imageA.cols);
    ImageMat<int, 3> dataB(imageA.rows, imageA.cols);
    cvMat2ImageMat(imageA, dataA);
    cvMat2ImageMat(imageB, dataB);


    PatchMatch pm(&dataA, &dataB, 5, 1);
    pm.NNS();

    ImageMat<int, 3> dataC(imageA.rows, imageA.cols);
    pm.reconstruction(&dataC);
    imageMat2cvMat(dataC, imageC);
    cv::imwrite(argv[3], imageC);
}
