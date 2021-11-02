#include "utility.h"
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>



#define MAXRGB 255
#define MINRGB 0
#define MINNORM 0.0
#define MAXNORM 1.0
#define MINHUE 0.0
#define MAXHUE 360.0

void utility::sobel_opencv(image &src, image &tgt, int kernel_size, bool isColor, ROI ROI_parameters)
{
    //implement with opencv functions 

    // convert loaded image type into cv Mat
    // convert only ROI into Mat type, then apply cv function 
    printf("inside sobel_opencv function\n");
	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

	// tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());
    tgt.copyImage(src);

    // create a cv Mat the size of the ROI

    // cv::Mat_<int> roiMat(Sx, Sy);
    cv::Mat roiMat(Sx, Sy, CV_32S);
    // cv::Mat roiMatFinal(Sx, Sy, CV_8U);

    // int matRow{0}, matCol{0};

    std::cout << roiMat.size() << '\n';
    std::cout << roiMat.type() << "\n";



	// now transform pixels and set tgt image
	for (int row = Y, matRow = 0; row < Y + Sy; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        // cv::Mat_<int> temp(roiMat);
        // do stuff here
        // printf("row, col. matRow, matCol: %d %d %d %d\n", row, col, matRow, matCol);
        roiMat.at<int>(matRow, matCol) =  src.getPixel(row, col);

        // tgt.setPixel(row, col, src.getPixel(row, col));
        // ++matRow;
        // ++matCol;
        // ++matRow;

        // roiMat = temp;
    }

    for (int row = Y, matRow = 0; row < Y + Sy; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {

        cv::Mat_<int> temp(roiMat);
        // printf("row, col. matRow, matCol: %d %d %d %d\n", row, col, matRow, matCol);

        // check if values are the same
        if (src.getPixel(row, col) != temp(matRow, matCol))
            printf("pixels not equal: vals = %d %d\n", src.getPixel(row, col), temp(matRow, matCol));
    }

    //set the kernel size
    int ksize = kernel_size;
    double delta = 0;
    double scale = 1.0;

    roiMat.convertTo(roiMat, CV_32F);

    int ddepth = CV_64F;

    using namespace cv;

    // cv::Mat data(roiMat);

        //![sobel]
    /// Generate grad_x and grad_y
    printf("test1\n");

    cv::Mat grad_x, grad_y, grad;
    cv::Mat abs_grad_x, abs_grad_y;
    printf("test2\n");


    /// Gradient X
    cv::Sobel(roiMat, grad_x, ddepth, 1, 0, ksize, scale, delta, cv::BORDER_DEFAULT);
    // cv::Sobel(roiMat, grad_x, ddepth, 1, 0, kernel_size);

    /// Gradient Y
    cv::Sobel(roiMat, grad_y, ddepth, 0, 1, ksize, scale, delta, cv::BORDER_DEFAULT);
    // cv::Sobel(roiMat, grad_y, ddepth, 0, 1, kernel_size);
    printf("test3\n");

    //![convert]
    // converting back to CV_8U
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    //![convert]

    //![blend]
    /// Total Gradient (approximate)
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    //![blend]

    // ![display]
    printf("test4\n");

    cv::imshow("window", grad);
    char key = (char)waitKey(0);
    //
    // ![display]

    // if(key == 27)
    // {
    // //   return EXIT_SUCCESS;
    // return;
    // }

    printf("test5\n");
    cv::Mat_<int> gradM(grad);

    std::cout << gradM.type() << "\n";

    for (int row = Y, matRow = 0; row < Y + Sy && matRow < grad.rows; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx && matCol < grad.cols; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        tgt.setPixel(row, col, gradM(matRow, matCol));
    }

    // copy Mat results back into tgt image type
}