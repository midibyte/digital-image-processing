#include "utility.h"
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <limits.h>



#define MAXRGB 255
#define MINRGB 0
#define MINNORM 0.0
#define MAXNORM 1.0
#define MINHUE 0.0
#define MAXHUE 360.0


/* *************************************************************

    some code below adapted from OpenCV examples on GitHub
    https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/ImgTrans/Sobel_Demo.cpp
    https://docs.opencv.org/3.4.15/da/d5c/tutorial_canny_detector.html



**************************************************************** */

/* **************************************************************************************************************
    Sobel edge detection
    steps:
    generate gradX and gradY
    convert floating point gradient back to [0 255] pixel range
    combine x, y gradients into one image

 ***************************************************************************************************************/

void utility::sobel_opencv(image &src, image &tgt, int T, int angle, int kernel_size, bool isColor, ROI ROI_parameters)
{
    //implemented with opencv functions 

	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

    // copy src into tgt, then overwrite changed pixels from ROI
    tgt.copyImage(src);

    // create a cv Mat the size of the ROI
    // make sure Mat type is the same as input
    cv::Mat_<int> roiMat(Sy, Sx);

    // copy pixels from ROI into cv Mat
	for (int row = Y, matRow = 0; row < Y + Sy && matRow < roiMat.rows; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx && matCol < roiMat.cols; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        roiMat(matRow, matCol) =  src.getPixel(row, col);
    }

    int count = 0;
    int minRow, maxRow, minCol, maxCol;
    minRow = maxRow = minCol = maxCol = -1;
    for (int row = Y, matRow = 0; row < Y + Sy && matRow < roiMat.rows; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx && matCol < roiMat.cols; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {

        // cv::Mat_<int> temp(roiMat);
        // printf("row, col. matRow, matCol: %d %d %d %d\n", row, col, matRow, matCol);

        // check if values are the same
        if (src.getPixel(row, col) != roiMat(matRow, matCol))
        {
            // printf("pixels not equal: vals = %d %d\n", src.getPixel(row, col), roiMat(matRow, matCol));
            ++count;

            if(minRow == -1) minRow = row;
            if(minCol == -1) minCol = col;
            if (maxCol < col) maxCol = col;
            if (maxRow < row) maxRow = row;

        }
    }  

    // printf("sobel pixels not equal count: %d\n", count);
    // printf("bad pix region row col to row col: (%d %d) to (%d %d)\n", minRow, minCol, maxRow, maxCol);

    //set the kernel size
    int ksize = kernel_size;
    double delta = 0;
    double scale = 1.0;

    // int ddepth = CV_64F;
    int ddepth = -1;

    cv::Mat abs_grad_x, abs_grad_y, grad;
    cv::Mat_<double> roiMat_double(roiMat);
    cv::Mat_<double> grad_x, grad_y;
    cv::Mat_<double> angles;
    cv::Mat detected_edges_CV_8U;

    // std::cout << "type of roiMat, gradx, grady: " << roiMat.type() << ", " << grad_x.type() << ", " << grad_y.type() << "\n";

    // Use sobel to get Gx Gy to later use to get angle 
    // generate gradient x and y
    cv::Sobel(roiMat_double, grad_x, ddepth, 1, 0, kernel_size, scale, delta, cv::BORDER_DEFAULT);
    cv::Sobel(roiMat_double, grad_y, ddepth, 0, 1, kernel_size, scale, delta, cv::BORDER_DEFAULT);

    // std::cout << "after convertscaleabs, type of gradx absgradx: " << grad_x.type() << "  " << abs_grad_x.type() << "\n";

    // // convert and scale gradient back to CV_8U
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    /// Total Gradient (approximate)
    // uint8 type output
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    // // display image
    // cv::imshow("window", grad);
    // char key = (char)cv::waitKey(0);

    // opencv thresh
    // only threhsold if T != -1
    if ( T != -1)
    {
        cv::threshold(grad, grad, (double)T, UCHAR_MAX, cv::THRESH_BINARY);
    }

    // get angle in degrees at every pixel - only if needed
    if (angle != -1)
        cv::phase(grad_x, grad_y, angles, true);

    // // display image
    // cv::imshow("window", grad);
    // key = (char)cv::waitKey(0);


    // printf("test5\n");
    cv::Mat_<int> gradM;
    cv::Mat_<int> grad_x_(abs_grad_x);
    cv::Mat_<int> grad_y_(abs_grad_y);


    // if t != -1 output binary image
    // if t and angle != -1 output binary with angle restrction
    // int minRow, maxRow, minCol, maxCol;
    minRow = maxRow = minCol = maxCol = -1;

    // std::cout << "gradM size: " << gradM.size() << "\n";
    // printf("gradM rows, cols: %d %d\n",gradM.rows, gradM.cols );
    // printf("ROI Sx Sy X Y endX endY: %d %d %d %d %d %d\n", Sx, Sy, X, Y, Sx+X, Sy+Y );


    grad.convertTo(gradM, CV_32S);

    // copy results from Mat to image object
    for (int row = Y, matRow = 0; row < Y + Sy && matRow < gradM.rows; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx && matCol < gradM.cols; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        // printf(" row, col matRow matCol inbounds?: %d %d %d %d %d\n", row, col, matRow, matCol, src.isInbounds(row, col));
        if(minRow == -1) minRow = row;
        if(minCol == -1) minCol = col;
        if (maxCol < col) maxCol = col;
        if (maxRow < row) maxRow = row;

        int newPixel = gradM(matRow, matCol);

        // double a = 0.0;
        //calc angle 
        if (angle != -1)
        {
            double a = angles(matRow, matCol);
            // if angle out of range set pixel to black
            if (a < angle - 10 || a > angle + 10)
                newPixel = 0;
        }

        if(!isColor)
            tgt.setPixel(row, col, newPixel);

        else if (isColor)
        {
            tgt.setPixel(row, col, RED, newPixel);
            tgt.setPixel(row, col, GREEN, newPixel);
            tgt.setPixel(row, col, BLUE, newPixel);
        }
    }
    // printf("region set row col to row col: (%d %d) to (%d %d)\n", minRow, minCol, maxRow, maxCol);


}

/* **************************************************************************************************************
    Canny edge detection
    steps:
    

 ***************************************************************************************************************/

void utility::canny_opencv(image &src, image &tgt, int T, int angle, int kernel_size, bool isColor, ROI ROI_parameters)
{
    //implemented with opencv functions 

	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

    // copy src into tgt, then overwrite changed pixels from ROI
    tgt.copyImage(src);

    // setup mat for ROI
    cv::Mat_<int> roiMat(Sy, Sx);

    // copy pixels from ROI into cv Mat
	for (int row = Y, matRow = 0; row < Y + Sy; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        roiMat(matRow, matCol) =  src.getPixel(row, col);
    }
    // printf("bf check\n");

    // test if pixels copied correctly
    for (int row = Y, matRow = 0; row < Y + Sy; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        // cv::Mat_<int> temp(roiMat);
        // check if values are the same
        if (src.getPixel(row, col) != roiMat(matRow, matCol))
            printf("canny pixels not equal: vals = %d %d\n", src.getPixel(row, col), roiMat(matRow, matCol));
    }

    double delta = 0;
    double scale = 1.0;
    // Canny ratio of lowerThreshold to upperThreshold recommended to be 3
    int ratio = 3;
    double T1, T2;
    T1 = (double)T;
    if (T1 < 0) T1 = 0;
    T2 = T1 * (double)ratio;

    // printf("bf roiMat convert\n");


    // convert input to please cv and for more precision during calculation

    // int ddepth = CV_64F;
    int ddepth = -1;

    cv::Mat_<double> roiMat_double(roiMat);
    cv::Mat_<double> grad_x, grad_y, grad;
    cv::Mat_<double> angles;
    cv::Mat detected_edges_CV_8U;
    // printf("test b/f sobel\n");

    // std::cout << "type of roiMat, gradx, grady: " << roiMat.type() << ", " << grad_x.type() << ", " << grad_y.type() << "\n";

    // Use sobel to get Gx Gy to later use to get angle 
    // generate gradient x and y
    cv::Sobel(roiMat_double, grad_x, ddepth, 1, 0, kernel_size, scale, delta, cv::BORDER_DEFAULT);
    cv::Sobel(roiMat_double, grad_y, ddepth, 0, 1, kernel_size, scale, delta, cv::BORDER_DEFAULT);

    // output from Canny should be in 8-bit format
    // set output type for Canny
    cv::Mat src_CV_U8;
    // printf("test 1\n");
    cv::convertScaleAbs(roiMat, src_CV_U8);
    detected_edges_CV_8U.create(roiMat.size(), CV_8U);

    // cv::Canny(grad_x, grad_y, cannyEdges, T1, T2);
    // makes the edge image with T
    cv::Canny(src_CV_U8, detected_edges_CV_8U, T1, T2, kernel_size);

    // get angle in degrees at every pixel - only if needed
    if (angle != -1)
        cv::phase(grad_x, grad_y, angles, true);

    // convert CV_U8 Mat to int Mat_
    cv::Mat_<int> detected_edges_int;
    detected_edges_CV_8U.convertTo(detected_edges_int, CV_32S);

    for (int row = Y, matRow = 0; row < Y + Sy && matRow < detected_edges_int.rows; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx && matCol < detected_edges_int.cols; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        int newPixel = detected_edges_int(matRow, matCol);

        // check angle if requested
        if((angle != -1))
        {
            double a = angles(matRow, matCol);
            // if angle out of range set pixel to black
            if (a < angle - 10 || a > angle + 10)
                newPixel = 0;
        }

        tgt.setPixel(row, col, newPixel);

        if (isColor)
        {
            tgt.setPixel(row, col, RED, newPixel);
            tgt.setPixel(row, col, GREEN, newPixel);
            tgt.setPixel(row, col, BLUE, newPixel);
        }
    }

    // copy Mat results back into tgt image type
}