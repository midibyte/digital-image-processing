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

    printf("inside sobel_opencv function\n");
	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

    // copy src into tgt, then overwrite changed pixels from ROI
    tgt.copyImage(src);

    // create a cv Mat the size of the ROI

    // use CV_32S type for calculations, then convert back to CV_8U for display
    cv::Mat roiMat(Sx, Sy, CV_32S);
    // int matRow{0}, matCol{0};

    std::cout << roiMat.size() << '\n';
    std::cout << roiMat.type() << "\n";

    // copy pixels from ROI into cv Mat
	for (int row = Y, matRow = 0; row < Y + Sy; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        roiMat.at<int>(matRow, matCol) =  src.getPixel(row, col);
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


    cv::Mat grad_x, grad_y, grad;
    cv::Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    cv::Sobel(roiMat, grad_x, ddepth, 1, 0, ksize, scale, delta, cv::BORDER_DEFAULT);

    /// Gradient Y
    cv::Sobel(roiMat, grad_y, ddepth, 0, 1, ksize, scale, delta, cv::BORDER_DEFAULT);

    //![convert]
    // convert and scale gradient back to CV_8U
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    //![convert]

    std::cout << "after convertscaleabs, type of gradx absgradx: " << grad_x.type() << "  " << abs_grad_x.type() << "\n";

    //![blend]
    /// Total Gradient (approximate)
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    std::cout << "addweightedtype: " << grad.type() << "\n";
    //![blend]

    // ![display]
    // printf("test4\n");

    // display image
    // cv::imshow("window", grad);
    // char key = (char)waitKey(0);

    // printf("test5\n");
    cv::Mat_<int> gradM(grad);

    std::cout << gradM.type() << "\n";

    // copy results from Mat to image object
    for (int row = Y, matRow = 0; row < Y + Sy && matRow < grad.rows; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx && matCol < grad.cols; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        tgt.setPixel(row, col, gradM(matRow, matCol));
    }

}

/* **************************************************************************************************************
    Canny edge detection
    steps:
    

 ***************************************************************************************************************/

void utility::canny_opencv(image &src, image &tgt, int T, int angle, int kernel_size, bool isColor, ROI ROI_parameters)
{
    //implemented with opencv functions 

    printf("inside canny_opencv function\n");
	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

    // copy src into tgt, then overwrite changed pixels from ROI
    tgt.copyImage(src);

    // create a cv Mat the size of the ROI

    // use CV_32S type for calculations, then convert back to CV_8U for display
    cv::Mat roiMat(Sx, Sy, CV_32S);
    // int matRow{0}, matCol{0};

    std::cout << roiMat.size() << '\n';
    std::cout << roiMat.type() << "\n";

    // copy pixels from ROI into cv Mat
	for (int row = Y, matRow = 0; row < Y + Sy; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        roiMat.at<int>(matRow, matCol) =  src.getPixel(row, col);
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

    // Canny ratio of lowerThreshold to upperThreshold recommended to be 3
    int ratio = 3;
    double T1, T2;
    T1 = (double)T;
    T2 = T1 * (double)ratio;

    // convert input to please cv and for more precision during calculation
    roiMat.convertTo(roiMat, CV_32F);

    // int ddepth = CV_64F;
    int ddepth = CV_16SC1;

    cv::Mat grad_x, grad_y, grad;
    // cv::Mat abs_grad_x, abs_grad_y;
    cv::Mat angles;
    cv::Mat detected_edges, dst;

    // Use sobel to get Gx Gy to later use to get angle 
    // generate gradient x and y
    cv::Sobel(roiMat, grad_x, ddepth, 1, 0, ksize, scale, delta, cv::BORDER_DEFAULT);
    cv::Sobel(roiMat, grad_y, ddepth, 0, 1, ksize, scale, delta, cv::BORDER_DEFAULT);

    cv::Mat cannyEdges;

    cannyEdges.create(roiMat.size(), CV_16SC1);
    std::cout << "canny out type before: " << cannyEdges.type() << "\n";
    std::cout << "gradx type " << grad_x.type() << "\n";



    // now pass Gx Gy to Canny
    // output from Canny should be in 8-bit format
    // inputs must be CV_16SC1 type
    // set output type for Canny
    detected_edges.create(roiMat.size(), CV_8U);
    // cv::Canny(grad_x, grad_y, cannyEdges, T1, T2);
    // makes the edge image with T
    cv::Canny(detected_edges, detected_edges, T1, T2, kernel_size);

    std::cout << "canny out type:: " << detected_edges.type() << "\n";


    // calculate angle from gradients - must be float type
    grad_x.convertTo(grad_x, CV_64F);
    grad_y.convertTo(grad_y, CV_64F);
    angles.create(grad_x.size(), grad_x.type());

    // get angle in degrees at ever pixel
    cv::phase(grad_x, grad_y, angles, true);

    //convert back to Mat_ type for easier accessing
    // cv::convertScaleAbs(cannyEdges, detected_edges);

    // equal to int type
    detected_edges.convertTo(detected_edges, CV_32S);
    cv::Mat_<int> gradM(detected_edges);

    std::cout << "gradient out, detected_edges type: "<< gradM.type() << ", " << detected_edges.type() << "\n";
    std::cout << "gradM, detected_edges size: " << gradM.size() << ", " << detected_edges.size() << "\n";

    for (int row = Y, matRow = 0; row < Y + Sy && matRow < gradM.rows; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx && matCol < gradM.cols; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        std::cout << "gradM value: " << gradM(matRow, matCol) << "\n";
        tgt.setPixel(row, col, gradM(matRow, matCol));
        std::cout << "pixel value: " << tgt.getPixel(row, col) << "\n";
    }

    // copy Mat results back into tgt image type
}