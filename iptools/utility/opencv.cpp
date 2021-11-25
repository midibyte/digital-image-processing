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

template <class T, class V>
T range_transform(const V in, const V inMin, const V inMax, const T outMin, const T outMax)
{

	double inRange, outRange;

	inRange = inMax - inMin;

	// check for edge case, prevent didide by 0
	if (inRange == 0) return outMin;

	else if (in <= inMin) return outMin;
	else if (in >= inMax) return outMax;

	else
        return (T)((((in - inMin) * (outMax - outMin)) / (inMax - inMin)) + outMin);
}


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
        int pixel = 0;
        // handle color image, convert to HSI
        if(isColor)
			{
				// get RGB pixels
				int pixelR = src.getPixel(row, col, RED);
				int pixelG = src.getPixel(row, col, GREEN);
				int pixelB = src.getPixel(row, col, BLUE);

				// convert to HSI -- only need I channel
                double pixel_I = RGB_to_I(pixelR, pixelG, pixelB);

                // convert I channel pixel to int pixel range
                pixel = range_transform(pixel_I, 0.0, 1.0, 0, 255);
            }        
        else 
            pixel = src.getPixel(row, col);
        roiMat(matRow, matCol) =  pixel;
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


        if (isColor)
        {
            tgt.setPixel(row, col, RED, newPixel);
            tgt.setPixel(row, col, GREEN, newPixel);
            tgt.setPixel(row, col, BLUE, newPixel);
        }
        else
            tgt.setPixel(row, col, newPixel);
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

    // // copy pixels from ROI into cv Mat
	// for (int row = Y, matRow = 0; row < Y + Sy; ++row, ++matRow)
    // for (int col = X, matCol = 0; col < X + Sx; ++col, ++matCol)
    // if(src.isInbounds(row, col))
    // {
    //     roiMat(matRow, matCol) =  src.getPixel(row, col);
    // }
    // copy pixels from ROI into cv Mat
	for (int row = Y, matRow = 0; row < Y + Sy && matRow < roiMat.rows; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx && matCol < roiMat.cols; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        int pixel = 0;
        // handle color image, convert to HSI
        if(isColor)
			{
				// get RGB pixels
				int pixelR = src.getPixel(row, col, RED);
				int pixelG = src.getPixel(row, col, GREEN);
				int pixelB = src.getPixel(row, col, BLUE);

				// convert to HSI -- only need I channel
                double pixel_I = RGB_to_I(pixelR, pixelG, pixelB);

                // convert I channel pixel to int pixel range
                pixel = range_transform(pixel_I, 0.0, 1.0, 0, 255);
            }        
        else 
            pixel = src.getPixel(row, col);

        roiMat(matRow, matCol) =  pixel;
    }

    // test if pixels copied correctly
    for (int row = Y, matRow = 0; row < Y + Sy; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        // cv::Mat_<int> temp(roiMat);
        // check if values are the same
        if (src.getPixel(row, col) != roiMat(matRow, matCol) && !isColor)
            printf("canny pixels not equal: vals = %d %d\n", src.getPixel(row, col), roiMat(matRow, matCol));
        if ((int)(255.0 * RGB_to_I(src.getPixel(row, col, RED), src.getPixel(row, col, GREEN), src.getPixel(row, col, BLUE))) != roiMat(matRow, matCol) && isColor)
            printf("canny pixels not equal: vals = %d %d\n", (int)(255.0 * RGB_to_I(src.getPixel(row, col, RED), src.getPixel(row, col, GREEN), src.getPixel(row, col, BLUE))), roiMat(matRow, matCol));
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

void utility::otsu_opencv(image &src, image &tgt, bool isColor, ROI ROI_parameters)
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

    // convert input to 8bit 
    cv::Mat roiMat_CV_8U, otsu_result_CV_8U;
    cv::convertScaleAbs(roiMat, roiMat_CV_8U);

    // roiMat.convertTo(roiMat_CV_8U, CV_8U)

    // do binary otsu here
    // int otsu_threshold;
    // otsu_threshold = 
    cv::threshold(roiMat_CV_8U, otsu_result_CV_8U, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // convert CV_U8 Mat to int Mat_
    cv::Mat_<int> otsu_result_int;
    otsu_result_CV_8U.convertTo(otsu_result_int, CV_32S);

    // printf("Otsu calculated threshold value: %d\n", otsu_threshold);

    for (int row = Y, matRow = 0; row < Y + Sy && matRow < otsu_result_int.rows; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx && matCol < otsu_result_int.cols; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        int newPixel = otsu_result_int(matRow, matCol);


        if (isColor)
        {
            tgt.setPixel(row, col, RED, newPixel);
            tgt.setPixel(row, col, GREEN, newPixel);
            tgt.setPixel(row, col, BLUE, newPixel);
        }

        else 
            tgt.setPixel(row, col, newPixel);

    }
}

void utility::equalize_foreground_otsu_opencv_alt(image &src, image &tgt, bool isColor, ROI ROI_parameters)
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

    // convert input to 8bit 
    cv::Mat roiMat_CV_8U, otsu_result_CV_8U;
    cv::convertScaleAbs(roiMat, roiMat_CV_8U);

    // roiMat.convertTo(roiMat_CV_8U, CV_8U)

    // do binary otsu here
    int otsu_threshold;
    otsu_threshold = cv::threshold(roiMat_CV_8U, otsu_result_CV_8U, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);




    // convert CV_U8 Mat to int Mat_
    cv::Mat_<int> otsu_result_int;
    otsu_result_CV_8U.convertTo(otsu_result_int, CV_32S);

    // equalize histogram only on foreground
    // get only pixels above threshold and add to new mat
    std::vector<int> foreground;
    unsigned long int foregroundCount = 0;

    for (int matRow = 0; matRow < otsu_result_int.rows;  ++matRow)
    for (int matCol = 0; matCol < otsu_result_int.cols; ++matCol)
    {
        
        if (roiMat(matRow, matCol) >= otsu_threshold)
        {
            foreground.push_back(roiMat(matRow, matCol));
            ++foregroundCount;
        }
    }

    cv::Mat foreground_CV_8U, foreground_equalized_CV_8U;

    cv::Mat temp(foreground);
    temp.convertTo(foreground_CV_8U, CV_8U);

    cv::equalizeHist(foreground_CV_8U, foreground_equalized_CV_8U);

    // convert CV_U8 Mat to int Mat_
    cv::Mat_<int> foreground_equalized_int;
    foreground_equalized_CV_8U.convertTo(foreground_equalized_int, CV_32S);

    // printf("Otsu calculated threshold value: %d\n", otsu_threshold);

    unsigned long int fgPixel = 0;

    for (int row = Y, matRow = 0; row < Y + Sy && matRow < otsu_result_int.rows; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx && matCol < otsu_result_int.cols; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        int newPixel = roiMat(matRow, matCol);

        if (roiMat(matRow, matCol) >= otsu_threshold && foregroundCount != 0)
            {
                newPixel = foreground_equalized_int(fgPixel);
                ++fgPixel;
                --foregroundCount;
            }

        if (isColor)
        {
            tgt.setPixel(row, col, RED, newPixel);
            tgt.setPixel(row, col, GREEN, newPixel);
            tgt.setPixel(row, col, BLUE, newPixel);
        }

        else 
            tgt.setPixel(row, col, newPixel);

    }
}

void utility::equalize_opencv(image &src, image &tgt, bool isColor, ROI ROI_parameters)
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

    // convert input to 8bit 
    cv::Mat roiMat_CV_8U, equalized_CV_8U;
    cv::convertScaleAbs(roiMat, roiMat_CV_8U);

    cv::equalizeHist(roiMat_CV_8U, equalized_CV_8U);

    // convert CV_U8 Mat to int Mat_
    cv::Mat_<int> equalized_int;
    equalized_CV_8U.convertTo(equalized_int, CV_32S);


    unsigned long int fgPixel = 0;

    for (int row = Y, matRow = 0; row < Y + Sy && matRow < equalized_int.rows; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx && matCol < equalized_int.cols; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        int newPixel = equalized_int(matRow, matCol);

        if (isColor)
        {
            tgt.setPixel(row, col, RED, newPixel);
            tgt.setPixel(row, col, GREEN, newPixel);
            tgt.setPixel(row, col, BLUE, newPixel);
        }

        else 
            tgt.setPixel(row, col, newPixel);

    }
}

void utility::equalize_foreground_otsu_opencv(image &src, image &tgt, bool isColor, ROI ROI_parameters)
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
    image equalized(src);

    // equalize
    utility::equalize_opencv(src, equalized, isColor, ROI_parameters);

    // setup mat for ROI
    cv::Mat_<int> roiMat(Sy, Sx);

    // copy pixels from ROI into cv Mat
	for (int row = Y, matRow = 0; row < Y + Sy; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        roiMat(matRow, matCol) =  src.getPixel(row, col);
    }

    // convert input to 8bit 
    cv::Mat roiMat_CV_8U, otsu_result_CV_8U;
    cv::convertScaleAbs(roiMat, roiMat_CV_8U);

    // roiMat.convertTo(roiMat_CV_8U, CV_8U)

    // do binary otsu here
    int otsu_threshold;
    otsu_threshold = cv::threshold(roiMat_CV_8U, otsu_result_CV_8U, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);



    // printf("Otsu calculated threshold value: %d\n", otsu_threshold);

    // use otsu binary image as a mask
    for (int row = Y; row < Y + Sy; ++row)
    for (int col = X; col < X + Sx; ++col)
    if(src.isInbounds(row, col))
    {

        int newPixel = 0;

        if(src.getPixel(row, col) > otsu_threshold)
            newPixel = equalized.getPixel(row, col);
        else
            newPixel = src.getPixel(row, col);

        if (isColor)
        {
            tgt.setPixel(row, col, RED, newPixel);
            tgt.setPixel(row, col, GREEN, newPixel);
            tgt.setPixel(row, col, BLUE, newPixel);
        }

        else 
            tgt.setPixel(row, col, newPixel);

    }
}





/* 

Project 4 code starts here

 */


void utility::DFT(image &src, image &tgt, bool isColor, ROI ROI_parameters)
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
    cv::Mat_<int> roiMat_int(Sy, Sx);

    // copy pixels from ROI into cv Mat
	for (int row = Y, matRow = 0; row < Y + Sy; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        roiMat_int(matRow, matCol) =  src.getPixel(row, col);
    }

    cv::Mat roiMat;

    // convert input to 8bit 
    // cv::Mat roiMat_CV_8U, otsu_result_CV_8U;
    cv::convertScaleAbs(roiMat_int, roiMat);


    /* 
        DFT code adapted from opencv docs
        https://docs.opencv.org/3.4/d8/d01/tutorial_discrete_fourier_transform.html
     */    

    // get optimal size of Mat for dft, use to add border to ROI 
    cv::Mat padded;
    int optimal_rows = cv::getOptimalDFTSize( roiMat.rows );
    int optimal_cols = cv::getOptimalDFTSize( roiMat.cols );
    cv::copyMakeBorder( roiMat, padded, 0, optimal_rows - roiMat.rows, 0, optimal_cols - roiMat.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // holds complex components of dft
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complex_roiMat;
    merge(planes, 2, complex_roiMat);

    // do dft with opencv
    cv::dft(complex_roiMat, complex_roiMat);

    // split result into planes
    cv::split(complex_roiMat, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat mag_roiMat = planes[0];

    mag_roiMat += cv::Scalar::all(1);
    cv::log(mag_roiMat, mag_roiMat);

    //crop spectrum
    mag_roiMat = mag_roiMat(cv::Rect(0, 0, mag_roiMat.cols & -2, mag_roiMat.rows & -2));

    // place origin at image center

    int cx = mag_roiMat.cols / 2;
    int cy = mag_roiMat.rows / 2;

    cv::Mat q0(mag_roiMat, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(mag_roiMat, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(mag_roiMat, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(mag_roiMat, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    cv::normalize(mag_roiMat, mag_roiMat, 0, 1, cv::NORM_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).

    // cv::imshow("Input Image"       , roiMat   );    // Show the result
    // cv::imshow("spectrum magnitude", mag_roiMat);
    // cv::waitKey();

    // result is in mag_roiMat
    // copy back to image object

    // convert float Mat with range [0 1] to range [0 255] to place in Image obj
    cv::Mat_<int> intMat;
    mag_roiMat.convertTo(intMat, CV_32S, 255);
    // cv::Mat_<unsigned short> tempMat(mag_roiMat);

    printf("TYPE OF THE FUCKING MAT: %d\n", mag_roiMat.type());

    for (int row = Y, matRow = 0; row < Y + Sy && matRow < intMat.rows; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx && matCol < intMat.cols; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {

        int newPixel = intMat(matRow, matCol);

        if (isColor)
        {
            tgt.setPixel(row, col, RED, newPixel);
            tgt.setPixel(row, col, GREEN, newPixel);
            tgt.setPixel(row, col, BLUE, newPixel);
        }

        else 
            tgt.setPixel(row, col, newPixel);

    }
}
