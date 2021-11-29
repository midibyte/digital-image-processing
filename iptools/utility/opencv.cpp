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

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Various helper functions
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

void show_dft_magnitude_from_complex(cv::Mat in)
{



    // holds complex components of dft
    cv::Mat planes[] = {cv::Mat_<float>(in), cv::Mat::zeros(in.size(), CV_32F)};
    // cv::Mat planes[2];
    cv::Mat complex_roiMat;
    merge(planes, 2, complex_roiMat);

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
    cv::imshow("spectrum magnitude", mag_roiMat);
    cv::waitKey();
}


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// image pixels in ROI to a uchar Mat (CV_8U)
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void image_to_Mat_uchar(image &src, cv::Mat &outMat_uchar, bool isColor, ROI parameters)
{
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // load ROI from image object, copy pixels into cv::Mat
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = parameters.Sx;
	Sy = parameters.Sy;
	X = parameters.X;
	Y = parameters.Y;

    // setup mat for ROI - will hold uchar type - values on range [0 255]
    // cv::Mat roiMat_uchar;
    cv::Mat roiMat_uchar;

    if (isColor)
    {
        // roiMat_uchar.convertTo(roiMat_uchar, CV_8UC3);
        roiMat_uchar = cv::Mat(Sy, Sx, CV_8UC3, cv::Scalar::all(0));
    }
    else
    {
        roiMat_uchar = cv::Mat(Sy, Sx, CV_8U, cv::Scalar::all(0));
    }

    // copy pixels from ROI into cv Mat
	for (int row = Y, matRow = 0; row < Y + Sy; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {
        if (!isColor)
            roiMat_uchar.at<uchar>(matRow, matCol) = src.getPixel(matRow, matCol);
        else
        {
            // order is BGR

            cv::Vec3b &color = roiMat_uchar.at<cv::Vec3b>(row, col);

            // cv::Vec3b color;
            color[0] = src.getPixel(row, col, BLUE);
            color[1] = src.getPixel(row, col, GREEN);
            color[2] = src.getPixel(row, col, RED);

            // roiMat_uchar.at<cv::Vec3b>(cv::Point(matRow, matCol)) = color;

        }
    }

    // convert input to 8bit 
    // ** may not need this 
    // cv::convertScaleAbs(roiMat_uchar, outMat_uchar);
    outMat_uchar.convertTo(outMat_uchar, roiMat_uchar.type());
    roiMat_uchar.copyTo(outMat_uchar);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
/* 
    move the dft image to the center of the Mat
    can run this same funtion again to move the center back to the top left before iDFT
 
    does this :
    q0  q1
    q2  q3

    TO

    q3  q2
    q1  q0

 
    adapted from opencv docs
*/
void swap_quadrants(cv::Mat &magI)
{
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;
    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//make a filter to use on the image in the fourier domain
void make_circular_filter(cv::Mat &circularMask, cv::Size filterSize, ROI parameters)
{

    double filterRadiusPercent = parameters.filter_radius;

    // generate circular filter mask. w/ circle at center of image
    circularMask = cv::Mat::zeros(filterSize, CV_8U);

    cv::Mat circularMask_2 = cv::Mat::zeros(filterSize, CV_8U);


    double pixel_size;

    // choose the larger size of the image to set the filtersize
    if(circularMask.rows > circularMask.cols) pixel_size = (double)circularMask.rows;
    else pixel_size = (double)circularMask.cols;

    // change diameter to radius
    pixel_size /= 2;

    //set the size of the filter circle
    double filter_radius_pixels = (filterRadiusPercent * pixel_size);
    // draw a circle in the mask mat
    cv::Point circleCenter = cv::Point(circularMask.cols / 2, circularMask.rows / 2);
    cv::circle(circularMask, circleCenter, filter_radius_pixels, cv::Scalar(255, 0, 0), -1);

    //the mask is white in the center - low pass
    // invert if high pass is needed
    if(parameters.high_pass)
    {
        cv::bitwise_not(circularMask, circularMask);
    }

    if(parameters.band_pass)
    {
        // need to make two circles for band pass and take the inersection 
        // draw a smaller black circle in the middle of the larger circle
        circleCenter = cv::Point(circularMask.cols / 2, circularMask.rows / 2);
        cv::circle(circularMask, circleCenter, parameters.filter_radius_2 * pixel_size, cv::Scalar(0, 0, 0), -1);

    }
}
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// adapted from code found on stackoverflow
// link: https://stackoverflow.com/questions/15955305/find-maximum-value-of-a-cvmat
void print_max_min(const cv::Mat &in)
{

    double minVal; 
    double maxVal; 
    cv::Point minLoc; 
    cv::Point maxLoc;

    minMaxLoc( in, &minVal, &maxVal, &minLoc, &maxLoc );

    std::cout << "min val: " << minVal << endl;
    std::cout << "max val: " << maxVal << endl;
}
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//  adapted from the opencv docs
void do_DFT(cv::Mat &I, cv::Mat &output)
{

    using namespace cv;

    std::cout << "size of input in do_DFT: "<< I.size() << "\n";

    // normalize(I, I, 0, 1, NORM_MINMAX);

    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    dft(complexI, complexI);            // this way the result may fit in the source matrix
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];
    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;
    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
    // imshow("Input Image"       , I   );    // Show the result
    // imshow("spectrum magnitude", magI);
    // waitKey();
    magI.copyTo(output);
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


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// DFT with opencv
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// void utility::DFT(image &src, image &tgt, bool isColor, ROI ROI_parameters)
// {
//     // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//     // load ROI from image object, copy pixels into cv::Mat
//     // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// 	// ROI variables
// 	unsigned int Sx, Sy, X, Y;
// 	Sx = ROI_parameters.Sx;
// 	Sy = ROI_parameters.Sy;
// 	X = ROI_parameters.X;
// 	Y = ROI_parameters.Y;

//     //setup tgt
//     tgt.copyImage(src);

//     // copy image pixels to Mat
//     cv::Mat roiMat;
//     image_to_Mat_uchar(src, roiMat, ROI_parameters);

//     // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//     //  begin DFT code
//     // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//     /* 
//         DFT code adapted from opencv docs
//         https://docs.opencv.org/3.4/d8/d01/tutorial_discrete_fourier_transform.html
//      */    
    
//     // get optimal size of Mat for dft, use to add border to ROI 
//     cv::Mat padded;
//     int optimal_rows = cv::getOptimalDFTSize( roiMat.rows );
//     int optimal_cols = cv::getOptimalDFTSize( roiMat.cols );
//     cv::copyMakeBorder( roiMat, padded, 0, optimal_rows - roiMat.rows, 0, optimal_cols - roiMat.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

//     // holds complex components of dft
//     // copies padded Mat and all zeros Mat to planes[] and converts padded to float
//     cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
//     cv::Mat complex_roiMat;
//     //make a multichannel Mat from two different Mats
//     merge(planes, 2, complex_roiMat);

//     // do dft with opencv
//     cv::dft(complex_roiMat, complex_roiMat);

//     // split result into planes
//     cv::split(complex_roiMat, planes);
//     cv::magnitude(planes[0], planes[1], planes[0]);
//     cv::Mat mag_roiMat = planes[0];

//     mag_roiMat += cv::Scalar::all(1);
//     cv::log(mag_roiMat, mag_roiMat);

//     // move dft origin to center
//     swap_quadrants(mag_roiMat);

//     cv::normalize(mag_roiMat, mag_roiMat, 0, 1, cv::NORM_MINMAX); // Transform the matrix with float values into a
//                                             // viewable image form (float between values 0 and 1).

//     // cv::imshow("Input Image"       , roiMat   );    // Show the result
//     // cv::imshow("spectrum magnitude", mag_roiMat);
//     // cv::waitKey();

//     // result is in mag_roiMat
//     // copy back to image object

//     // convert float Mat with range [0 1] to range [0 255] to place in Image obj
//     cv::Mat_<int> intMat;
//     mag_roiMat.convertTo(intMat, CV_32S, 255);
//     // cv::Mat_<unsigned short> tempMat(mag_roiMat);

//     // printf("TYPE OF THE FUCKING MAT: %d\n", mag_roiMat.type());

//     for (int row = Y, matRow = 0; row < Y + Sy && matRow < intMat.rows; ++row, ++matRow)
//     for (int col = X, matCol = 0; col < X + Sx && matCol < intMat.cols; ++col, ++matCol)
//     if(src.isInbounds(row, col))
//     {

//         int newPixel = intMat(matRow, matCol);

//         if (isColor)
//         {
//             tgt.setPixel(row, col, RED, newPixel);
//             tgt.setPixel(row, col, GREEN, newPixel);
//             tgt.setPixel(row, col, BLUE, newPixel);
//         }

//         else 
//             tgt.setPixel(row, col, newPixel);

//     }
// }

void utility::DFT(image &src, image &tgt, bool isColor, ROI ROI_parameters)
{
    // ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;


    //copy ROI to Mat
    cv::Mat roiMat_uchar;
    image_to_Mat_uchar(src, roiMat_uchar, false, ROI_parameters);

    //show roi with opencv functions
    // cv::imshow("ROI from image", roiMat_uchar);
    // cv::waitKey(0);

    /* 
        DFT code adapted from opencv docs
        https://docs.opencv.org/3.4/d8/d01/tutorial_discrete_fourier_transform.html
     */    

    // convert image to float
    cv::Mat Mat_roi_float;
    roiMat_uchar.convertTo(Mat_roi_float, CV_32F);
    
    // // handle output %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // // place into tgt image obj
    cv::Mat imageOut_uchar;

    do_DFT(Mat_roi_float, imageOut_uchar);

    // cv::imshow("dft image", imageOut_uchar);
    // cv::waitKey();

    // convert result to proper range for saving 
    cv::convertScaleAbs(imageOut_uchar, imageOut_uchar, 255);
    // set file name and save
    char outName[4096];
    sprintf(outName, "%.1024s_DFT.pgm", ROI_parameters.ogImageName);
    cv::imwrite(outName, imageOut_uchar);
}

/* 

Converts to fourier domain then back to spatial domain

 */
void utility::IDFT(image &src, image &tgt, bool isColor, ROI ROI_parameters)
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
    // 0 to 255 range
    cv::convertScaleAbs(roiMat_int, roiMat);

    /* 
        DFT code adapted from opencv docs
        https://docs.opencv.org/3.4/d8/d01/tutorial_discrete_fourier_transform.html
     */    

    // convert image to float
    cv::Mat roiMatFloat;
    roiMat.convertTo(roiMatFloat, CV_32F);

    //dft

    cv::Mat dft;
    cv::dft(roiMatFloat, dft, cv::DFT_SCALE | cv:: DFT_COMPLEX_OUTPUT);

    cv::Mat idft;
    cv::dft(dft, idft, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);

    cv::Mat imageOut;
    idft.convertTo(imageOut, CV_8U);


    // printf("TYPE OF THE FUCKING MAT: %d\n", inverseFourierTransform.type());

    for (int row = Y, matRow = 0; row < Y + Sy && matRow < imageOut.rows; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx && matCol < imageOut.cols; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {

        int newPixel = imageOut.at<uchar>(matRow, matCol);

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

Converts to fourier domain, applies filter, then converts back to spatial domain

 */
void utility::dft_filter(image &src, image &tgt, bool isColor, ROI ROI_parameters)
{

    // ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

    //setup tgt
    tgt.copyImage(src);

    //copy ROI to Mat
    cv::Mat roiMat_uchar, input_temp;
    image_to_Mat_uchar(src, input_temp, isColor, ROI_parameters);

    input_temp.convertTo(input_temp, CV_32F);

    cv::Mat colors[3];

    std::cout << "Number of channels in input_temp: "<< input_temp.channels() << " type: " << input_temp.type() << std::endl;

    if (isColor)
    {
        //convert BGR to HSV and get  needed component 
        cv::Mat temp(input_temp.size(), input_temp.type());
        cv::cvtColor(input_temp, temp, cv::COLOR_BGR2HSV);


        cv::split(temp, colors);

        if (ROI_parameters.H_filter)
        {
            roiMat_uchar = colors[0];
        }
        if (ROI_parameters.V_filter)
        {
            roiMat_uchar = colors[2];
        }

        printf("min max of colors[0]: \n");
        print_max_min(colors[0]);
        printf("min max of colors[1]: \n");
        print_max_min(colors[1]);
        printf("min max of colors[2]: \n");
        print_max_min(colors[2]);

    }

    /* 
        DFT code adapted from opencv docs
        https://docs.opencv.org/3.4/d8/d01/tutorial_discrete_fourier_transform.html
     */    

    // convert image to float
    cv::Mat Mat_roi_float;
    roiMat_uchar.convertTo(Mat_roi_float, CV_32F);

    std::cout << "Number of channels in Mat_roi_float: "<< Mat_roi_float.channels() << " type: " << Mat_roi_float.type() << std::endl;

    // cv::normalize(Mat_roi_float, Mat_roi_float, 0, 1, cv::NORM_MINMAX);
    
    //get optimal image image size for dft
    int rowPad = cv::getOptimalDFTSize(Mat_roi_float.rows);
    int colPad = cv::getOptimalDFTSize(Mat_roi_float.cols);
    // pad the image
    cv::copyMakeBorder(Mat_roi_float, Mat_roi_float, 0, rowPad - Mat_roi_float.rows, 0, colPad - Mat_roi_float.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    std::cout << "rowpad, colpad: " << rowPad << ", " << colPad << std::endl;

    // cv::imshow("input with border", Mat_roi_float);
    // cv::waitKey();

    // DFT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // planes will hold the the complex and real components from the dft
    cv::Mat planes[2];
    cv::Mat Mat_roi_complex_float;
    
    // do dft
    cv::dft(Mat_roi_float, Mat_roi_complex_float, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);

    // bring the origin to the center by swapping the quadrants in the image
    swap_quadrants(Mat_roi_complex_float);

    // split the real and imaginary components into the planes array -> planes[0] = real planes[1] = imag
    cv::split(Mat_roi_complex_float, planes);

    //get phase and magnitude from the real and imaginary components
    cv::Mat phase, magnitude;
    phase.zeros(planes[0].rows, planes[0].cols, CV_32F);
    magnitude.zeros(planes[0].rows, planes[0].cols, CV_32F);
    cv::cartToPolar(planes[0], planes[1], magnitude, phase);

    //make the circular filter mask
    cv::Mat circularMask;
    make_circular_filter(circularMask, magnitude.size(), ROI_parameters);

    std::cout << "circular mask size: " << circularMask.size() << "\n";

    // cv::imshow("circular mask", circularMask);
    // cv::waitKey(0);
    
    // filter image - only copy over pixels where filter is white [255]
    // only apply filter to the magnitude
    cv::Mat magnitude_filtered(magnitude.size(), magnitude.type());
    magnitude.copyTo(magnitude_filtered, circularMask);
    // dont apply to phase
    // phase.copyTo(phase, circularMask);

    printf("min max of Magnitude: \n");
    print_max_min(magnitude);
    printf("min max of circularmask: \n");
    print_max_min(circularMask);

    // printf("types of magnitude filtered, phase, circularFilter, magnitude: %d %d %d %d\n", magnitude_filtered.type(), phase.type(), circularMask.type(), magnitude.type());

    //convert phase and magnitude back to real and imaginary components
    cv::polarToCart(magnitude_filtered, phase, planes[0], planes[1]);
    //merge planes (real and imag) into one Mat
    cv::Mat idft_image;
    cv::merge(planes, 2, idft_image);

    // bring the origin back to its original position
    swap_quadrants(idft_image);

    cv::dft(idft_image, idft_image, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);

    printf("min max of idft_image: \n");
    print_max_min(idft_image);

    cv::Mat dft_of_idft;

    do_DFT(idft_image, dft_of_idft);

    printf("min max of dft_of_idft: \n");
    print_max_min(dft_of_idft);

    cv::convertScaleAbs(dft_of_idft, dft_of_idft, 255);

    char outName[4096];
    sprintf(outName, "%.1024s_filtered_DFT.pgm", ROI_parameters.ogImageName);
    cv::imwrite(outName, dft_of_idft);


    // crop output to be the same as the input
    cv::Rect cropArea(0, 0, input_temp.cols, input_temp.rows);
    cv::Mat idft_image_crop = idft_image(cropArea).clone();

    std::cout << "Number of channels in idft_image: "<< idft_image.channels() << " type: " << idft_image.type() << " size: " << idft_image.size << std::endl;
    std::cout << "Number of channels in idft_image_crop: "<< idft_image_crop.channels() << " type: " << idft_image_crop.type() << " size: " << idft_image_crop.size << std::endl;


    // handle output %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // place into tgt image obj
    cv::Mat imageOut_uchar;
    cv::normalize(idft_image_crop, idft_image_crop, 0, 1, cv::NORM_MINMAX);

    if (!isColor)
    {
        cv::convertScaleAbs(idft_image_crop, idft_image_crop, 255);
        idft_image_crop.convertTo(imageOut_uchar, CV_8U);
    }

    // printf("min max of imageOut_uchar: \n");
    // print_max_min(imageOut_uchar);

    cv::Mat colors_out[3];


    //if color convert from HSV back to RGB
    if (isColor)
    {
        if (ROI_parameters.H_filter)
        {
            // cv::normalize(idft_image_crop, idft_image_crop, 0, 365, cv::NORM_MINMAX, colors[1].type());
            colors[0] = idft_image_crop;
        }
        if (ROI_parameters.V_filter)
        {
            // cv::normalize(idft_image_crop, idft_image_crop, 0, 1, cv::NORM_MINMAX, colors[1].type());
            colors[2] = idft_image_crop;
        }

        //merge channels then convert HSV to BGR
        cv::Mat temp(input_temp.size(), input_temp.type());

        std::cout << "Number of channels in temp: "<< temp.channels() << " type: " << temp.type() << " size: " << temp.size << std::endl;
        std::cout << "Number of channels in colors[0]: "<< colors[0].channels() << " type: " << colors[0].type() << " size: " << colors[0].size << std::endl;
        std::cout << "Number of channels in colors[1]: "<< colors[1].channels() << " type: " << colors[1].type() << " size: " << colors[1].size << std::endl;
        std::cout << "Number of channels in colors[2]: "<< colors[2].channels() << " type: " << colors[2].type() << " size: " << colors[2].size << std::endl;


        cv::merge(colors, 3, temp);
        cv::cvtColor(temp, temp, cv::COLOR_HSV2BGR);
        
        // cv::normalize(temp, temp, 0, 1, cv::NORM_MINMAX);
        cv::convertScaleAbs(idft_image_crop, idft_image_crop, 255);
        // convert to uchar type with 3 channels
        temp.convertTo(temp, CV_8UC3);

        // split back into colors array
        cv::split(temp, colors_out);

    }

    std::cout << "Number of channels in colors_out[0]: "<< colors_out[0].channels() << " type: " << colors_out[0].type() << " size: " << colors_out[0].size << std::endl;


    for (int row = Y, matRow = 0; row < Y + Sy && matRow < imageOut_uchar.rows; ++row, ++matRow)
    for (int col = X, matCol = 0; col < X + Sx && matCol < imageOut_uchar.cols; ++col, ++matCol)
    if(src.isInbounds(row, col))
    {

        if (isColor)
        {
            tgt.setPixel(row, col, RED, colors_out[2].at<uchar>(matRow, matCol));
            tgt.setPixel(row, col, GREEN, colors_out[1].at<uchar>(matRow, matCol));
            tgt.setPixel(row, col, BLUE, colors_out[0].at<uchar>(matRow, matCol));
        }

        else 
            tgt.setPixel(row, col, imageOut_uchar.at<uchar>(matRow, matCol));

    }
}