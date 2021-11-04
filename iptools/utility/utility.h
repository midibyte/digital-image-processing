#ifndef UTILITY_H
#define UTILITY_H

#include "../image/image.h"
#include <sstream>
#include <math.h>

// struct to handle ROI info
struct ROI
{
	// hold the size and location of the ROI
	int Sx, Sy, X, Y;
	char ogImageName[1024];
	char histogramName[1024];
	bool isModified = false;
	int idxROI;
	int option;
};

struct HSI_pixel
{
	/*
		hue [0, 360]
		saturation [0, 1]
		intensity [0, 1]
	*/

	double H, S, I;
};

struct RGB_pixel
{
	int R, G, B;
};

// returns I with range [0.0 1.0]
static double RGB_to_I(const int R, const int G, const int B)
{
	return (double)(R + G + B) / (3.0 * 255);
}

class utility
{
	public:
		utility();
		virtual ~utility();
		static std::string intToString(int number);
		static int checkValue(int value);
		static void addGrey(image &src, image &tgt, int value, const ROI ROI_parameters);
		static void binarize(image &src, image &tgt, int threshold, const ROI ROI_parameters);
		static void dual_threshold(image &src, image &tgt, int T, int V1, int V2, const ROI ROI_parameters);
		static void scale(image &src, image &tgt, float ratio);

		// extra utilities
		static void setModifiedROI(image &wasModified, const int Sx, const int Sy, const int X, const int Y);


		// Project 1 Functions

		static void uniformSmoothing(image &src, image &tgt, int WS, ROI ROI_parameters);
		// Color functions
		static void colorMultiplicativeBrightness(image &src, image &tgt, float C, ROI ROI_parameters);
		static void colorBinarize(image &src, image &tgt, int T_Color, int CR, int CG, int CB, ROI ROI_parameters);

		// PROJECT 2

		static HSI_pixel RGB_to_HSI(RGB_pixel in);
		static RGB_pixel HSI_to_RGB(HSI_pixel in);

		static void histo_stretch(image &src, image &tgt, int a1, int b1, ROI ROI_parameters);
		static void thresh_histo_stretch(image &src, image &tgt,
										int T, int a1, int b1, 
										int a2, int b2, 
										ROI ROI_parameters);

		// color funcitons

		static void histo_stretch_RGB_single(image &src, image &tgt, 
											int a1, int b1, 
											int channel, ROI ROI_parameters);
		static void histo_stretch_RGB_multi(image &src, image &tgt, 
											int aR, int bR,
											int aG, int bG, 
											int aB, int bB, 
											ROI ROI_parameters);
											
		static void histo_stretch_I(image &src, image &tgt, 
									double a1, double b1, 
									ROI ROI_parameters);

		// set a b to min max to ignore channel
		static void histo_stretch_HSI(image &src, image &tgt, 
									double aH, double bH,
									double aS, double bS,
									double aI, double bI, 
									ROI ROI_parameters);

		template <typename T>
		static void histo_stretch_vector(vector<T> * data, T a, T b);

		static void make_histogram_image(image &src, image &tgt, bool isColor, ROI ROI_parameters);

		// project 3

		static void edge_detect(image &src, image &tgt, int kernel_size, bool isColor, ROI ROI_parameters);
		static void edge_detect_binary(image &src, image &tgt, int kernel_size, int T, int angle, bool isColor, ROI ROI_parameters);

		// funcitons using opencv
		static void sobel_opencv(image &src, image &tgt, int T, int angle, int kernel_size, bool isColor, ROI ROI_parameters);
		static void canny_opencv(image &src, image &tgt, int T, int angle, int kernel_size, bool isColor, ROI ROI_parameters);
		static void otsu_opencv(image &src, image &tgt, bool isColor, ROI ROI_parameters);
		static void equalize_foreground_otsu_opencv(image &src, image &tgt, bool isColor, ROI ROI_parameters);
		static void equalize_foreground_otsu_opencv_alt(image &src, image &tgt, bool isColor, ROI ROI_parameters);
		static void equalize_opencv(image &src, image &tgt, bool isColor, ROI ROI_parameters);

};

#endif

