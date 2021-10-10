#ifndef UTILITY_H
#define UTILITY_H

#include "../image/image.h"
#include <sstream>
#include <math.h>

// struct to handle ROI info
struct ROI
{
	// hold the size and location of the ROI
	unsigned int Sx, Sy, X, Y;
	char histogramName[1024];
	unsigned int idxROI;
};

// parameters to pass to functions
struct parameters
{
	// holds the number of ROIs - size of the ROI array
	unsigned int count_ROI;
	// holds the parameters for each ROI
	struct ROI * ROIs;
	// indicates debug print on or off
	unsigned int debug;

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
		static void thresh_histo_stretch(image src, image tgt, int T, int a1, int b1, int a2, int b2, ROI ROI_parameters);

};

#endif
