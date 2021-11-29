#include "utility.h"
#include <string.h>


#define MAXRGB 255
#define MINRGB 0
#define MINNORM 0.0
#define MAXNORM 1.0
#define MINHUE 0.0
#define MAXHUE 360.0

std::string utility::intToString(int number)
{
   std::stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

int utility::checkValue(int value)
{
	if (value > MAXRGB)
		return MAXRGB;
	if (value < MINRGB)
		return MINRGB;
	return value;
}

/*-------------------------------------------------------------------*/
// set all pixels in the ROI to 1 to mark them as modified
void utility::setModifiedROI(image &wasModified, const int Sx, const int Sy, const int X, const int Y)
{
	for (int col = X; col < Sx + X; ++col)
		for(int row = Y; row < Sy + Y; ++row)
		{
			if (wasModified.isInbounds(row, col))
				wasModified.setPixel(row, col, 1);
		}
}

/*
	NOTES
	all functions operate within ROI only

	all images use row, col order

	Behaviors:
		if ROI is out of bounds, return without modifying image

*/

/*-----------------------------------------------------------------------**/
void utility::addGrey(image &src, image &tgt, int value, const ROI ROI_parameters)
{

	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

	// in loop make sure pixel is inbounds of the image
	// some pixels in the ROI might fall outside the image

	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());

	for (int row = Y; row < Y + Sy; ++row)
		for (int col = X; col < X + Sx; ++col)
		{
			if(tgt.isInbounds(row, col))
			{
				tgt.setPixel( row, col, checkValue(src.getPixel(row, col) + value) ); 
			}
		}
}

/*-----------------------------------------------------------------------**/
void utility::binarize(image &src, image &tgt, int threshold, const ROI ROI_parameters)
{
	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());

	for (int row = Y; row < Y + Sy; ++row)
		for (int col = X; col < X + Sx; ++col)
		{
			if(!tgt.isInbounds(row, col)) continue;

			if (src.getPixel(row, col) < threshold)
				tgt.setPixel(row, col, MINRGB);
			else
				tgt.setPixel(row, col, MAXRGB);
		}

}

/*-----------------------------------------------------------------------**/
void utility::dual_threshold(image &src, image &tgt, int T, int V1, int V2, const ROI ROI_parameters)
{

	/*
		inputs: threshold T, Values V1, V2
		if pixel intensity:
			> T, set pixel to V1
			< T, set pixel to V2
			== T, do nothing
	*/

	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;



	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());

	for (int row = Y; row < Y + Sy; ++row)
		for (int col = X; col < X + Sx; ++col)
		{
			// skip if pixel out of bounds
			if(!tgt.isInbounds(row, col)) continue;
			if (src.getPixel(row, col) == T)
				;	// do nothing
			else if (src.getPixel(row, col) < T)
				tgt.setPixel(row, col, checkValue(V2));
			else if (src.getPixel(row, col) > T)
				tgt.setPixel(row, col, checkValue(V1));
		}

}

/*-----------------------------------------------------------------------**/
void utility::scale(image &src, image &tgt, float ratio)
{
	int rows = (int)((float)src.getNumberOfRows() * ratio);
	int cols  = (int)((float)src.getNumberOfColumns() * ratio);
	tgt.resize(rows, cols);
	for (int i=0; i<rows; i++)
	{
		for (int j=0; j<cols; j++)
		{	
			/* Map the pixel of new image back to original image */
			int i2 = (int)floor((float)i/ratio);
			int j2 = (int)floor((float)j/ratio);
			if (ratio == 2) {
				/* Directly copy the value */
				tgt.setPixel(i,j,checkValue(src.getPixel(i2,j2)));
			}

			if (ratio == 0.5) {
				/* Average the values of four pixels */
				int value = src.getPixel(i2,j2) + src.getPixel(i2,j2+1) + src.getPixel(i2+1,j2) + src.getPixel(i2+1,j2+1);
				tgt.setPixel(i,j,checkValue(value/4));
			}
		}
	}
}


// Project 1 Functions
/*-----------------------------------------------------------------------**/
void utility::uniformSmoothing(image &src, image &tgt, int WS, ROI ROI_parameters)
{
	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

	// setup output image
	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());

	// offset from center of window = WS / 2 (integer division with bitshift)
   int offset = WS >> 1;  


	for (int row = Y; row < Y + Sy; ++row)
		for (int col = X; col < X + Sx; ++col)
		{
			// skip if pixel out of bounds
			if(!tgt.isInbounds(row, col)) continue;

			// track number of pixels and sum of pixel values
			int count_pixels = 0;
			int sum_pixels = 0;

			// process the values inside the window around the center pixel
			for (int row_w = row - offset; row_w < row + offset; ++row_w)
				for (int col_w = col - offset; col_w < col + offset; ++col_w)
				{
					// skip if pixel out of bounds
					if(!tgt.isInbounds(row_w, col_w)) continue;

					sum_pixels += src.getPixel(row_w, col_w);
					count_pixels += 1;
				}

			//set new pixel value based on window average
			// catch divide by zero error, set to 0
			if (count_pixels != 0)
				tgt.setPixel(row, col, checkValue( (int)(sum_pixels/count_pixels) ));
			else tgt.setPixel(row, col, 0);

		}


}


/* Color functions ----------------------------------------------------------------------**/

/*-----------------------------------------------------------------------**/
void utility::colorMultiplicativeBrightness(image &src, image &tgt, float C, ROI ROI_parameters)
{


	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

	// setup output image
	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());

	for (int row = Y; row < Y + Sy; ++row)
		for (int col = X; col < X + Sx; ++col)
		{
			// make sure pixel is inbounds
			if(!tgt.isInbounds(row, col)) continue;

			// get values from each channel in each pixel

			// formula - multiplies original pixel by C
			// R1=R*More-C, G1=G*More-C, B1=B*More-C.

			int R, G, B, R_new, G_new, B_new;

			R = src.getPixel(row, col, RED);
			G = src.getPixel(row, col, GREEN);
			B = src.getPixel(row, col, BLUE);

			R_new = int(R*C);
			G_new = int(G*C);
			B_new = int(B*C);

			tgt.setPixel(row, col, RED, checkValue(R_new));
			tgt.setPixel(row, col, GREEN, checkValue(G_new));
			tgt.setPixel(row, col, BLUE, checkValue(B_new));


		}
}

/*-----------------------------------------------------------------------**/
void utility::colorBinarize(image &src, image &tgt, int T_Color, int CR, int CG, int CB, ROI ROI_parameters)
{

	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

	// setup output image
	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());

	for (int row = Y; row < Y + Sy; ++row)
		for (int col = X; col < X + Sx; ++col)
		{
			// make sure pixel is inbounds
			if(!tgt.isInbounds(row, col)) continue;

			//get original values
			int R, G, B;

			R = src.getPixel(row, col, RED);
			G = src.getPixel(row, col, GREEN);
			B = src.getPixel(row, col, BLUE);

			// get color distance

			// formula: distance = sqrt( (R2 - R1)^2 +  (G2 - G1)^2 + (B2 - B1)^2)

			double distance = sqrt( pow((R - CR), 2) + pow((G - CG), 2) + pow((B - CB), 2) );

			// if pixel is within T_Color distance to C, set to red
			// between T_Color and 2*T_Color, set to green
			// else white

			if (distance <= T_Color)
			{
				tgt.setPixel(row, col, RED, MAXRGB);	// all red
				tgt.setPixel(row, col, GREEN, MINRGB);
				tgt.setPixel(row, col, BLUE, MINRGB);
			}

			else if (distance > T_Color && distance <= 2 * T_Color)
			{
				tgt.setPixel(row, col, RED, MINRGB);
				tgt.setPixel(row, col, GREEN, MAXRGB);	// all green
				tgt.setPixel(row, col, BLUE, MINRGB);
			}

			else
			{
				tgt.setPixel(row, col, RED, MAXRGB);
				tgt.setPixel(row, col, GREEN, MAXRGB);
				tgt.setPixel(row, col, BLUE, MAXRGB);
			}
		}
}
/*-----------------------------------------------------------------------**/
// PROJECT 2
/*-----------------------------------------------------------------------**/

/*-----------------------------------------------------------------------**/
//use to transform ranges for histogram stretching
// make sure a value is in range
// if below, set to min, if above, set to max
template <class T>
T check_value(T value, T minVal, T maxVal)
{
	if (value <= minVal ) return minVal;
	else if (value >= maxVal) return maxVal;
	else return value;
}


template <class T, class V>
T range_transform(const V in, const V inMin, const V inMax, const T outMin, const T outMax)
{

	double out, inRange, outRange;

	inRange = inMax - inMin;

	// check for edge case, prevent didide by 0
	if (inRange == 0) return outMin;

	else if (in <= inMin) return outMin;
	else if (in >= inMax) return outMax;

	else
	{
		outRange = outMax - outMin;


		return (T)((((in - inMin) * (outMax - outMin)) / (inMax - inMin)) + outMin);
	}
}

/*-----------------------------------------------------------------------**/
/* 
	creates the histogram array from the ROI in src
	inputs:
		src - the source image
		minVal - min value to count in the histogram
		maxVal - max value to count in the histogram
		RGB - channel to consider in RGB 
			for HSI R = H, G = S, B = I

		ROI_parameters
			contains the ROI position and size, only count pixels in here


		use range transform to convert input range to histogram range
		new range = [0, histo_width]

		then transforms the counts such that the maximum value = histo height
		NOTES:
		use min and max to use this function with other ranges like HSI channels
 */
void create_histogram_array(image &src, unsigned * countArray, 
							unsigned histo_height, unsigned histo_width, 
							ROI ROI_parameters,
							int minVal=0, int maxVal=255,
							int RGB=RED)
{
	// ROI variables
	unsigned Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

	// keep track on number of pixels counted
	// track max and min val to fix histogram height
	unsigned totalPixelCount{0}, maxPixelCount{0};

	for (int row = Y; row < Y + Sy; ++row)
		for (int col = X; col < X + Sx; ++col)
		{
			if (src.isInbounds(row, col))
			{
				++totalPixelCount;
				int pixelVal = src.getPixel(row, col, RGB);

				// convert pixel value to histogram width range [0, histo_width]
				unsigned countedVal = (unsigned)range_transform(pixelVal, minVal, maxVal, 0.0, (double)histo_width);
				//increment count array
				++countArray[countedVal];
			}
		}

	for (int i = 0; i < histo_width; ++i)
	{
		if(countArray[i] > maxPixelCount) maxPixelCount = countArray[i];
	}

	for (int i = 0; i < histo_width; ++i)
	{
		// convert max value in each bucket to histo_height with range transform 
		// fix the histogram height range from [0, totalPixelCount] to [0, histo_height]
		countArray[i] = range_transform(countArray[i], (unsigned)0, maxPixelCount, (unsigned)0, histo_height);
	}
}

template<typename T>
void create_histogram_array(vector<T> &src, unsigned * countArray, 
							unsigned histo_height, unsigned histo_width, 
							T minVal, T maxVal)
{

	// keep track on number of pixels counted
	// track max and min val to fix histogram height
	unsigned totalPixelCount{0}, maxPixelCount{0};

	for (int i = 0; i < src.size(); ++i)
	{
		++totalPixelCount;
		double pixelVal = src[i];

		// convert pixel value to histogram width range [0, histo_width]
		unsigned countedVal = (unsigned)range_transform((double)pixelVal, (double)minVal, (double)maxVal, 0.0, (double)histo_width);
		//increment count array
		++countArray[countedVal];
	}

	for (int i = 0; i < histo_width; ++i)
	{
		if(countArray[i] > maxPixelCount) maxPixelCount = countArray[i];
	}

	for (int i = 0; i < histo_width; ++i)
	{
		// convert max value in each bucket to histo_height with range transform 
		// fix the histogram height range from [0, totalPixelCount] to [0, histo_height]
		countArray[i] = range_transform(countArray[i], (unsigned)0, maxPixelCount, (unsigned)0, histo_height);
	}
}

/* 

	visually print out the histogram array to the console

 */
void print_histogram_array(unsigned * countArray, unsigned histo_width, unsigned valsPerLine, unsigned combineFactor=8)
{
	unsigned tempCount = 0;
	for (int i = 0; i < histo_width; ++i)
	{
		if (i % combineFactor != 0) tempCount += countArray[i];
		else
		{
			if (i % valsPerLine == 0) printf("\n ************************** \n");
			printf("bucket: %d, count: %d\t", i, tempCount);
			tempCount = 0;
		}
	}

	printf("\n ************************** \n");
}

void save_histogram_image(image &histogram_image, unsigned * countArray,
							unsigned histo_height, unsigned histo_width,
							const char * saveFileName,
							unsigned bucketSize=2, bool saveImage = true)
{
	//set histogram size
	histogram_image.resize(histo_height, histo_width);


	//bucket size version
	// make bars in the image representing the bucket magnitude in the histogram countArray
	// increment by bucket size and make bucketSize bars
	for (int col = 0; col <= histo_width - bucketSize; col += bucketSize)
		for (int row = histo_height - 1; row >= 0; --row)
		{
			// get sum of columns to combine and make one bucket
			unsigned bucketSum{0};
			for (int bucket = 0; bucket < bucketSize && (col+bucket) < histo_width; ++bucket)
				bucketSum += countArray[col+bucket];

			// fill enough colums to make a bucketSize wide column
			for (int bucket = 0; bucket < bucketSize; ++bucket)
			{	
				// start from the bottom
				//white bg with black pixels for bars
				if ( (histo_height - row) < bucketSum)
					histogram_image.setPixel(row, col + bucket, MINRGB);
				else 
					histogram_image.setPixel(row, col + bucket, MAXRGB);
			}
		}

	if (saveImage) histogram_image.save(saveFileName);
}
/* 

	vector input version

 */
template <typename T, typename V>
void do_histogram(vector<T> &src,
					unsigned histo_height, unsigned histo_width,
					ROI ROI_parameters,
					unsigned bucketSize=2,
					bool isModified=false,
					V minVal=0.0, V maxVal=100.0)
{
	// create, allocate, and init to all 0's
	unsigned * countArray;
	size_t histo_size = (histo_width + 2) * sizeof(countArray[0]);
	countArray = (unsigned *) malloc(histo_size);
	memset(countArray, 0, histo_size);
	
	image histogramImage;

	if (isModified)
		sprintf(ROI_parameters.histogramName, "%s_NEW.pgm", ROI_parameters.histogramName );


	create_histogram_array(src, countArray, histo_height, histo_width, minVal, maxVal);
	save_histogram_image(histogramImage, countArray, histo_height, histo_width, ROI_parameters.histogramName, bucketSize);


	// free mem
	free(countArray);
}

void do_histogram(image &src,
					unsigned histo_height, unsigned histo_width,
					ROI ROI_parameters,
					unsigned bucketSize=2,
					bool isModified=false,
					int minVal=0, int maxVal=255,
					int RGB=RED, bool saveImage = true)
{
	// create, allocate, and init to all 0's
	unsigned * countArray;
	size_t histo_size = (histo_width + 2) * sizeof(countArray[0]);
	countArray = (unsigned *) malloc(histo_size);
	memset(countArray, 0, histo_size);
	
	image histogramImage;

	if (isModified)
		sprintf(ROI_parameters.histogramName, "%s_NEW.pgm", ROI_parameters.histogramName );


	create_histogram_array(src, countArray, histo_height, histo_width, ROI_parameters, minVal, maxVal, RED);
	if(saveImage) save_histogram_image(histogramImage, countArray, histo_height, histo_width, ROI_parameters.histogramName, bucketSize);

	// free mem
	free(countArray);
}
/* 
	makes a histogram image in tgt
 */
void utility::make_histogram_image(image &src, image &tgt, bool isColor, ROI ROI_parameters)
{
	int histoHeight, histoWidth;
	int minVal=0;
	int maxVal=255;
	unsigned bucketSize = 2;

	histoHeight = histoWidth = 512;

	// create, allocate, and init to all 0's
	unsigned * countArray;
	size_t histo_size = (histoWidth + 2) * sizeof(countArray[0]);
	countArray = (unsigned *) malloc(histo_size);
	memset(countArray, 0, histo_size);
	
	image histogramImage;

	create_histogram_array(src, countArray, histoHeight, histoWidth, ROI_parameters, minVal, maxVal, RED);
	save_histogram_image(histogramImage, countArray, histoHeight, histoWidth, ROI_parameters.histogramName, bucketSize, true);

	// tgt.copyImage(histogramImage);

	// free mem
	free(countArray);
}



/*-----------------------------------------------------------------------**/
void utility::histo_stretch(image &src, image &tgt, int a1, int b1, ROI ROI_parameters)
{

	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

	// make the histogram

	unsigned histo_height{(MAXRGB + 1) * 2}, histo_width{(MAXRGB + 1) * 2}, bucketSize{2};


	vector<int> pixels, newPixels;

	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());

	// now transform pixels and set tgt image
	for (int row = Y; row < Y + Sy; ++row)
		for (int col = X; col < X + Sx; ++col)
		{
			if(tgt.isInbounds(row, col))
			{
				int pixel = src.getPixel(row, col);

				pixels.push_back(pixel);

				// get new pixel val and set
				int newVal = range_transform(pixel, a1, b1, MINRGB, MAXRGB);

				newPixels.push_back(newVal);

				tgt.setPixel(row, col, checkValue((int)newVal));
			}
		}

	do_histogram(pixels, histo_height, histo_width, ROI_parameters, bucketSize, false, MINRGB, MAXRGB);
	do_histogram(newPixels, histo_height, histo_width, ROI_parameters, bucketSize, true, MINRGB, MAXRGB);
	
}

/*-----------------------------------------------------------------------**/
void utility::thresh_histo_stretch(image &src, image &tgt, int T, int a1, int b1, int a2, int b2, ROI ROI_parameters)
{

	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

	unsigned histo_height{(MAXRGB + 1) * 2}, histo_width{(MAXRGB + 1) * 2}, bucketSize{2};

	do_histogram(src, histo_height, histo_width, ROI_parameters, bucketSize);

	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());

	// now transform pixels and set tgt image
	for (int row = Y; row < Y + Sy; ++row)
		for (int col = X; col < X + Sx; ++col)
		{
			if(src.isInbounds(row, col) && tgt.isInbounds(row, col))
			{
				int pixel = src.getPixel(row, col);
				int newVal;

				if (pixel < T)
				{
					// dark pixels, use a1, b1
					newVal = range_transform(pixel, a1, b1, MINRGB, T);
				}
				else // >= T bright pixels, use a2, b2
				{
					newVal = range_transform(pixel, a2, b2, T, MAXRGB);
				}
				// set new pixel value
				tgt.setPixel(row, col, checkValue((int)newVal));
			}
		}

	do_histogram(tgt, histo_height, histo_width, ROI_parameters, floor(histo_width / (b1 - a1)), true);
}

/*Project 2 Color functions*/


/*-----------------------------------------------------------------------**/
/*
	RGB to HSI
	converts a single pixel
 	NOTE formula from wikipedia
	H[0, 360] S and I [0, 1]

	RGB range [MINRGB, MAXRGB]
*/
HSI_pixel utility::RGB_to_HSI(RGB_pixel in)
{
	double R, G, B, min_RGB, max_RGB, C;
	double H, S, I, h_dash;

	// check and fix values if needed
	R = check_value(in.R, MINRGB, MAXRGB);
	G = check_value(in.G, MINRGB, MAXRGB);
	B = check_value(in.B, MINRGB, MAXRGB);

	// BASE CASES

	if (R == 0 && G == 0 && B == 0)
	{
		H = S = I = 0.0;
	}

	else if (R == 255 && G == 255 && B == 255)
	{
		H = 0.0;
		S = 0.0;
		I = 255;
	}

	else
	{
		min_RGB = fmin(fmin(R, G), B);
		max_RGB = fmax(fmax(R, G), B);

		// chroma
		C = max_RGB - min_RGB;

		/*
			intensity
			I = 1/3 (R + G + B)
		*/

		I = (R + G + B)/3.0;

		if (R == G && G == B) 
			H = 0;

		else
		{
			/*
				saturation
				S = 0 if I = 0
				else, S = 1 - min(RGB) / I
			*/
			if (C == 0)
			{
				// H = 0;
				S = 0;
			}
			else if (I == 0) 
				S = 0;
			else 
				S = 1 - (min_RGB / I);

			// HUE

			if (C == 0) 
				H = 0;
			else
			{
				if (max_RGB == R) 
					H = fmod( (G - B)/C, 6);
				else if (max_RGB == G) 
					H = (B - R)/ C + 2;
				else if (max_RGB == B)
					H = (R - G) / C + 4;
				//undefined
				else 
					H = 0;
				
				H *= 60;
			}
		}
	}

	if (H < 0) H += 360;

	HSI_pixel out;

	out = (HSI_pixel){.H = H, .S = S, .I = I/255};

	return out;
}
/*-----------------------------------------------------------------------**/
/*
	HSI to RGB
	converts a single pixel

	H [0, 360] float 
	S and I [0, 1] float

	output:
	RGB [MINRGB, MAXRGB] integer

	using formula from wikipedia
	https://en.wikipedia.org/wiki/HSL_and_HSV#HSI_to_RGB

*/
RGB_pixel utility::HSI_to_RGB(HSI_pixel in)
{

	// double R, G, B, H, S, I, C, X, h_dash, Z, m;
	double R, G, B, H, S, I;
	RGB_pixel out;

	// R = G = B = 0;

	// make sure values are correct
	H = check_value(in.H, MINHUE, MAXHUE); 
	S = check_value(in.S, MINNORM, MAXNORM); 
	I = check_value(in.I, MINNORM, MAXNORM); 

	// SPECIAL CASES
	// if (I == 0) return (RGB_pixel){.R = 0, .G = 0, .B = 0};
	// if (S == 0) return (RGB_pixel){.R = I*255, .G = I*255, .B = I* 255};
	// if (I == 1.0) return (RGB_pixel){.R = 255, .G = 255, .B = 255};


	double h_dash, Z, chroma, X;

	h_dash = H / 60.0; 	
	Z = 1.0 - fabs(fmod(h_dash, 2.0) - 1.0);
	chroma = (3.0 * I * S) / (1.0 + Z);
	X = chroma * Z;

	if ( 0 <= h_dash && h_dash <= 1)
		{
			R = chroma;
			G = X;
			B = 0;
		}
	else if (1 <= h_dash && h_dash <= 2)
		{
			R = X;
			G = chroma;
			B = 0;
		}
	else if (2 <= h_dash && h_dash <= 3)
		{
			R = 0;
			G = chroma;
			B = X;
		}
	else if (3 <= h_dash && h_dash <= 4)
		{
			R = 0;
			G = X;
			B = chroma;
		}
	else if (4 <= h_dash && h_dash <= 5)
		{
			R = X;
			G = 0;
			B = chroma;
		}
	else if (5 <= h_dash && h_dash <= 6)
		{
			R = chroma;
			G = 0;
			B = X;
		}
	else
		{
			R = 0;
			G = 0;
			B = 0;
		}

	double m = I * (1 - S);
	R = R + m;
	G = G + m;
	B = B + m;



	// prevent colors out of range
	double max_RGB = max(R, max(G, B));

	if (max_RGB > 1)
	{
		R = R/max_RGB;
		B = B/max_RGB;
		G = G/max_RGB;
	}


	R *= 255;
	G *= 255;
	B *= 255;

	out = (RGB_pixel){.R = (int)round(R), .G = (int)round(G), .B = (int)round(B)};
	return out;
}

// color funcitons
/* 

	stretch a user selected channel R G or B 
	user selected a and b values

 */
void utility::histo_stretch_RGB_single(image &src, image &tgt, 
									int a1, int b1, 
									int channel, ROI ROI_parameters)
{
	
	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

	// make the histogram

	unsigned histo_height{(MAXRGB + 1) * 2}, histo_width{(MAXRGB + 1) * 2}, bucketSize{2};

	do_histogram(src, histo_height, histo_width, ROI_parameters, bucketSize);

	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());

	// now transform pixels and set tgt image
	for (int row = Y; row < Y + Sy; ++row)
		for (int col = X; col < X + Sx; ++col)
		{
			if(tgt.isInbounds(row, col))
			{
				int pixel = src.getPixel(row, col, channel);

				// get new pixel val and set
				int newVal = range_transform(pixel, a1, b1, MINRGB, MAXRGB);
				tgt.setPixel(row, col, channel, checkValue((int)newVal));

				// set other channels
				for (int i = 0; i <= BLUE; ++i)
				{
					if (i != channel)
						tgt.setPixel(row, col, i, checkValue(src.getPixel(row, col, channel)));
				}

			}
		}

	do_histogram(tgt, histo_height, histo_width, ROI_parameters, floor(histo_width / (b1 - a1)), true, MINRGB, MAXRGB, channel);
	
}

/* 

	stretch all RGB channels
	user selected a and b values, one set for each channel

 */
void utility::histo_stretch_RGB_multi(image &src, image &tgt, 
									int aR, int bR,
									int aG, int bG, 
									int aB, int bB, 
									ROI ROI_parameters)
{
	
	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

	// make the histogram

	unsigned histo_height{(MAXRGB + 1) * 2}, histo_width{(MAXRGB + 1) * 2}, bucketSize{2};

	// do_histogram(src, histo_height, histo_width, ROI_parameters, bucketSize);

	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());

	// now transform pixels and set tgt image
	for (int row = Y; row < Y + Sy; ++row)
		for (int col = X; col < X + Sx; ++col)
		{
			if(tgt.isInbounds(row, col))
			{
				int pixelR = src.getPixel(row, col, RED);
				int pixelG = src.getPixel(row, col, GREEN);
				int pixelB = src.getPixel(row, col, BLUE);

				// get new pixel val and set
				int newValR = range_transform(pixelR, aR, bR, MINRGB, MAXRGB);
				int newValG = range_transform(pixelG, aG, bG, MINRGB, MAXRGB);
				int newValB = range_transform(pixelB, aB, bB, MINRGB, MAXRGB);

				tgt.setPixel(row, col, RED, checkValue((int)newValR));
				tgt.setPixel(row, col, GREEN, checkValue((int)newValG));
				tgt.setPixel(row, col, BLUE, checkValue((int)newValB));
			}
		}
	
}		

/* 
	convert RGB to HSI
	stretch I channel
	save hisotgram of I channel
	save gray level image of I channel

	convert back to RGB and save color image after stretched I channel

 */
void utility::histo_stretch_I(image &src, image &tgt, 
							double a1, double b1, 
							ROI ROI_parameters)
{
	
	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

	// make the histogram

	unsigned histo_height{(int)MAXNORM * 500}, histo_width{(int)MAXNORM * 500}, bucketSize{1};


	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());

	// use vectors to hold I channel values in ROI to generate histogram
	vector<double> I, newI;

	// now transform pixels and set tgt image
	for (int row = Y; row < Y + Sy; ++row)
		for (int col = X; col < X + Sx; ++col)
		{
			if(tgt.isInbounds(row, col))
			{
				// get RGB pixels
				int pixelR = src.getPixel(row, col, RED);
				int pixelG = src.getPixel(row, col, GREEN);
				int pixelB = src.getPixel(row, col, BLUE);

				// convert to HSI for stretching
				HSI_pixel pixelConverted = RGB_to_HSI(RGB_pixel{.R=pixelR, .G=pixelG, .B=pixelB});

				I.push_back(pixelConverted.I);

				// strech HSI pixels
				double newValI = range_transform(pixelConverted.I, a1, b1, MINNORM, MAXNORM);

				newI.push_back(newValI);

				// convert back to RGB
				RGB_pixel pixelModified = HSI_to_RGB(HSI_pixel{.H=pixelConverted.H, .S=pixelConverted.S, .I=newValI});

				// set values in new image
				tgt.setPixel(row, col, RED, checkValue((int)pixelModified.R));
				tgt.setPixel(row, col, GREEN, checkValue((int)pixelModified.G));
				tgt.setPixel(row, col, BLUE, checkValue((int)pixelModified.B));
			}
		}

	// do_histogram(tgt, histo_height, histo_width, ROI_parameters, floor(histo_width / (b1 - a1)), true);

	do_histogram(I, histo_height, histo_width, ROI_parameters, bucketSize, false, MINNORM, MAXNORM);
	do_histogram(newI, histo_height, histo_width, ROI_parameters, bucketSize, true, MINNORM, MAXNORM);

	
	
}

// set a b to min max to ignore channel
/* 

	stretch HSI channels to user provided [a, b] range
	
	pass in [min, max] or [0, 0] for [a, b] to ignore channel

 */
void utility::histo_stretch_HSI(image &src, image &tgt, 
							double aH, double bH,
							double aS, double bS,
							double aI, double bI, 
							ROI ROI_parameters)
{
	
	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

	// make the histogram

	vector<double> H, newH;

	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());

	// now transform pixels and set tgt image
	for (int row = Y; row < Y + Sy; ++row)
		for (int col = X; col < X + Sx; ++col)
		{
			if(tgt.isInbounds(row, col))
			{
				// get RGB pixels
				int pixelR = src.getPixel(row, col, RED);
				int pixelG = src.getPixel(row, col, GREEN);
				int pixelB = src.getPixel(row, col, BLUE);

				// convert to HSI for stretching
				HSI_pixel pixelConverted = RGB_to_HSI(RGB_pixel{.R=pixelR, .G=pixelG, .B=pixelB});


				// strech HSI pixels
				double newValH = range_transform(pixelConverted.H, aH, bH, MINHUE, MAXHUE);
				double newValS = range_transform(pixelConverted.S, aS, bS, MINNORM, MAXNORM);
				double newValI = range_transform(pixelConverted.I, aI, bI, MINNORM, MAXNORM);

				H.push_back(pixelConverted.H);
				newH.push_back(newValH);

				// convert back to RGB
				RGB_pixel pixelModified = HSI_to_RGB(HSI_pixel{.H=newValH, .S=newValS, .I=newValI});

				// set values in new image
				tgt.setPixel(row, col, RED, checkValue((int)pixelModified.R));
				tgt.setPixel(row, col, GREEN, checkValue((int)pixelModified.G));
				tgt.setPixel(row, col, BLUE, checkValue((int)pixelModified.B));
			}
		}

	do_histogram(H, 500, 500, ROI_parameters, 1, false, 0.0, 360.0);	
	do_histogram(newH, 500, 500, ROI_parameters, 1, true, 0.0, 360.0);	
}


/*-----------------------------------------------------------------------**/
// PROJECT 3
/*-----------------------------------------------------------------------**/

/*-----------------------------------------------------------------------**/

//index into vector using row, col
// row*data.numColumns+col
template <typename T>
T getValFrom2DVec(std::vector<T> &data, int row, int col, int num_cols)
{
	int index = row * num_cols + col;
	return data[index];
}

void utility::edge_detect(image &src, image &tgt, int kernel_size, bool isColor, ROI ROI_parameters)
{

	//TODO
	/* 
		applies the sobel filter to the input image
		computes: dx, dy, gradient amplitude, edge direction
		kernel_size can ONLY be 3 or 5
		IF a color image is used, the images is converted to HSI and the sobel filter is applied to the I channel
		
		outputs:
			grayscale image representing the amplitude of the gradient operator
			binary edge image derived from amplitude image by thresholding
			binary edge image thresholded with direction information 
				- option to display edges within range of degrees
	 */

	/*
		sobel kernels defined below
	*/

	vector<vector<int>> sobel_3_Gx
	{
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1}
	};

	vector<vector<int>> sobel_3_Gy
	{
		{-1, -2, -1},
		{0, 0, 0},
		{1, 2, 1}
	};

	vector<vector<int>> sobel_5_Gx
	{
		{-5, -4, 0, 4, 5},
		{-8, -10, 0, 10, 8},
		{-10, -20, 0, 20, 10,},
		{-8, -10, 0, 10, 8},
		{-5, -4, 0, 4, 5}
	};

	vector<vector<int>> sobel_5_Gy
	{
		{-5, -8, -10, -8, -5},
		{-4, -10, -20, -10, -4},
		{0, 0, 0, 0, 0},
		{4, 10, 20, 10, 4},
		{5, 8, 10, 8, 5}
	};

	// set correct kernel based on input  
	vector<vector<int>> kernel_x = (kernel_size == 3) ? sobel_3_Gx : sobel_5_Gx;
	vector<vector<int>> kernel_y = (kernel_size == 3) ? sobel_3_Gy : sobel_5_Gy;

	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

	// setup target image
	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());

	// offset from center of kernel window based on kernel size - bitshift division
	int offset = kernel_size >> 1;

	// now transform pixels and set tgt image
	for (int row = Y; row < Y + Sy; ++row)
	for (int col = X; col < X + Sx; ++col)
	if(src.isInbounds(row, col))
	{
		double Gx{0.0}, Gy{0.0}, G{0.0}, theta{0.0};
		// int kernel_row{0}, kernel_col{0};
		int pixels_processed_count = 0;

		int row_w, col_w;

		// process pixels for window
		// iterate through pixels in window
		// for each pixel, need to get each pixel in the window
		for (int kernel_row = 0, row_w = row - offset; kernel_row < kernel_size; ++kernel_row, ++row_w)
		for (int kernel_col = 0, col_w = col - offset; kernel_col < kernel_size; ++kernel_col, ++col_w)
		{
			
			// if (!src.isInbounds(row_w, col_w)) printf("OUTOF BOUUNDS FOOL\n");
			// ignore pixels in window that are outside image
			if (!src.isInbounds(row_w, col_w)) continue;

			// keep count of pixels processed inside window
			pixels_processed_count += 1;
			
			if(isColor)
			{
				// get RGB pixels
				int pixelR = src.getPixel(row_w, col_w, RED);
				int pixelG = src.getPixel(row_w, col_w, GREEN);
				int pixelB = src.getPixel(row_w, col_w, BLUE);

				// convert to HSI -- only need I channel
				double pixel = RGB_to_I(pixelR, pixelG, pixelB);
				// HSI_pixel pixelConverted = RGB_to_HSI(RGB_pixel{.R=pixelR, .G=pixelG, .B=pixelB});
			
				// double pixel = pixelConverted.I;

				Gx = Gx + (pixel * (double)kernel_x[kernel_row][kernel_col]);
				Gy = Gy +  (pixel * (double)kernel_y[kernel_row][kernel_col]);
			}

			// grayscale image version 
			// range [0 255]
			else 
			{
				int pixel = src.getPixel(row_w, col_w);

				Gx = Gx + ((double)pixel * (double)kernel_x[kernel_row][kernel_col]);
				Gy = Gy + ((double)pixel * (double)kernel_y[kernel_row][kernel_col]);
			}
		}

		// after window processed, get new pixel value
		// calculate Gx and Gy, then G
		// adjust so that range is [0 255] -> max value will be 4 * MAXRGB
		// Gx = Gx / 4;
		// Gy = Gy / 4;
		G = sqrt(Gx*Gx + Gy*Gy);
		theta = atan2(Gy, Gx);

		// set pixel in tgt for edge image
		// if derived from I channel, need to move to correct range 
		int newPixel;

		if (isColor) newPixel = G * 255.0;

		else
			newPixel = G;


		// set pixel in gradient image
		if (isColor)
		{
			// printf("color new pixel %d\n", newPixel);
			tgt.setPixel(row, col, RED, newPixel);
			tgt.setPixel(row, col, GREEN, newPixel);
			tgt.setPixel(row, col, BLUE, newPixel);
		}
		else
		tgt.setPixel(row, col, newPixel);

		// set meta info in image class
		tgt.setPixelMeta(row, col, GRADIENT, G);
		tgt.setPixelMeta(row, col, THETA, theta);
	}
}


// this function assumes that the image meta channels have 
// been set by another function and are not empty
void utility::edge_detect_binary(image &src, image &tgt, int kernel_size, int T, int angle, bool isColor, ROI ROI_parameters)
{

	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;

	image edgeImg(tgt);

	// do edge detection sobel
	edge_detect(src, edgeImg, kernel_size, isColor, ROI_parameters);
	// binarize image in ROI
	if (T != -1)
		binarize(edgeImg, tgt, T, ROI_parameters);
	// if no T, skip binary
	else
		tgt.copyImage(edgeImg);

	// set pixel in gradient image for color input
	if (isColor)
	{
		for (int row = Y; row < Y + Sy; ++row)
		for (int col = X; col < X + Sx; ++col)
		if (edgeImg.isInbounds(row, col))
		{
			int pixel = tgt.getPixel(row, col);
			// printf("color new pixel %d\n", newPixel);
			tgt.setPixel(row, col, RED, pixel);
			tgt.setPixel(row, col, GREEN, pixel);
			tgt.setPixel(row, col, BLUE, pixel);
		}
	}

	// stop here if no angle requested, must also do binary if angle requested
	if(angle == -1 || T == -1) return;

	// only do this part if an angle is requested
	// iterate through pixels in ROI
	// set pixels outside angle to black (0)
	for (int row = Y; row < Y + Sy; ++row)
	for (int col = X; col < X + Sx; ++col)
	if (tgt.isInbounds(row, col))
	{
		// now do binary with angle info
		// used angle info to only show edges at correct angle
		double theta = edgeImg.getPixelMeta(row, col, THETA);
		double degrees = ( theta * 180 ) / M_PI;

		// if +- 10 degrees of requested angle
		if (!(degrees > angle-10 && degrees < angle+10))
		{
			// set pixel in gradient image
			if (isColor)
			{
				// printf("color new pixel %d\n", newPixel);
				tgt.setPixel(row, col, RED, 0);
				tgt.setPixel(row, col, GREEN, 0);
				tgt.setPixel(row, col, BLUE, 0);
			}
			else
			tgt.setPixel(row, col, 0);
		}
	}
}