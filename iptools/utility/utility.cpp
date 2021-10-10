#include "utility.h"

#define MAXRGB 255
#define MINRGB 0

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
all functions operate within ROI only

uses row, col order

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
			tgt.setPixel(row, col, checkValue( (int)(sum_pixels/count_pixels) ));

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

			// printf("RGB, RGBnew: %d,%d,%d, %d,%d,%d\n", R, G, B, R_new, G_new, B_new);

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
template <class T>
T range_transform(const T in, const T inMin, const T inMax, const T outMin, const T outMax)
{

	T out, inRange, outRange;

	inRange = inMax - inMin;

	// check for edge case, prevent didide by 0
	if (inRange == 0) return outMin;

	else if (in <= inMin) return outMin;
	else if (in >= inMax) return outMax;

	else
	{
		outRange = outMax - outMin;


		return (((in - inMin) * (outMax - outMin)) / (inMax - inMin)) + outMin;
	}
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

	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());

	// now transform pixels and set tgt image
	for (int row = Y; row < Y + Sy; ++row)
		for (int col = X; col < X + Sx; ++col)
		{
			if(tgt.isInbounds(row, col))
			{
				int pixel = src.getPixel(row, col);
				if (pixel == -1) printf("get pixel error at row col: %d %d\n",row, col );

				int newVal = range_transform(pixel, a1, b1, 0, 255);
				tgt.setPixel(row, col, checkValue((int)newVal));

	
			}
		}
}


/*-----------------------------------------------------------------------**/
void utility::thresh_histo_stretch(image src, image tgt, int T, int a1, int b1, int a2, int b2, ROI ROI_parameters)
{

	// ROI variables
	unsigned int Sx, Sy, X, Y;
	Sx = ROI_parameters.Sx;
	Sy = ROI_parameters.Sy;
	X = ROI_parameters.X;
	Y = ROI_parameters.Y;


	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());

	// now transform pixels and set tgt image
	for (int row = Y; row < Y + Sy; ++row)
		for (int col = X; col < X + Sx; ++col)
		{
			if(src.isInbounds(row, col) && tgt.isInbounds(row, col))
			{
				
				int pixel = src.getPixel(row, col);
				int newVal;

				if (pixel <= T)
				{
					// dark pixels, use a1, b1
					newVal = range_transform(pixel, a1, b1, 0, 255);

				}
				else
				{
					newVal = range_transform(pixel, a2, b2, 0, 255);
				}

				tgt.setPixel(row, col, (int)newVal);
	
			}
		}

}

/*Project 2 Color functions*/

// make sure a value is in range
// if not set to max or min
template <class T>
T check_value(T value, T minVal, T maxVal)
{
	if (value <= minVal ) return minVal;
	else if (value >= maxVal) return maxVal;
	else return value;
}

/*-----------------------------------------------------------------------**/
/*
	RGB to HSI
	converts a single pixel
	// NOTE formula from wikipedia
	// H[0, 360] S and I [0, 1]
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
	RGB [0, 255] integer

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
	H = check_value(in.H, 0.0, 360.0); 
	S = check_value(in.S, 0.0, 1.0); 
	I = check_value(in.I, 0.0, 1.0); 

	// SPECIAL CASES
	// if (I == 0) return (RGB_pixel){.R = 0, .G = 0, .B = 0};
	// if (S == 0) return (RGB_pixel){.R = I*255, .G = I*255, .B = I* 255};
	// if (I == 1.0) return (RGB_pixel){.R = 255, .G = 255, .B = 255};


	double h_dash, Z, chroma, X;

	h_dash = H / 60.0;
	Z = 1.0 - abs(fmod(h_dash, 2.0) - 1.0);
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

	// R = (int)round(R);
	// G = (int)round(G);
	// B = (int)round(B);

	out = (RGB_pixel){.R = (int)round(R), .G = (int)round(G), .B = (int)round(B)};
	return out;
}
