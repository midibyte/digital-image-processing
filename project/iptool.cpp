/************************************************************
 *															*
 * This sample project include three functions:				*
 * 1. Add intensity for gray-level image.					*
 *    Input: source image, output image name, value			*
 *															*
 * 2. Image thresholding: pixels will become black if the	*
 *    intensity is below the threshold, and white if above	*
 *    or equal the threhold.								*
 *    Input: source image, output image name, threshold		*
 *															*
 * 3. Image scaling: reduction/expansion of 2 for 			*
 *    the width and length. This project uses averaging 	*
 *    technique for reduction and pixel replication			*
 *    technique for expansion.								*
 *    Input: source image, output image name, scale factor	*


	P0:

		4. Dual threshold
		
		inputs: threshold T, Values V1, V2
		if pixel intensity:
			> T, set pixel to V1
			< T, set pixel to V2
			== T, do nothing

	P1:
		add ROI to functions (except scaling)
			input file format: 
				in_file out_file function ROI_count ROI_parameters

			ROI_parameters format: (one for each ROI)
				Size_X Size_Y coord_X coord_Y
			NOTE: ROIs will be processed in order. if the next ROI overlaps the first, it will be skipped

		COLOR

			Implement uniform smoothing filter operation using square odd window size (WS). Implement adaptive processing when smoothing window is close to the ROI boundary by progressively reducing window size all the way to 3x3, then reduce to 3x2 (or 2x2 if close to the corner). [3 points]

			3. Processing of Color Images
			
			a. multiplicative color brightness modification to your toolbox. Let value More-C be user defined threshold. Process each color channel: R1=R*More-C, G1=G+More-C, B1=B+More-C. Make sure (R1, G1, B1) are within allowable values. This function should operate within specified ROI with different parameters for each ROI. [1 points]
			
			b. color binarization option to your image processing toolbox. Let threshold TColor be user defined input parameter. TColor defined as a Euclidian distance from user defined color C (CR, CB, CG) in RGB space. Set all pixels within TColor distance to "red", between TColor and 2*TColor to “green” and the rest to "white". This function should operate within specified ROI with different parameters for each ROI. [4 points]
	
	P2:
		gray level:
			make histogram image
			histogram stretching
			histogram stretching with thresholding 

		color:
			histo-stretch one channel (RGB)
			histo-stretch all RGB channels
			RGB to HSI conversion
			histo-stretch on I channel
				output gray and color
			
			EXTRA
				histo-stretch on HS
				histo-stretch on HSI

 *															*
 ************************************************************/

#include "../iptools/core.h"
#include <strings.h>
#include <string.h>

using namespace std;

// maxlen of strings
#define MAXLEN 1024

int main (int argc, char** argv)
{
	// holds images
	image src;
	// grayscale binary image 
	// val > 0 = pixel was modified
	image wasModified;
	// file pointer
	FILE *fp;
	// holds input line from file
	char str[MAXLEN];
	// output file name
	char outfile[MAXLEN];
	// input file name
	char infile[MAXLEN];
	// holds strings parsed from the lines in the parameters file
	char *pch;
	// holds function name
	char function_name[MAXLEN];
	unsigned int count_ROI;
	// debug options value
	int debug = 0;



	// open parameters file 
	if ((fp = fopen(argv[1],"r")) == NULL) {
		fprintf(stderr, "Can't open file: %s\n", argv[1]);
		exit(1);
	}

	// control printing debug to console
	if (argc > 2)
	{
		debug = atoi(argv[2]);
	}

	if ( debug > 0) printf("DEBUG ON\n");

	// read each line from the parameters file
	while(fgets(str,MAXLEN,fp) != NULL) {

		if ( debug > 0) printf("line read: %s\n", str);
		
		// skip comment lines in parameters file
		// if ( (char)*str == '#') continue;

		// get source image file name
		pch = strtok(str, " ");
		strcpy(infile, pch);
		src.read(pch);

		if (debug > 0) printf("source image: %s\n", pch);
		if (debug > 0) printf("img size X, Y: %d, %d\n", src.getNumberOfColumns(), src.getNumberOfRows());

		// get destination file name
		pch = strtok(NULL, " ");
		strcpy(outfile, pch);

		if (debug > 0) printf("dest image: %s\n", outfile);

		// read in function - then parameters
		pch = strtok(NULL, " ");
		//store function name
		strcpy(function_name, pch);

		if (debug > 0) printf("function_name: %s\n", function_name);


		// get number of ROIs
		pch = strtok(NULL, " ");
		count_ROI = atoi(pch);

		if ( debug > 0) printf("number of ROIs %d\n", count_ROI);

		// setup image to record modified pixels
		wasModified.resize(src.getNumberOfRows(), src.getNumberOfColumns());
		// setup final output image - copy of source, overwrite ROI areas with new pixels
		image final(src);


		// handle ROIs =========================================================
		// do the selected function for each ROI
		// at the end of each ROI, add to final image
		for (int idxROI = 0; idxROI < count_ROI; ++idxROI )
		{
			// struct holds ROI options
			struct ROI ROI_parameters;

			//reset tgt
			// tgt.deleteImage();
			image tgt;

			// read in sizeX, sizeY, X, Y for ROI
			// then get parameters for functions
			pch = strtok(NULL, " ");
			ROI_parameters.Sx = atoi(pch);
			pch = strtok(NULL, " ");
			ROI_parameters.Sy = atoi(pch);
			pch = strtok(NULL, " ");
			ROI_parameters.X = atoi(pch);
			pch = strtok(NULL, " ");
			ROI_parameters.Y = atoi(pch);

			ROI_parameters.idxROI=idxROI;
			
			char hName[MAXLEN];
			strcpy(hName, outfile);
			hName[strlen(hName) - 4] = '\0';

			sprintf(ROI_parameters.histogramName, "HISTOGRAM_%s_ROI_%d.pgm",hName, idxROI );

			if (debug > 0) printf("Sx, Sy, X, Y: %d, %d, %d, %d\n", ROI_parameters.Sx, ROI_parameters.Sy, ROI_parameters.X, ROI_parameters.Y );

			// make sure top left ROI coordnates are inside the image
			if (ROI_parameters.X < 0 || ROI_parameters.X > src.getNumberOfColumns() || ROI_parameters.Y < 0 || ROI_parameters.Y > src.getNumberOfRows())
			{
				printf("ROI top left X, Y coordinates out of image bounds\n");
				printf("ROI top left X, Y: %d, %d, Image size X, Y: %d, %d\n", ROI_parameters.X, ROI_parameters.Y, src.getNumberOfColumns(), src.getNumberOfRows());
				// skip, go to next ROI
				continue;
			}

			bool ROI_overlap = false;

			if (idxROI >= 1 )
			{
				// check for ROI overlap with previous ROI, skip if overlapping
				for (int row = ROI_parameters.Y; row < ROI_parameters.Y + ROI_parameters.Sy; ++row)
					for (int col = ROI_parameters.X; col < ROI_parameters.X + ROI_parameters.Sx; ++col)
					{
						if (wasModified.isInbounds(row, col))
						{
							if(wasModified.getPixel(row, col) > 0)
							{
								// overlaps with previous ROI, skip
								printf("ROI# %d overlaps with a previous ROI, skipping...\n", idxROI);
								ROI_overlap = true;
								goto overlap;
							}
						}
					}
			}

			// skip ROI if it overlaps with a previous ROI
			overlap: if(ROI_overlap) continue;

			// set pixels modified in this ROI
			// for checking overlap
			utility::setModifiedROI(wasModified, ROI_parameters.Sx, ROI_parameters.Sy, ROI_parameters.X, ROI_parameters.Y );


			/* Add Intensity ============================================*/
	        if (strncasecmp(function_name,"add",MAXLEN)==0) 
	        {
				pch = strtok(NULL, " ");
				if (debug > 0) printf("value: %d\n", atoi(pch));
	        	utility::addGrey(src,tgt,atoi(pch), ROI_parameters);
	        }

	        /* Thresholding ============================================*/
	        else if (strncasecmp(function_name,"binarize",MAXLEN)==0) 
	        {
				pch = strtok(NULL, " ");
				utility::binarize(src,tgt,atoi(pch), ROI_parameters);
			}

			/* Image scaling ============================================*/
			//  does not use ROIs
			else if (strncasecmp(function_name,"scale",MAXLEN)==0) 
			{
				pch = strtok(NULL, " ");
				utility::scale(src,tgt,atof(pch));
			}

			/* Dual thresholding - project 0 ============================================*/
			else if (strncasecmp(function_name,"dual_threshold",MAXLEN)==0) 
			{
				pch = strtok(NULL, " ");
				int T = atoi(pch);
				pch = strtok(NULL, " ");
				int V1 = atoi(pch);
				pch = strtok(NULL, " ");
				int V2 = atoi(pch);

				if (debug) printf("in iptool.cpp: T, V1, V2: %d %d %d\n", T, V1, V2);

				utility::dual_threshold(src,tgt, T, V1, V2, ROI_parameters);
			}

	        /* Uniform Smoothing ============================================*/
	        else if (strncasecmp(function_name,"uniformSmoothing",MAXLEN)==0) 
	        {
	        	//get window size WS
				pch = strtok(NULL, " ");
				unsigned int WS = atoi(pch);

				if (WS % 2 == 0)
				{
					// WS is even, show warning
					printf("WS: %d is not odd. Use an odd number for WS. skipping...\n", WS);
				}
				else utility::uniformSmoothing(src,tgt,WS , ROI_parameters);
			}

	        /* colorMultiplicativeBrightness ============================================*/
	        else if (strncasecmp(function_name,"colorMultiplicativeBrightness",MAXLEN)==0) 
	        {
	        	//get more-C
				pch = strtok(NULL, " ");
				if (debug) printf("more-C value: %f\n", atof(pch));
				utility::colorMultiplicativeBrightness(src,tgt, atof(pch), ROI_parameters);
			}

	        /* colorBinarize ============================================*/
	        else if (strncasecmp(function_name,"colorBinarize",MAXLEN)==0) 
	        {
	        	//get color values 
				pch = strtok(NULL, " ");
				int T_Color = atoi(pch);
				pch = strtok(NULL, " ");
				int CR = atoi(pch);
				pch = strtok(NULL, " ");
				int CG = atoi(pch);
				pch = strtok(NULL, " ");
				int CB = atoi(pch);

				utility::colorBinarize(src,tgt, T_Color, CR, CG, CB, ROI_parameters);
			}
// PROJECT 2 =================================================================
	        /* histo_stretch ============================================*/
	        else if (strncasecmp(function_name,"histo_stretch",MAXLEN)==0) 
	        {

				pch = strtok(NULL, " ");
				int a1 = atoi(pch);
				pch = strtok(NULL, " ");
				int b1 = atoi(pch);

				if (debug) printf("histo_stretch (a1, b1): (%d, %d)\n", a1,b1);

				utility::histo_stretch(src, tgt, a1, b1, ROI_parameters);
			}

	        /* thresh_histo_stretch ============================================*/
	        else if (strncasecmp(function_name,"thresh_histo_stretch",MAXLEN)==0) 
	        {

	        	pch = strtok(NULL, " ");
				int T = atoi(pch);
				pch = strtok(NULL, " ");
				int a1 = atoi(pch);
				pch = strtok(NULL, " ");
				int b1 = atoi(pch);
				pch = strtok(NULL, " ");
				int a2 = atoi(pch);
				pch = strtok(NULL, " ");
				int b2 = atoi(pch);

				if (debug) printf("thresh_histo_stretch (T, a1, b1, a2, b2): (%d, %d, %d, %d, %d)\n", T, a1, b1, a2, b2);

				utility::thresh_histo_stretch(src,tgt, T, a1, b1, a2, b2, ROI_parameters);
			}

	        /* stretch_RGB_single ============================================*/
	        else if (strncasecmp(function_name,"stretch_RGB_single",MAXLEN)==0) 
	        {

				pch = strtok(NULL, " ");
				int channel = atoi(pch);
	        	pch = strtok(NULL, " ");
				int a1 = atoi(pch);
				pch = strtok(NULL, " ");
				int b1 = atoi(pch);

				if (debug) printf("stretch_RGB_single (channel, a1, b1): (%d, %d, %d)\n", channel, a1, b1);

				utility::histo_stretch_RGB_single(src, tgt, a1, b1, channel, ROI_parameters);
			}

	        /* stretch_RGB_multi ============================================*/
	        else if (strncasecmp(function_name,"stretch_RGB_multi",MAXLEN)==0) 
	        {

				pch = strtok(NULL, " ");
				int aR = atoi(pch);
				pch = strtok(NULL, " ");
				int bR = atoi(pch);
				pch = strtok(NULL, " ");
				int aG = atoi(pch);
				pch = strtok(NULL, " ");
				int bG = atoi(pch);
				pch = strtok(NULL, " ");
				int aB = atoi(pch);
				pch = strtok(NULL, " ");
				int bB = atoi(pch);

				if (debug) printf("thresh_histo_stretch (aR, bR, aG, bG, aB, bB): (%d, %d, %d, %d, %d, %d)\n", aR, bR, aG, bG, aB, bB);

				utility::histo_stretch_RGB_multi(src,tgt, aR, bR, aG, bG, aB, bB, ROI_parameters);
			}

	        /* stretch_I ============================================*/
	        else if (strncasecmp(function_name,"stretch_I",MAXLEN)==0) 
	        {

	        	pch = strtok(NULL, " ");
				double a1 = atof(pch);
				pch = strtok(NULL, " ");
				double b1 = atof(pch);

				if (debug) printf("stretch_I (a1, b1): (%f, %f)\n", a1, b1);

				utility::histo_stretch_I(src, tgt, a1, b1, ROI_parameters);
			}

	        /* stretch_HSI ============================================*/
	        else if (strncasecmp(function_name,"stretch_HSI",MAXLEN)==0) 
	        {

				pch = strtok(NULL, " ");
				double aH = atof(pch);
				pch = strtok(NULL, " ");
				double bH = atof(pch);
				pch = strtok(NULL, " ");
				double aS = atof(pch);
				pch = strtok(NULL, " ");
				double bS = atof(pch);
				pch = strtok(NULL, " ");
				double aI = atof(pch);
				pch = strtok(NULL, " ");
				double bI = atof(pch);

				if (debug) printf("thresh_histo_stretch (aH, bH, aS, bS, aI, bI): (%f, %f, %f, %f, %f, %f)\n", aH, bH, aS, bS, aI, bI);

				utility::histo_stretch_HSI(src,tgt, aH, bH, aS, bS, aI, bI, ROI_parameters);
			}

			else {
				printf("No function: %s\n", pch);
				continue;
			}

			if(debug)
			{
				printf("final size: %d %d, tgt size: %d %d\n", final.getNumberOfRows(), final.getNumberOfColumns(), tgt.getNumberOfRows(), tgt.getNumberOfColumns());
			}

			int count = 0;
			// copy modified pixels to final image
			for (int row = ROI_parameters.Y; row < ROI_parameters.Y + ROI_parameters.Sy; ++row)
				for (int col = ROI_parameters.X; col < ROI_parameters.X + ROI_parameters.Sx; ++col)
					{
						if(final.isInbounds(row, col) && tgt.isInbounds(row, col))
						{
							++count;
							final.setPixel(row, col, RED, tgt.getPixel(row, col, RED));
							final.setPixel(row, col, GREEN, tgt.getPixel(row, col, GREEN));
							final.setPixel(row, col, BLUE, tgt.getPixel(row, col, BLUE));
						}
					}

			if (debug) printf("set %d pixels in ROI %d\n", count, idxROI);
		}

       
		final.save(outfile);
	}
	fclose(fp);
	return 0;
}

