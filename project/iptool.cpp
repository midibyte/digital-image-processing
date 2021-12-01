/************************************************************

 * 	check included readme file for descriptions of functions and parameters

 *															*
 ************************************************************/

#include "../iptools/core.h"
#include <strings.h>
#include <string.h>
#include <chrono>

using namespace std;

// maxlen of strings
#define MAXLEN 4096

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
	int count_ROI;
	// debug options value
	int debug = 0;

	// used with edge detect
	image binaryEdge, binaryEdgeAngle;

	//used for timing functions
	// adaoted from stackoverflow example
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;



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
		
		// skip comment lines and blank lines in parameters file
		if (str[0] == '#' || str[0] == '\n')
			continue;

		if ( debug > 0) printf("\n");
		if ( debug > 0) printf("parameters-> %s", str);

		// get source image file name
		pch = strtok(str, " ");
		strcpy(infile, pch);
		src.read(pch);

		if (debug == 1) printf("source image: %s\n", pch);
		if (debug == 1) printf("img size X, Y: %d, %d\n", src.getNumberOfColumns(), src.getNumberOfRows());

		// get destination file name
		pch = strtok(NULL, " ");
		strcpy(outfile, pch);

		if (debug == 1) printf("dest image: %s\n", outfile);

		// read in function - then parameters
		pch = strtok(NULL, " ");
		//store function name
		strcpy(function_name, pch);

		if (debug == 1) printf("function_name: %s\n", function_name);


		// get number of ROIs
		pch = strtok(NULL, " ");
		count_ROI = atoi(pch);

		if ( debug == 1) printf("number of ROIs %d\n", count_ROI);

		// setup image to record modified pixels
		wasModified.resize(src.getNumberOfRows(), src.getNumberOfColumns());
		// setup final output image - copy of source, overwrite ROI areas with new pixels
		image final(src);

		// for measuring performance
		auto start_time = high_resolution_clock::now();


		// handle ROIs =========================================================
		// do the selected function for each ROI
		// at the end of each ROI, add to final image
		for (int idxROI = 0; idxROI < count_ROI; ++idxROI )
		{
			// struct holds ROI options
			struct ROI ROI_parameters;

			strcpy(ROI_parameters.inputImageName, infile);

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

			//get the base name of the image w/o extension
			hName[strlen(hName) - 4] = '\0';
			sprintf(ROI_parameters.ogImageName, "%s", hName);
			sprintf(ROI_parameters.histogramName, "%.1024s_ROI_%d_HISTOGRAM.pgm",hName, idxROI );

			if (debug == 1) printf("Sx, Sy, X, Y: %d, %d, %d, %d\n", ROI_parameters.Sx, ROI_parameters.Sy, ROI_parameters.X, ROI_parameters.Y );

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
			// check for ROI overlap with previous ROI, skip if overlapping
			for (int row = ROI_parameters.Y; row < ROI_parameters.Y + ROI_parameters.Sy; ++row)
			for (int col = ROI_parameters.X; col < ROI_parameters.X + ROI_parameters.Sx; ++col)
			if (wasModified.isInbounds(row, col))
			if(wasModified.getPixel(row, col) > 0)
			{
				// overlaps with previous ROI, skip
				printf("ROI# %d overlaps with a previous ROI, skipping...\n", idxROI);
				ROI_overlap = true;
				goto overlap;
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
				if (debug == 1) printf("value: %d\n", atoi(pch));
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

				if (debug == 1) printf("in iptool.cpp: T, V1, V2: %d %d %d\n", T, V1, V2);

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
				if (debug == 1) printf("more-C value: %f\n", atof(pch));
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

				if (debug == 1) printf("histo_stretch (a1, b1): (%d, %d)\n", a1,b1);

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

				if (debug == 1) printf("thresh_histo_stretch (T, a1, b1, a2, b2): (%d, %d, %d, %d, %d)\n", T, a1, b1, a2, b2);

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

				if (debug == 1) printf("stretch_RGB_single (channel, a1, b1): (%d, %d, %d)\n", channel, a1, b1);

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

				if (debug == 1) printf("thresh_histo_stretch (aR, bR, aG, bG, aB, bB): (%d, %d, %d, %d, %d, %d)\n", aR, bR, aG, bG, aB, bB);

				utility::histo_stretch_RGB_multi(src,tgt, aR, bR, aG, bG, aB, bB, ROI_parameters);
			}

	        /* stretch_I ============================================*/
	        else if (strncasecmp(function_name,"stretch_I",MAXLEN)==0) 
	        {

	        	pch = strtok(NULL, " ");
				double a1 = atof(pch);
				pch = strtok(NULL, " ");
				double b1 = atof(pch);

				if (debug == 1) printf("stretch_I (a1, b1): (%f, %f)\n", a1, b1);

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

				if (debug == 1) printf("thresh_histo_stretch (aH, bH, aS, bS, aI, bI): (%f, %f, %f, %f, %f, %f)\n", aH, bH, aS, bS, aI, bI);

				utility::histo_stretch_HSI(src,tgt, aH, bH, aS, bS, aI, bI, ROI_parameters);
			}

			/* edge detect ============================================*/
	        else if (strncasecmp(function_name,"edge_detect",MAXLEN)==0) 
	        {

				pch = strtok(NULL, " ");
				int kernel_size = atoi(pch);
				pch = strtok(NULL, " ");

				bool isColor = false;

				// check input -- if color image, set the flag
				if (strstr(infile, ".ppm") != NULL) 
				{	/* PPM Color Image */
					isColor = true;
				}

				if (debug == 1) printf("edge_detect (kernel_size, T, angle, color?): (%d, %s)\n", kernel_size, isColor?"true":"false");

				utility::edge_detect(src,tgt, kernel_size, isColor, ROI_parameters);
			}

			else if (strncasecmp(function_name,"edge_detect_binary",MAXLEN)==0) 
	        {

				pch = strtok(NULL, " ");
				int kernel_size = atoi(pch);
				pch = strtok(NULL, " ");
				int T = atoi(pch);
				pch = strtok(NULL, " ");
				int angle = atoi(pch);

				bool isColor = false;

				// check input -- if color image, set the flag
				if (strstr(infile, ".ppm") != NULL) 
				{	/* PPM Color Image */
					isColor = true;
				}

				if (debug == 1) printf("edge_detect_binary (kernel_size, T, angle, color?): (%d, %d, %d, %s)\n", kernel_size, T, angle, isColor?"true":"false");

				utility::edge_detect_binary(src,tgt, kernel_size, T, angle, isColor, ROI_parameters);
			}

			else if (strncasecmp(function_name,"histogram",MAXLEN)==0) 
	        {

				bool isColor = false;

				// check input -- if color image, set the flag
				if (strstr(infile, ".ppm") != NULL) 
				{	/* PPM Color Image */
					isColor = true;
				}

				if (debug == 1) printf("histogram (color?): (%s)\n", isColor?"true":"false");

				utility::make_histogram_image(src, tgt, isColor, ROI_parameters);
			}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++ openCV fuctions

			// SOBEL
			else if (strncasecmp(function_name,"sobel_opencv",MAXLEN)==0) 
	        {

				pch = strtok(NULL, " ");
				int kernel_size = atoi(pch);
				pch = strtok(NULL, " ");
				int T = atoi(pch);
				pch = strtok(NULL, " ");
				int angle = atoi(pch);

				bool isColor = false;

				// check input -- if color image, set the flag
				if (strstr(infile, ".ppm") != NULL) 
				{	/* PPM Color Image */
					isColor = true;
				}

				if (debug == 1) printf("sobel_opencv (kernel_size, color?): (%d, %s)\n", kernel_size, isColor?"true":"false");

				utility::sobel_opencv(src, tgt, T, angle, kernel_size, isColor, ROI_parameters);
			}
			//  CANNY
			else if (strncasecmp(function_name,"canny_opencv",MAXLEN)==0) 
	        {

				pch = strtok(NULL, " ");
				int kernel_size = atoi(pch);
				pch = strtok(NULL, " ");
				int T = atoi(pch);
				pch = strtok(NULL, " ");
				int angle = atoi(pch);

				bool isColor = false;

				// check input -- if color image, set the flag
				if (strstr(infile, ".ppm") != NULL) 
				{	/* PPM Color Image */
					isColor = true;
				}

				if (debug == 1) printf("canny_opencv (kernel_size, color?): (%d, %s)\n", kernel_size, isColor?"true":"false");

				utility::canny_opencv(src, tgt, T, angle, kernel_size, isColor, ROI_parameters);
			}

			// OTSU
			else if (strncasecmp(function_name,"otsu_opencv",MAXLEN)==0) 
	        {

				bool isColor = false;

				// check input -- if color image, set the flag
				if (strstr(infile, ".ppm") != NULL) 
				{	/* PPM Color Image */
					isColor = true;
				}

				if (debug == 1) printf("otsu_opencv (color?): (%s)\n", isColor?"true":"false");

				utility::otsu_opencv(src, tgt, isColor, ROI_parameters);
			}

			else if (strncasecmp(function_name,"equalize_foreground_otsu_opencv",MAXLEN)==0) 
	        {

				bool isColor = false;

				// check input -- if color image, set the flag
				if (strstr(infile, ".ppm") != NULL) 
				{	/* PPM Color Image */
					isColor = true;
				}

				if (debug == 1) printf("equalize_foreground_otsu_opencv (color?): (%s)\n", isColor?"true":"false");

				utility::equalize_foreground_otsu_opencv(src, tgt, isColor, ROI_parameters);
			}

			else if (strncasecmp(function_name,"equalize_opencv",MAXLEN)==0) 
	        {

				bool isColor = false;

				// check input -- if color image, set the flag
				if (strstr(infile, ".ppm") != NULL) 
				{	/* PPM Color Image */
					isColor = true;
				}

				if (debug == 1) printf("equalize_opencv (color?): (%s)\n", isColor?"true":"false");

				utility::equalize_opencv(src, tgt, isColor, ROI_parameters);
			}

			//  DFT ======================================================================================
			else if (strncasecmp(function_name,"DFT",MAXLEN)==0) 
	        {
				bool isColor = false;

				// check input -- if color image, set the flag
				if (strstr(infile, ".ppm") != NULL) 
				{	/* PPM Color Image */
					isColor = true;
				}

				if (debug == 1) printf("DFT (color?): (%s)\n", isColor?"true":"false");

				utility::DFT(src, tgt, isColor, ROI_parameters);
			}

			//  INVERSE DFT ======================================================================================
			else if (strncasecmp(function_name,"IDFT",MAXLEN)==0) 
	        {
				bool isColor = false;

				// check input -- if color image, set the flag
				if (strstr(infile, ".ppm") != NULL) 
				{	/* PPM Color Image */
					isColor = true;
				}

				if (debug == 1) printf("INVERSE DFT (color?): (%s)\n", isColor?"true":"false");

				utility::IDFT(src, tgt, isColor, ROI_parameters);
			}

			//  FILTERING ======================================================================================
			else if (strncasecmp(function_name,"low_pass",MAXLEN)==0 || strncasecmp(function_name,"high_pass",MAXLEN)==0) 
	        {
				pch = strtok(NULL, " ");
				ROI_parameters.filter_radius = atof(pch);

				ROI_parameters.H_filter = 0;
				ROI_parameters.V_filter = 0;
				int type;
				
				
				if(strncasecmp(function_name,"high_pass",MAXLEN)==0)
				{
					ROI_parameters.low_pass = 0;
					ROI_parameters.high_pass = 1;
				}
				else
				{
					ROI_parameters.low_pass = 1;
					ROI_parameters.high_pass = 0;
				}
				bool isColor = false;

				// check input -- if color image, set the flag
				if (strstr(infile, ".ppm") != NULL) 
				{	/* PPM Color Image */
					isColor = true;

					pch = strtok(NULL, " ");
				
					// filter type 0 = normal, 1 = H channel on color image, 2 = V channel on color image
					type = atoi(pch);

					if (type == 1) 
					{
						ROI_parameters.H_filter = 1;
						ROI_parameters.V_filter = 0;
					}
					if (type == 2)
					{
						ROI_parameters.H_filter = 0;
						ROI_parameters.V_filter = 1;
					}

				}

				if (debug == 1) printf("filter in (color?): (%s)\t(low_pass?): (%s)\t(high pass?): (%s)\n", isColor?"true":"false", ROI_parameters.low_pass?"true":"false", ROI_parameters.high_pass?"true":"false");
				if (debug == 1 && isColor) printf(" color mode: %d\n", type);
				utility::dft_filter(src, tgt, isColor, ROI_parameters);
			}

			// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

			// band pass
			else if (strncasecmp(function_name,"band_pass",MAXLEN)==0) 
	        {
				pch = strtok(NULL, " ");
				ROI_parameters.filter_radius = atof(pch);
				pch = strtok(NULL, " ");
				ROI_parameters.filter_radius_2 = atof(pch);
				ROI_parameters.band_pass = 1;
				ROI_parameters.low_pass = 0;
				ROI_parameters.high_pass = 0;

				ROI_parameters.H_filter = 0;
				ROI_parameters.V_filter = 0;

				bool isColor = false;

				// check input -- if color image, set the flag
				if (strstr(infile, ".ppm") != NULL) 
				{	/* PPM Color Image */
					isColor = true;

					pch = strtok(NULL, " ");
				
					// filter type 0 = normal, 1 = H channel on color image, 2 = V channel on color image
					int type = atoi(pch);

					if (type == 1) 
					{
						ROI_parameters.H_filter = 1;
						ROI_parameters.V_filter = 0;
					}
					if (type == 2)
					{
						ROI_parameters.H_filter = 0;
						ROI_parameters.V_filter = 1;
					}

				}

				if (debug == 1) printf("band-pass in (color?): (%s)\n", isColor?"true":"false");

				utility::dft_filter(src, tgt, isColor, ROI_parameters);
			}

			//  UNSHARP MASK ======================================================================================
			else if (strncasecmp(function_name,"unsharp_mask",MAXLEN)==0) 
	        {	
				//set all options to zeros
				ROI_init_options(ROI_parameters);
				
				pch = strtok(NULL, " ");
				ROI_parameters.filter_radius = atof(pch);
				pch = strtok(NULL, " ");
				ROI_parameters.unsharp_mask_amount = atof(pch);
				
				
				bool isColor = false;

				// check input -- if color image, set the flag
				if (strstr(infile, ".ppm") != NULL) 
				{	/* PPM Color Image */
					isColor = true;
				}

				if (debug == 1) printf("function:\tunsharp_mask\tradius:\t%f\tamount:\t%f", ROI_parameters.filter_radius, ROI_parameters.unsharp_mask_amount);
				utility::unsharp_mask(src, tgt, isColor, ROI_parameters);
			}

			// END OF FUNCTION SELECTION =================================================
			else {
				printf("No function: %s\n", pch);
				continue;
			}

			if(debug == 1)
			{
				printf("final size: %d %d, tgt size: %d %d\n", final.getNumberOfRows(), final.getNumberOfColumns(), tgt.getNumberOfRows(), tgt.getNumberOfColumns());
			}

			int count = 0;
			// copy modified pixels to final image
			for (int row = ROI_parameters.Y; row < ROI_parameters.Y + ROI_parameters.Sy; ++row)
			for (int col = ROI_parameters.X; col < ROI_parameters.X + ROI_parameters.Sx; ++col)
			if(final.isInbounds(row, col) && tgt.isInbounds(row, col))
			{
				++count;
				final.setPixel(row, col, RED, tgt.getPixel(row, col, RED));
				final.setPixel(row, col, GREEN, tgt.getPixel(row, col, GREEN));
				final.setPixel(row, col, BLUE, tgt.getPixel(row, col, BLUE));

			}

			if (debug == 1) printf("set %d pixels in ROI %d\n", count, idxROI);
		}

		//finished all processing

		auto end_time = high_resolution_clock::now();

		/* Getting number of milliseconds as an integer. */
		auto ms_int = duration_cast<milliseconds>(end_time - start_time);

		/* Getting number of milliseconds as a double. */
		duration<double, std::milli> ms_double = end_time - start_time;

		if (debug > 1)
		{
			std::cout << "\n";
			std::cout << "Performance of function: " << function_name << "\n";
			std::cout << ms_int.count() << "ms\n";
			std::cout << ms_double.count() << "ms\n";
			std::cout << "****************************************************************************************************\n";

		}

		if(!(strncasecmp(function_name,"DFT",MAXLEN)==0))
			final.save(outfile);

	}
	fclose(fp);
	return 0;
}

