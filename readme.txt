This software is architectured as follows:

iptools -This folder hosts the files that are compiled into a static library. 
	image - This folder hosts the files that define an image.
	utility- this folder hosts the files that students store their implemented algorithms.
	
lib- This folder hosts the static libraries associated with this software.

project- This folder hosts the files that will be compiled into executables.
	bin- This folder hosts the binary executables created in the project directory.



*** INSTALATION *********************************************************************

	On Linux

	Enter the project directory in terminal and run make

	As a result you should get iptool in project/bin directory.

	If there are any errors when running make, run make clean, then make again
	
	NOTES: 
		the project/bin folder must exist for make to succeed
		the project/bin/result folder must exist, unless
	

*** FUNCTIONS *********************************************************************

	Each function operates only in its ROI
	For each function, the function name is listed first, followed by the required parameters for that function
	A histogram can be created for any image. Use the histogram function described under Projcet 3.

	add value
		Increase the intensity for a gray-level image.

	binarize T
		Binarize the pixels with the threshold.

	Scale f
		Reduce or expand the heigh and width with two scale factors.
		Scaling factor = 2: double height and width of the input image.
		Scaling factor = 0.5: half height and width of the input image.

PROJECT 0: ******************************************************************

	dual_threshold T V1 V2
		inputs: 
			threshold T, Values V1, V2
			V1 < V2 
			all inputs are in the range [0 255]
		function logic:
		if pixel intensity:
			> T, set pixel to V1
			< T, set pixel to V2
			== T, do nothing

PROJECT 1: ******************************************************************

	uniformSmoothing window_size
		perform uniform smoothing with a window size of "window_size"
		uses an odd numbered square window size only

	colorMultiplicativeBrightness C_value
		for each pixel, multiply the pixel RGB values each by C_value 

	colorBinarize TColor CR CG CB
		TColor is the Euclidean distance from each pixel to the user defined color: C(CR CG CB)
		For each pixel, if the pixel is from  0 to TColor distance to C, set the pixel to "red" (255, 0, 0). 
		If the pixel is between TColor and 2*TColor distance to C, set the pixel to "green" (0, 255, 0).
		Set all other pixels to "white" (255, 255, 255).

PROJECT 2 ******************************************************************

	histo_stretch a b 
		transforms the values inside the range [a b] to [0 255]
		outputs a histogram both before and after the transformation

	thresh_histo_stretch T a1 b1 a2 b2
		transforms the values :
			less than T:
				inside the range [a1 b1] to [0 255]
			otherwise:
				inside the range [a2 b2] to [0 255]
		outputs a histogram both before and after the transformation
		

	stretch_RGB_single channel a b
		transforms the values in an RGB image in "channel" inside the range [a b] to [0 255]
		channel is a number 0, 1, 2 representing the channels R, G, B

	stretch_RGB_multi aR bR aB bB aG bG
		transforms the values in the RED channel inside the range [aR bR] to [0 255]
		transforms the values in the GREEN channel inside the range [aG bG] to [0 255]
		transforms the values in the BLUE channel inside the range [aB bB] to [0 255]
		NOTE: can set any of the a b values to a=0 and b=1.0 to ignore that channel 

	stretch_I a b
		converts the image from the RGB color space to HSI 
		outputs a histogram image of the I channel 
		outputs a gray level image representing the I channel 
		using the I channel, transforms the values inside the range [a b] to [0 1.0]
		outputs a histogram of the new I channel
		outputs a gray level image of the representing the new I channel
		converts the image with the new I channel from HSI back to RGB then outputs the new image

	stretch_HSI aH bH aS bS aI bI
		transforms the values in the HUE channel inside the range [aH bH] to [0 360]
		transforms the values in the SATURATION channel inside the range [aS bS] to [0 1.0]
		transforms the values in the INTENSITY channel inside the range [aI bI] to [0 1.0]
		NOTE: can set any of the a b values to a=0 and b=1.0 to ignore that channel 

	Histogram stretching on I-component of HSI 
		display I as a grey level image 
		convert back to RGB and display color after stretched I channel

PROJECT 3 *******************************************************************************

	histogram
		takes no input 
		input and output file names should be the same
		outputs a histogram for each ROI 
		output name will include ROI_#_HISTOGRAM, where # is the ROI number
		NOTE: use this function to create a histogram of any image and ROIs

	edge_detect kernel_size
		NOTE: kernel_size can ONLY be 3 or 5
		uses the sobel filter of size "kernel_size"
		outputs gradient image only
		gradient is the combined Gx Gy gradients

	edge_detect_binary kernel_size threshold angle
		applies the sobel filter of size "kernel_size" to the input image
		input ranges:
			threshold T range [0 255]
			angle range [0 360]
			kernel_size MUST be 3 or 5
		angle displays edges within +-10 degrees of input
		computes: dx, dy, gradient amplitude, edge direction
		kernel_size can ONLY be 3 or 5
		IF a color image is used, the images is converted to HSI and the sobel filter is applied to the I channel
		the resulting ROIs will be grayscale
		output options:
			T = -1 angle = -1
				outputs gradient image 
			T = value, angle = -1
				outputs gradient image with binarizaiton with threshold T
			T = value, angle = value
				binary gradient image using threshold T, shows only angles +- 10 of input angle value
			NOTE: only one output image is produced. 
				Use different combinations of inputs to get desired output
				
PROJECT 4 *******************************************************************************
	NOTES: 
		in all cases the circular filter diameter in pixels = filter_size * longest_side_in_pixels


	DFT
		outputs:
			an image of the magnitude of the fourier transform of the image

	low_pass filter_size
		outputs:
			an image of the magnitude of the fourier transform of the image
			the filtered image
			an image of the magnitude of the fourier transform of the filtered image
		inputs:
			filter_size: 
				a floating point value. range [0.0,  1.0]
				this will determine the diameter of the circular filter to apply to the image in the fourier domain
				the circle will be located at the center of the image, with a diameter = pixels*filter_size
				where pixels is equal to the number of rows or columns of the input image, whichever is larger
				the filter will be white in the middle and black outside the circle. 
				it is used as a mask on the input image in the frequency domain, only letting the pixels that overlap with white pass through

	high_pass filter_size
		outputs:
			an image of the magnitude of the fourier transform of the image
			the filtered image
			an image of the magnitude of the fourier transform of the filtered image
		inputs:
			filter_size: 
				a floating point value. range [0.0,  1.0]
				this will determine the diameter of the circular filter to apply to the image in the fourier domain
				the circle will be located at the center of the image, with a diameter = pixels*filter_size
				where pixels is equal to the number of rows or columns of the input image, whichever is larger
				the filter will be black in the middle and whilte outside the circle. 
				it is used as a mask on the input image in the frequency domain, only letting the pixels that overlap with white pass through

	band_pass filter_size1 filter_size2
		outputs:
			an image of the magnitude of the fourier transform of the image
			the filtered image
			an image of the magnitude of the fourier transform of the filtered image
		inputs:
			filter_size1 and filter_size2:
				NOTE: filter_size2 MUST be < filter_size1 
				a floating point value. range [0.0,  1.0]
				this will determine the diameter of the circular filter to apply to the image in the fourier domain
				the circle will be located at the center of the image, with a diameter = pixels*filter_size
				where pixels is equal to the number of rows or columns of the input image, whichever is larger
				creates a band pass filter where filter_size2 is the size of teh inner circle (black),
				and filter_size1 is the size fo the outer circle (white)
				This creates a filter where the space between the two circles is white, making a band-pass filter in the frequency domain
	

	unsharp_mask filter_size amount
		applies a high_pass filter to the image, 
		then multiplies the high frequency data by amount and adds it back to the original image
		
		outputs:
			an image of the magnitude of the fourier transform of the image
			the filtered image
			an image of the magnitude of the fourier transform of the filtered image

	COLOR FUNCTIONS:

	high_pass filter_size type
		same as grayscale version but only for color images
		type is the type of filtering -> 1 = filter H component, 2 = filter V component

	low_pass 
		same as grayscale version but only for color images
		type is the type of filtering -> 1 = filter H component, 2 = filter V component
	
	band_pass
		same as grayscale version but only for color images
		type is the type of filtering -> 1 = filter H component, 2 = filter V component

		
	OpenCV Functions ***************************************************************
	
	sobel_opencv kernel_size T angle
		uses the sobel filter for edge detection
		input ranges:
			threshold T range [0 255]
			angle range [0 360]
			kernel_size MUST be 3 or 5
		IF a color image is used, the images is converted to HSI and the sobel filter is applied to the I channel
		the resulting ROIs will be grayscale
		output options:
			T = -1 angle = -1
				outputs gradient image 
			T = value, angle = -1
				outputs gradient image with binarizaiton with threshold T
			T = value, angle = value
				binary gradient image using threshold T, shows only angles +- 10 of input angle value
			NOTE: only one output image is produced. 
				Use different combinations of inputs to get desired output
		
		
	canny_opencv kernel_size T angle
		uses the canny filter for edge detection
		input ranges:
			threshold T range [0 255]
			angle range [0 360]
			kernel_size MUST be 3 or 5
		IF a color image is used, the images is converted to HSI and the sobel filter is applied to the I channel
		the resulting ROIs will be grayscale
		output options:
			T = -1 angle = -1
				outputs gradient image 
			T = value, angle = -1
				outputs gradient image with binarizaiton with threshold T
			T = value, angle = value
				binary gradient image using threshold T, shows only angles +- 10 of input angle value
			NOTE: only one output image is produced. 
				Use different combinations of inputs to get desired output
	
	otsu_opencv
		takes no input 
		uses the opencv threshold function with the otsu binary option 
		outputs a binary image
	
	equalize_opencv
		takes no inputs
		uses the opencv function equalizeHist to equalize the ROIs 
		outputs an equalized image

	equalize_foreground_otsu_opencv
		takes no input 
		uses otsu to find the threshold value
		applies the opencv histogram equalization function to the ROIs
		then uses the following logic:
			if a pixel is >= threshold, set pixel to equalized value 
			else, set to original value 

		 

*** PARAMETERS FILE *********************************************************************

	Each line in the parameters.txt file follows this format:

	infile outfile function ROI_count ROI_parameters function_parameters (one set of ROI_parameters and function_parameters for each ROI)

	the function names are specified in the Functions section above

	parameters.txt line example
	miami.ppm miami_coloradd_3ROI.ppm colorMultiplicativeBrightness 3 300 300 100 100 1.75 600 80 500 500 0.35 800 125 10 800 2.0

	the line above is formatted like this: infile outfile funtion ROI_count ROI1(Sx, Sy, X, Y, C_value) ROI2(Sx, Sy, X, Y, C_value) ROI3(Sx, Sy, X, Y, C_value)


RUNNING THE PROGRAM *********************************************************************

	Run the program with: ./iptool parameters.txt
	where parameters.txt can be any name corresponding to your parameters file
	NOTE:
		can use: 
		./iptool parameters.txt debug_val
		debug_val = 
			1 = print debug strings to console
			2 = use print time measurements
	NOTE:
		if you are getting segmentation fault errors, check to see if you included a function for each line in the parameters file 
		.pgm and .ppm files must be in binary format, not ascii format 

PROGRAM LOGIC *********************************************************************

	load parameters file
	get parameters from a single line in the file and read in input file

	enter for loop to handle ROIs
		get parameters for a single ROI
			if an ROI overlaps with a previous ROI, skip it
			make sure top left ROI coordinates are within image bounds
				if not, skip this ROI
				if the ROI size causes some ROI pixels to fall outside the image, still process ROI but ignore these pixels
		mark pixels within ROI as modified
		do function
		copy modified pixels to final image


