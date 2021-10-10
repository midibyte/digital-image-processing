This software is architectured as follows. This software can work on grad server.

iptools -This folder hosts the files that are compiled into a static library. 
	image - This folder hosts the files that define an image.
	utility- this folder hosts the files that students store their implemented algorithms.
	
lib- This folder hosts the static libraries associated with this software.

project- This folder hosts the files that will be compiled into executables.
	bin- This folder hosts the binary executables created in the project directory.



*** INSTALATION ***

On Linux

Enter the project directory in terminal and run make

As a result you should get iptool in project/bin directory.

*** FUNCTIONS ***

Each function operates only in its ROI

1. Add intensity: add
Increase the intensity for a gray-level image.

2. Binarization: binarize
Binarize the pixels with the threshold.

3. Scaling: Scale
Reduce or expand the heigh and width with two scale factors.
Scaling factor = 2: double height and width of the input image.
Scaling factor = 0.5: half height and width of the input image.

Project 0:

4. Dual threshold: dual_threshold
	
	inputs: threshold T, Values V1, V2
	if pixel intensity:
		> T, set pixel to V1
		< T, set pixel to V2
		== T, do nothing

Project 1:

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

PROJECT 2

Histogram image creation added

Histgram stretching

Histogram stretching with thresholding

Histogram stretching on one channel of RGB

Histogram stretching on all RGB

RGB to HSI

HSI to RGB 

Histogram stretching on I-component of HSI 
	display I as a grey level image 
	convert back to RGB and display color after stretched I channel

Histogram stretching on H and S and HSI separately 

*** PARAMETERS FILE ***

Each line in the parameters.txt file follows this format:

infile outfile function ROI_count ROI_parameters function_parameters (one set of ROI_parameters and function_parameters for each ROI)

Function names and parameters:

add value
binarize threhold
scale value
dual_threshold T V1 V2
uniformSmoothing window_size
colorMultiplicativeBrightness C_value
colorBinarize TColor CR CG CB

parameters.txt line example
miami.ppm miami_coloradd_3ROI.ppm colorMultiplicativeBrightness 3 300 300 100 100 1.75 600 80 500 500 0.35 800 125 10 800 2.0

the line above is formatted like this: infile outfile funtion ROI_count ROI1(Sx, Sy, X, Y, C_value) ROI2(Sx, Sy, X, Y, C_value) ROI3(Sx, Sy, X, Y, C_value)

*** Run the program: ./iptool parameters.txt
can use: 
./iptool parameters.txt 1
to print debug strings to console

PROGRAM LOGIC

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


