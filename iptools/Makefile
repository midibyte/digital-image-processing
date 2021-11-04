# source files.
SRC =  image/image.cpp utility/utility.cpp utility/opencv.cpp

OBJ = $(SRC:.cpp=.o)

OUT = ../lib/libiptools.a

#opencv library
OPENCVLIBS := -L /apps/opencv/4.0.1/lib64 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -Wl,-rpath=/apps/opencv/4.0.1/lib64
# this line for MacOS
# OPENCVLIBS := -L /usr/local/Cellar/opencv/4.5.3_2/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect -lopencv_imgproc
# this line for server
# OPENCVLIBS := -L /usr/include/opencv2 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect -lopencv_imgproc


# include opencv directory
OPENCVINCLUDE := -I/apps/opencv/4.0.1/include/opencv4
# macos
# OPENCVINCLUDE:= -I/usr/local/Cellar/opencv/4.5.3_2/include/opencv4
# server
# OPENCVINCLUDE := -I/usr/include/opencv2

# include directories
INCLUDES = -I. -I/usr/local/include -Iimage/  -Iutility/ -I/usr/include/c++/4.2.1 

# C++ compiler flags (-g -O2 -Wall)
CCFLAGS = -std=c++11 -g

# compiler
CCC = g++


# compile flags
LDFLAGS = -g

.SUFFIXES: .cpp

.cpp.o:
	$(CCC) $(OPENCVLIBS) $(OPENCVINCLUDE) $(INCLUDES) $(CCFLAGS) -c $< -o $@

$(OUT): $(OBJ)
	ar rcs $(OUT) $(OBJ)

clean:
	rm -f $(OBJ) $(OUT) Makefile.bak# DO NOT DELETE
