# source files.
SRC =  image/image.cpp utility/utility.cpp utility/opencv.cpp

OBJ = $(SRC:.cpp=.o)

OUT = ../lib/libiptools.a

#opencv library
# SERVER
# OPENCVLIBS := -L /apps/opencv/4.0.1/lib64 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -Wl,-rpath=/apps/opencv/4.0.1/lib64
# this line for MacOS
# OPENCVLIBS := -L /usr/local/Cellar/opencv/4.5.3_2/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect -lopencv_imgproc
# popos
# OPENCVLIBS := -L /usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -Wl,-rpath,/usr/include/opencv4
# OPENCVLIBS := -L/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
OPENCVLIBS := `pkg-config --libs opencv4`



# include opencv directory
# SERVER
# OPENCVINCLUDE := -I/apps/opencv/4.0.1/include/opencv4
# macos
# OPENCVINCLUDE:= -I/usr/local/Cellar/opencv/4.5.3_2/include/opencv4
# popos
# OPENCVINCLUDE := -I/usr/include/opencv4
OPENCVINCLUDE := `pkg-config --cflags opencv4`


# include directories
INCLUDES = -I. -I/usr/local/include -Iimage/  -Iutility/ -I/usr/include/c++/4.2.1 

# C++ compiler flags (-g -O2 -Wall)
CCFLAGS = -std=c++11 -g

# compiler
CCC = g++

.SUFFIXES: .cpp

.cpp.o:
	$(CCC) $(CCFLAGS) -c $< -o $@ $(OPENCVINCLUDE) $(INCLUDES) $(OPENCVLIBS)

$(OUT): $(OBJ)
	ar rcs $(OUT) $(OBJ)

clean:
	rm -f $(OBJ) $(OUT) Makefile.bak# DO NOT DELETE
