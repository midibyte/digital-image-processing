
BIN_DIR=bin/
EXEC = iptool
CCFLAGS = -std=c++11 -g
CCFLAGSDEBUG := -std=c++11 -g -Wall -pedantic -Wextra

#opencv library
# SERVER
# OPENCVLIBS := -L /apps/opencv/4.0.1/lib64 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -Wl,-rpath=/apps/opencv/4.0.1/lib64
# macos
# OPENCVLIBS := -L /usr/local/Cellar/opencv/4.5.3_2/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect -lopencv_imgproc
# popos
# OPENCVLIBS := -L/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

# automatic libs with pkg-config
OPENCVLIBS := `pkg-config --libs opencv4`
# include precompiled libs from other makefile
LIBS := -L../lib -l iptools $(OPENCVLIBS)

# include opencv directory
# SERVER
# OPENCVINCLUDE := -I/apps/opencv/4.0.1/include/opencv4
# macos
# OPENCVINCLUDE:= -I/usr/local/Cellar/opencv/4.5.3_2/include/opencv4
# popos
# OPENCVINCLUDE := -I/usr/include/opencv4

# automatic include directorty with pkg-config
OPENCVINCLUDE := `pkg-config --cflags opencv4`

# INCLUDES = -I

all: $(EXEC)

test: 
	(cd ../iptools; make)
	g++  $(CCFLAGS) -o $(BIN_DIR)$@ $@.cpp $(OPENCVINCLUDE) $(LIBS)

iptools:
	(cd ../iptools; make)

# remove all .o files, and run make clean in iptools folder
# run to fix issues compiling errors like "undefined reference to ..."
clean:
	(cd ../iptools; make clean)
	$(RM) $(BIN_DIR)$(EXEC);
	find "../" -type f -name '*.o' -exec rm {} +

# ORDER OF ARGUMENTS IS VERY IMPORTANT - especially libraries
# g++, compiler flags, output, includes, libs
DO_EXEC = g++ $(CCFLAGS) -o $(BIN_DIR)$@ $@.cpp $(OPENCVINCLUDE) $(LIBS) 

#------ now compile
$(EXEC):   		iptools
				$(DO_EXEC)
