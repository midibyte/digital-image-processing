#all:
	#cd ../iptools; make
#	g++ -o bin/test test.cpp  -L ../lib -l iptools 
#	g++ -o bin/test2 test2.cpp  -L ../lib -l iptools -l mpeg 

	# source files.

BIN_DIR=bin/

EXEC = iptool

CCFLAGS = -std=c++11
CCFLAGSDEBUG := -std=c++11 -g -Wall -pedantic -Wextra

#opencv library
# OPENCVLIBS := -L /apps/opencv/4.0.1/lib64 -lopencv_core -lopencv_highgui -lopencv_imgcodexs -lopencv-objdetect -Wl,-rpath=/apps/opencv/4.0.1/lib64
OPENCVLIBS := -L /usr/local/Cellar/opencv/4.5.3_2/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect -lopencv_imgproc

# OPENCVLIBS := 

# include opencv directory
# OPENCVINCLUDE := -I/apps/opencv/4.0.1/include/opencv4
OPENCVINCLUDE:= -I/usr/local/Cellar/opencv/4.5.3_2/include/opencv4



all: $(EXEC)

test: 
	(cd ../iptools; make)
	g++ -std=c++14 -g -O3 $(OPENCVLIBS) $(OPENCVINCLUDE) -o $(BIN_DIR)$@ $@.cpp -L ../lib -l iptools

iptools:
	(cd ../iptools; make)

# remove all .o files
# run to fix issues compiling errors like "undefined reference to ..."
clean:
	$(RM) $(BIN_DIR)$(EXEC);
	find "../" -type f -name '*.o' -exec rm {} +


DO_EXEC = g++ $(CCFLAGS) $(OPENCVLIBS) $(OPENCVINCLUDE) -o $(BIN_DIR)$@ $@.cpp  -L ../lib -l iptools

#------ now compile
$(EXEC):   		iptools
				$(DO_EXEC)
