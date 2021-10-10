#all:
	#cd ../iptools; make
#	g++ -o bin/test test.cpp  -L ../lib -l iptools 
#	g++ -o bin/test2 test2.cpp  -L ../lib -l iptools -l mpeg 

	# source files.

BIN_DIR=bin/

EXEC = iptool


all: $(EXEC)

test: 
	(cd ../iptools; make)
	g++ -O3 -o $(BIN_DIR)$@ $@.cpp -L ../lib -l iptools

iptools:
	(cd ../iptools; make)

# remove all .o files
# run to fix issues compiling errors like "undefined reference to ..."
clean:
	$(RM) $(BIN_DIR)$(EXEC);
	find "../" -type f -name '*.o' -exec rm {} +


DO_EXEC = g++ -O3 -g -o $(BIN_DIR)$@ $@.cpp -L ../lib -l iptools

#------ now compile
$(EXEC):   		iptools
				$(DO_EXEC)
