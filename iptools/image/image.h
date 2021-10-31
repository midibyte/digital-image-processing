#ifndef IMAGE_H
#define IMAGE_H
#include <iostream>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>

using namespace std;

enum channel{RED, GREEN, BLUE, THETA, GRADIENT, GREY=RED, GRAY=GREY};

struct imageData
{
   vector<int>  redChannel, greenChannel, blueChannel; 
   // stores extra pixel info when using edgedetect
   vector<double> gradient, theta;
   int numRows, numColumns; 
};

class image 
{
private:
   imageData data;
   int getint(FILE *fp);
   
public:
   image();
   image(image &img);
   image(int rows, int columns);
   ~image();
   
   void deleteImage();
   void copyImage(image &img);
   void resize (int numberOfRows, int numberOfColumns);
   void setNumberOfRows(int rows);
   void setNumberOfColumns(int columns);
   void setPixel(const int row, const int col, const int value);
   void setPixel(const int row, const int col, const int rgb, const int value);

   int rowColToIdx(const int row, const int col, const int colCount);

   // template<typename T>
   void setPixelMeta(const int row, const int col, const int channel, const double);
   
   int getPixel(const int row, const int col);
   int getPixel(const int row, const int col, const int rgb);
   int getNumberOfRows();
   int getNumberOfColumns(); 
   vector<int>* getChannel(int rgb);

   // template<typename T>
   double getPixelMeta(const int row, const int col, const int channel);
   
   bool setChannel(int rgb, vector<int> &channel);   
   bool save (char* file);
   bool save (const char* file);
   bool read (char* file);
   bool isInbounds (const int row, const int col);


};

#endif

