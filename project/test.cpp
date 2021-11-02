#include "../iptools/core.h"
#include <strings.h>
#include <string.h>
#include <opencv2/opencv.hpp>


using namespace std;

// maxlen of strings
#define MAXLEN 1024
#define TESTCOUNT 4

// conversion tester

void RGB_HSI_tester()
{
    // test HSI RGB conversion 

    // srand(time(NULL));

    int testCount = 4;
    // unsigned int R[testCount], G[testCount], B[testCount];
    // double H[testCount], S[testCount], I[testCount];

    // set known correct values

    unsigned int _R[TESTCOUNT] = {61, 0, 222, 165};
    unsigned int _G[TESTCOUNT] = {254, 0, 74, 55};
    unsigned int _B[TESTCOUNT] = {164, 0, 40, 35};

    double _H[TESTCOUNT] = {152, 0, 11, 9};
    double _S[TESTCOUNT] = {0.62, 0.0, 0.64, 0.59};
    double _I[TESTCOUNT] = {0.63, 0.0, 0.44, 0.33};


    for (int i = 0; i < TESTCOUNT; ++ i)
    {
        // unsigned int R[], G, B;

        // R = G = B = 0;

        // R = rand() % 255;
        // G = rand() % 255;
        // B = rand() % 255;
        int R, G, B;
        double H, S, I;

        R = _R[i];
        G = _G[i];
        B = _B[i];

        H = _H[i];
        S = _S[i];
        I = _I[i];



        printf("Testing RGB to HSI and HSI to RGB\n");
        printf("original RGB = (%d, %d, %d)\n", R, G, B);
        printf("correct HSI = (%f, %f, %f)\n", H, S, I);
        

        RGB_pixel testPixel{.R=R, .G=G, .B=B};

        HSI_pixel testPixelResult;

        testPixelResult = utility::RGB_to_HSI(testPixel);

        printf("RGB to HSI conversion: %f %f %f\n", testPixelResult.H, testPixelResult.S, testPixelResult.I);

        RGB_pixel testPixelHSItoRGB;

        testPixelHSItoRGB = utility::HSI_to_RGB(testPixelResult);
        
        printf("HSI to RGB conversion: %d %d %d\n", testPixelHSItoRGB.R, testPixelHSItoRGB.G, testPixelHSItoRGB.B);
        printf("===================================================\n");
    }
}

int main (int argc, char** argv)
{
    printf("test program start\n");


    RGB_HSI_tester();


    // return 0;


    // read into image
    image test;
    char inFile[] = "./img/baboon.pgm";
    test.read(inFile);

    int rows = test.getNumberOfRows();
    int cols = test.getNumberOfColumns();

    // cv::Mat ocvImage(rows, cols, cv::int8_t);
    cv::Mat cvImage = cv::imread(inFile, 0);
    
    std::cout << cvImage.rows << '\n' << cvImage.cols << "\n";

    printf("%d %d\n", cvImage.rows, cvImage.cols);


    // for (int row = 0; row < test.getNumberOfRows(); ++row)
    // for (int col = 0; col < test.getNumberOfColumns(); ++col)
    // {

    // }

}