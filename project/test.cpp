#include "../iptools/core.h"
#include <strings.h>
#include <string.h>

using namespace std;

// maxlen of strings
#define MAXLEN 1024

// conversion tester

void RGB_HSI_tester()
{
    // test HSI RGB conversion 

    // srand(time(NULL));

    int testCount = 4;
    // unsigned int R[testCount], G[testCount], B[testCount];
    // double H[testCount], S[testCount], I[testCount];

    // set known correct values

    unsigned int _R[testCount] = {61, 0, 222, 165};
    unsigned int _G[testCount] = {254, 0, 74, 55};
    unsigned int _B[testCount] = {164, 0, 40, 35};

    double _H[testCount] = {152, 0, 11, 9};
    double _S[testCount] = {0.62, 0.0, 0.64, 0.59};
    double _I[testCount] = {0.63, 0.0, 0.44, 0.33};


    for (int i = 0; i < testCount; ++ i)
    {
        // unsigned int R[], G, B;

        // R = G = B = 0;

        // R = rand() % 255;
        // G = rand() % 255;
        // B = rand() % 255;
        unsigned int R, G, B;
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


    return 0;
}