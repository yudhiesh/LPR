//
//  main.cpp
//  opencv
//
//  Created by Yudhiesh Ravindranath on 20/05/2019.
//  Copyright © 2019 Yudhiesh Ravindranath. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include <tesseract/baseapi.h>
//#include <leptonica/allheaders.h>

using namespace std;
using namespace cv;

Mat converttogrey(Mat RGBImg)
{
    Mat greyimg = Mat::zeros(RGBImg.size(), CV_8UC1); //1 = Grey(1 channel) , 3 = RGB(3 channel)
    for (int i = 0; i < RGBImg.rows; i++)
        for (int j = 0; j < RGBImg.cols * 3; j = j + 3)
            greyimg.at<uchar>(i, j / 3) = ((RGBImg.at<uchar>(i, j)) + (RGBImg.at<uchar>(i, j + 1)) + (RGBImg.at<uchar>(i, j + 2))) / 3;
    return greyimg;
}

Mat converttobinary(Mat binaryimg, float TH)
{
    Mat bin_img = Mat::zeros(binaryimg.size(), CV_8UC1);
    
    for (int i = 0; i < binaryimg.rows; i++)
    {
        for (int j = 0; j < binaryimg.cols; j++)
        {
            if (binaryimg.at<uchar>(i, j) > TH)
            {
                bin_img.at<uchar>(i, j) = 255;
            }
        }
    }
    return bin_img;
}
//
//void calculateTotalPixel(Mat plate)
//{
//    int countWhite = 0, countBlack = 0;
//    int countTotal;
//    Mat calPixel = Mat::zeros(plate.size(), CV_8UC1);
//    for (int i = 0; i < plate.rows; i++)
//    {
//        for (int j = 0; j < plate.cols; j++)
//        {
//            if (plate.at<uchar>(i, j) == 255)
//            {
//                countWhite++;
//            }
//            else if(plate.at<uchar>(i, j) == 0)
//            {
//                countBlack++;
//            }
//        }
//    }
//
//    return calPixel;
//}

Mat EqualizeHisto(Mat eqgrey)
{
    int count[256] = { 0 };
    float prob[256] = { 0.0 };
    float accprob[256] = { 0.0 };
    int newpixel[256] = { 0 };
    
    Mat eqhisto = Mat::zeros(eqgrey.size(), CV_8UC1);
    
    for (int i = 0; i < eqgrey.rows; i++)//count, adding into array
    {
        for (int j = 0; j < eqgrey.cols; j++)
        {
            count[eqgrey.at<uchar>(i, j)]++;
        }
    }
    
    for (int k = 0; k < 256; k++)//probablity
    {
        prob[k] = (float)count[k] / (float)(eqgrey.rows * eqgrey.cols);
    }
    
    accprob[0] = prob[0];
    
    for (int a = 1; a < 256; a++)//acc probability
    {
        accprob[a] = (prob[a] + accprob[a - 1]);
    }
    
    for (int i = 0; i < 256; i++)//Multiply to find NEW PIXEL
    {
        newpixel[i] = accprob[i] * 255;
    }
    
    for (int i = 0; i < eqgrey.rows; i++)//Change Old image to new pixel image
    {
        for (int j = 0; j < eqgrey.cols; j++)
        {
            eqhisto.at<uchar>(i, j) = newpixel[eqgrey.at<uchar>(i, j)];
        }
    }
    //return
    return eqhisto;
}

Mat dilation(Mat binary, int winsize)
{
    Mat dilate_img = Mat::zeros(binary.size(), CV_8UC1);
    for (int i = winsize; i < (binary.rows - winsize); i++)
    {
        for (int j = winsize; j < (binary.cols - winsize); j++)
        {
            for (int ii = (-winsize); ii <= (+winsize); ii++)
            {
                for (int jj = (-winsize); jj <= (+winsize); jj++)
                {
                    if (binary.at<uchar>(i + ii, j + jj) == 255)
                    {
                        dilate_img.at<uchar>(i, j) = 255;
                    }
                }
            }
        }
        
    }
    return dilate_img;
}

Mat Erosion(Mat Dilation, int winsize)
{
    Mat erosion_img = Mat::zeros(Dilation.size(), CV_8UC1);
    
    for (int i = winsize; i < (Dilation.rows - winsize); i++)
    {
        for (int j = winsize; j < (Dilation.cols - winsize); j++)
        {
            int newwhite = 0;
            for (int ii = (-winsize); ii <= (+winsize); ii = ii + 2)
            {
                for (int jj = (-winsize); jj <= (+winsize); jj = jj + 2)
                {
                    
                    if (Dilation.at<uchar>(i + ii, j + jj) == 255)
                    {
                        newwhite++;
                    }
                }
            }
            int newwinsize = ((winsize + 1)*(winsize + 1));
            if (newwhite == newwinsize)
            {
                erosion_img.at<uchar>(i, j) = 255;
            }
        }
        
    }
    return erosion_img;
}
Mat Blur(Mat grey, int winsize)
{
    Mat blurimg = Mat::zeros(grey.size(), CV_8UC1);
    for (int i = winsize; i < (grey.rows - winsize); i++)
    {
        for (int j = winsize; j < (grey.cols - winsize); j++)
        {
            int sum = 0;
            
            for (int ii = (-winsize); ii <= (+winsize); ii++)
            {
                for (int jj = (-winsize); jj <= (+winsize); jj++)
                {
                    sum += grey.at<uchar>(i + ii, j + jj);
                }
                blurimg.at<uchar>(i, j) = sum / (((winsize * 2) + 1) * ((winsize * 2) + 1)); // ((winsize*2)+1)^2;
                
            }
        }
    }
    return blurimg;
}

Mat Edgedetection(Mat Blur)
{
    Mat edgedetect = Mat::zeros(Blur.size(), CV_8UC1);
    double avgR, avgL;
    for (int i = 1; i < (Blur.rows - 1); i++)
    {
        for (int j = 1; j < (Blur.cols - 1); j++)
        {
            avgR = (Blur.at<uchar>(i, j + 1) + Blur.at<uchar>(i + 1, j + 1) + Blur.at<uchar>(i - 1, j + 1)) / 3;
            avgL = (Blur.at<uchar>(i, j - 1) + Blur.at<uchar>(i + 1, j - 1) + Blur.at<uchar>(i - 1, j - 1)) / 3;
            if (abs(avgR - avgL) > 40)
            {
                edgedetect.at<uchar>(i, j) = 255;
            }
            else
            {
                edgedetect.at<uchar>(i, j) = 0;
            }
        }
    }
    return edgedetect;
}

float OTSU(Mat binarize_plate)
{
    int count[256] = { 0 };
    float prob[256] = { 0.0 };
    float accprob[256] = { 0.0 };
    float sigma[256] = { 0.0 };
    float meu[256] = { 0.0 };
    
    Mat otsu1 = Mat::zeros(binarize_plate.size(), CV_8UC1);
    
    for (int i = 0; i < binarize_plate.rows; i++)//count, adding into array
    {
        for (int j = 0; j < binarize_plate.cols; j++)
        {
            count[binarize_plate.at<uchar>(i, j)]++;
        }
    }
    
    for (int k = 0; k < 256; k++)//probablity
    {
        prob[k] = (float)count[k] / (float)(binarize_plate.rows * binarize_plate.cols);
    }
    
    accprob[0] = prob[0];
    meu[0] = 0;
    
    for (int a = 1; a < 256; a++)//acc probability
    {
        accprob[a] = (prob[a] + accprob[a - 1]);
    }
    
    for (int q = 1; q < 256; q++) //meu
    {
        meu[q] = (q * (prob[q])) + meu[q - 1];
        
    }
    
    
    for (int w = 1; w < 256; w++) //sigma
    {
        sigma[w] = pow(meu[255] * accprob[w] - meu[w], 2) / (accprob[w] * (1 - accprob[w]));
    }
    float otsuval = 0;
    float max = -1;
    for (int t = 0; t < 256; t++)
    {
        
        if (max < sigma[t])
        {
            max = sigma[t];
            otsuval = t;
        }
    }
    //return
    return otsuval + 65; // binarize value lower = thicker, higher = thinner //50
}

Mat revertcolor(Mat platepic)
{
    Mat revertplatepic = Mat::zeros(platepic.size(), CV_8UC1);
    
    for (int i = 0; i < platepic.rows; i++)
    {
        for (int j = 0; j < platepic.cols; j++)
        {
            if (platepic.at<uchar>(i, j) == 255)
            {
                revertplatepic.at<uchar>(i, j) = 0;
            }
            else
            {
                revertplatepic.at<uchar>(i, j) = 255;
            }
        }
    }
    return revertplatepic;
}

int main()
{
    string folderpath = "/Users/yudhiesh/Desktop/mm/Dataset\(1\)\\(2\)/Dataset";
    vector<string> filesname;
    glob(folderpath, filesname);
    //grey > equalize historgram > blur > edge detection
    for (size_t i = 0; i < filesname.size(); i++)
    {
        Mat img = imread(filesname[i]);
        imshow("color_image", img);
        
        Mat grey = converttogrey(img);                        //converttogrey is function in lib. Use img because is RGB to GREY
        imshow("grey_image", grey);
        
        Mat eqhistogram = EqualizeHisto(grey);                //equalize histogram with grey image
        imshow("equalizeHistogram_image", eqhistogram);
        int a = 1;
        do
        {
            Mat blurry = Blur(eqhistogram, a);                    //blur image with equalize histogram
            
            Mat edgeD = Edgedetection(blurry);                    //edgedetection with blur image
            
            int t = 3;
            do
            {
                Mat dilate = dilation(edgeD, t);
                Mat Ero = Erosion(dilate, 1);
                imshow("Erosion", Ero);
                Mat Blob;
                Blob = Ero.clone();
                
                vector<vector<Point> > contours1;
                vector<Vec4i> hierachy1;
                findContours(dilate, contours1, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
                
                Mat platepic = Mat::zeros(grey.size(), CV_8UC3);
                
                if (!contours1.empty())
                {
                    for (int i = 0; i < contours1.size(); i++)
                    {
                        Scalar colour((rand() & 255), (rand() & 255),
                                      (rand() & 255));
                        drawContours(platepic, contours1, i, colour, -1, 8, hierachy1);
                    }
                }
                
                Mat plate;
                Rect rect_first;
                Scalar black = CV_RGB(0, 0, 0);
                for (size_t i = 0; i < contours1.size(); i++)
                {
                    rect_first = boundingRect(contours1[i]);
                    double density = (double)contourArea(contours1[i]) / (double)rect_first.area();
                    int ratio = rect_first.width / rect_first.height;
                    
                    if (rect_first.width < 45 || rect_first.width > 300 || rect_first.height < 25 || rect_first.height > 75 ||
                        rect_first.x > (Blob.rows * 0.8) || rect_first.x < (Blob.rows * 0.2) || rect_first.y >(Blob.cols * 0.8) || rect_first.y < (Blob.cols * 0.2))
                    {
                        drawContours(Blob, contours1, i, black, -1, 8, hierachy1);
                        
                    }
                    else
                    {
                        plate = grey(rect_first);
                        cout << rect_first.width << endl;
                        cout << rect_first.height << endl;
                        cout << rect_first.x << endl;
                        cout << rect_first.y << endl;
                    }
                }
                if (plate.empty())
                {
                    t = t + 2;
                    if (t == 7)
                    {
                        a--;
                        continue;
                    }
                    else
                    {
                        continue;
                    }
                }
                
                
                
                Mat equalize_plate = EqualizeHisto(plate);
                float iBin = OTSU(equalize_plate);
                Mat binarize_plate = converttobinary(equalize_plate, iBin);
                
                
                
                Mat Blob1;
                Blob1 = binarize_plate.clone();
                
                vector<vector<Point> > contours2;
                vector<Vec4i> hierachy2;
                findContours(binarize_plate, contours2, hierachy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
                
                Mat dst1 = Mat::zeros(equalize_plate.size(), CV_8UC3);
                
                if (!contours2.empty())
                {
                    for (int u = 0; u < contours2.size(); u++)
                    {
                        Scalar colour((rand() & 255), (rand() & 255),(rand() & 255));
                        drawContours(dst1, contours2, u, colour, -1, 8, hierachy2);
                    }
                }
                
                //imshow("dst1", dst1);
                
                Mat plate1;
                Rect rect_first1;
                Scalar black1 = CV_RGB(0, 0, 0);
                int count = 0;
                for (size_t v = 0; v < contours2.size(); v++)
                {
                    rect_first1 = boundingRect(contours2[v]);
                    
                    if (rect_first1.width < 4 || rect_first1.width > 50 || rect_first1.height < 10 )
                    {
                        drawContours(Blob1, contours2, v, black1, -1, 8, hierachy2);
                    }
                    else
                    {
                        count++;
                        ostringstream name;
                        //ostringstream charname;
                        plate1 = plate(rect_first1);
                        
                        ////Mat resizing;
                        ////resize(plate1, resizing, Size(),2,2);
                        
                        
                        //name << "Image" << count;
                        //imshow(name.str(), plate1);
                        //imwrite("carplate\\" + name.str() + ".png", plate1);
                        
                    }
                    
                }
                
                
                ostringstream qwerty;
                ostringstream ascending_carplate;
                Mat reverting = revertcolor(Blob1);
                imshow("reverted pic", reverting);
                imshow("color_image", img);
                qwerty <<"dilation" << t;
                ascending_carplate << "blur_image" << a;
                imwrite("carplate\\carplatenumber.png", reverting);
                imshow(ascending_carplate.str(), blurry);
                imshow("edgedetection", edgeD);
                imshow(qwerty.str(), dilate);
                //imshow("dst", platepic);/
                imshow("blob_img", Blob);
                imshow("plate", plate);
                imshow("equalizex plate", equalize_plate);
                imshow("binary plate", binarize_plate);
                imshow("dst1", dst1);
                imshow("blob_img1", Blob1);
                
                waitKey();
                
//                char *outText;
//                
//                tesseract::TessBaseAPI * api = new tesseract::TessBaseAPI();
//                if (api->Init(NULL, "eng"))
//                {
//                    fprintf(stderr, "Could not initialize tesseract. \n");
//                    exit(1);
//                }
//                
//                //open input image with leptonica library
//                
//                PIX *image = pixRead("carplate\\carplatenumber.png");
//                api->SetImage(image);
//                api->SetSourceResolution(300);
//                api->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
//                outText = api->GetUTF8Text();
//                //if(outText )/
//                // printf("OCR output: \n%s", outText);
//                //Get OCR result
//                
//                
//                //Destrou used object and release memory
//                api->End();
//                delete[] outText;
//                pixDestroy(&image);
//                
//                a = -1;
//                destroyAllWindows();
//                break;
                
                
            } while (t <= 7);
            
            
            
        }while (a >= 0);
        
        //remove("carplate\\carplatenumber.png");
    }
    
    waitKey();
}


// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started:
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

