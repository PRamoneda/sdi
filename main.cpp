#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;

int main()
{
    VideoCapture capture("test2.webm");
    if (!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open file!" << endl;
        return 0;
    }

    Mat frame1, prvs;
    capture >> frame1;
    cvtColor(frame1, prvs, COLOR_BGR2GRAY);

    while(true){
        Mat frame2, next;
        capture >> frame2;
        if (frame2.empty())
            break;
        cvtColor(frame2, next, COLOR_BGR2GRAY);

        Mat flow(prvs.size(), CV_32FC2);
        calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        // visualization
        Mat flow_parts[2];
        split(flow, flow_parts);
        Mat magnitude, angle, magn_norm;
        cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
        normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
        angle *= ((1.f / 360.f) * (180.f / 255.f));

        //build hsv image
        Mat _hsv[3], hsv, hsv8, bgr, gray, binary;
        _hsv[0] = angle;
        _hsv[1] = Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magn_norm;
        merge(_hsv, 3, hsv);
        hsv.convertTo(hsv8, CV_8U, 255.0);
        cvtColor(hsv8, bgr, COLOR_HSV2BGR);

        cvtColor(bgr, gray, COLOR_BGR2GRAY);
        threshold(gray, binary, 20, 255, THRESH_BINARY);

        imshow("bgr", bgr);
        imshow("normal video", frame2);
        imshow("binary", binary);
        imshow("gray", gray);

        ///////////////////////

//        Mat canny_output, gray;
//        vector<vector<Point> > contours;
//        vector<Vec4i> hierarchy;
//
//        // bgr to gray scale
//        cvtColor( bgr, gray, COLOR_BGR2GRAY );
//
//        // detect edges using canny
//        Canny( gray, canny_output, 50, 150, 3 );
//
//        // find contours
//        findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
//
//        // get the moments
//        vector<Moments> mu(contours.size());
//        for( int i = 0; i<contours.size(); i++ )
//        { mu[i] = moments( contours[i], false ); }
//
//        // get the centroid of figures.
//        vector<Point2f> mc(contours.size());
//        for( int i = 0; i<contours.size(); i++)
//        { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }
//
//
//        // draw contours
//        Mat drawing(canny_output.size(), CV_8UC3, Scalar(255,255,255));
//        for( int i = 0; i<contours.size(); i++ )
//        {
//            Scalar color = Scalar(167,151,0); // B G R values
//            drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
//            circle( drawing, mc[i], 4, color, -1, 8, 0 );
//        }
//
//        // show the resultant image
//        namedWindow( "Contours", WINDOW_AUTOSIZE );
//        imshow( "Contours", drawing );
//        waitKey(0);



        ////////
        int keyboard = waitKey(1);
        if (keyboard == 'q' || keyboard == 27)
            break;

        prvs = next;
    }
}