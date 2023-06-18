#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

void detectStainOnNormalBanknote() {
	// Phát hiện vết bẩn của tờ tiền b
	// Mat templ = imread("Monney/image.jpg", IMREAD_GRAYSCALE);
	// Mat sample = imread("Monney/image1.jpg", IMREAD_GRAYSCALE);
	Mat templ = imread("Monney/detectStainOnNormalBanknote/img.jpg", IMREAD_GRAYSCALE);
	Mat sample = imread("Monney/detectStainOnNormalBanknote/img_fake.jpg", IMREAD_GRAYSCALE);
	// Mat result = abs(sample - templ);
	Mat result;
	absdiff(sample, templ, result);

	threshold(result, result, 40, 255, THRESH_BINARY);
	imshow("template", templ);
	imshow("sample", sample);
	imshow("diff", result);
	cv::imwrite("Monney/detectStainOnNormalBanknote/ouput/img.jpg", templ);
	cv::imwrite("Monney/detectStainOnNormalBanknote/ouput/img_fake.jpg", sample);
	cv::imwrite("Monney/detectStainOnNormalBanknote/ouput/result.jpg", result);
	waitKey(0);
}

int getMaxPixel(Mat src) {
	int histogram[256] = {};
	int maxPos = 0;

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			histogram[src.at<uchar>(i, j)]++;
			if (histogram[src.at<uchar>(i, j)] > histogram[maxPos]) {
				maxPos = src.at<uchar>(i, j);
			}
		}
	return maxPos;
}

Mat scaleHistogram(Mat src, double rate) {
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {

			src.at<uchar>(i, j) = src.at<uchar>(i, j) * rate > 255 ? 255 : (src.at<uchar>(i, j) * rate < 0 ? 0 : src.at<uchar>(i, j) * rate);

		}
	return src;
}
void drawHistogram(const cv::Mat& image, string paths) {
    // Tính histogram
    cv::Mat histogram;
    int histSize = 256;  // Số lượng bin cho histogram
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::calcHist(&image, 1, nullptr, cv::Mat(), histogram, 1, &histSize, &histRange);
    
    // Vẽ biểu đồ histogram
    int histWidth = 512;
    int histHeight = 400;
    int binWidth = cvRound((double)histWidth / histSize);
    cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    
    cv::normalize(histogram, histogram, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    
    for (int i = 1; i < histSize; i++) {
        cv::line(histImage, cv::Point(binWidth * (i - 1), histHeight - cvRound(histogram.at<float>(i - 1))),
            cv::Point(binWidth * (i), histHeight - cvRound(histogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }
    // Lưu biểu đồ histogram thành tệp tin hình ảnh
    cv::imwrite(paths, histImage);
    // Hiển thị biểu đồ histogram
    // cv::namedWindow("Histogram", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Histogram", histImage);
    
    // cv::waitKey(0);
}
void detectStainOnBanknoteHasDifferentBrightness() {

	Mat templ = imread("Monney/detectStainOnBanknoteHasDifferentBrightness/img.jpg", IMREAD_GRAYSCALE);
	Mat sample = imread("Monney/detectStainOnBanknoteHasDifferentBrightness/img_fake.jpg", IMREAD_GRAYSCALE);
	imshow("sample", sample);
	drawHistogram(templ, "Monney/detectStainOnBanknoteHasDifferentBrightness/ouput/hist_img.jpg");
	drawHistogram(sample, "Monney/detectStainOnBanknoteHasDifferentBrightness/ouput/hist_img_fake.jpg");

	sample = scaleHistogram(sample, (double)getMaxPixel(templ) / getMaxPixel(sample));
	drawHistogram(sample, "Monney/detectStainOnBanknoteHasDifferentBrightness/ouput/hist_img_fake_custom.jpg");
	Mat result = abs(sample - templ);
	threshold(result, result, 40, 255, THRESH_BINARY);
	imshow("template", templ);
	imshow("diff", result);
	cv::imwrite("Monney/detectStainOnBanknoteHasDifferentBrightness/ouput/img.jpg", templ);
	cv::imwrite("Monney/detectStainOnBanknoteHasDifferentBrightness/ouput/img_fake.jpg", sample);
	cv::imwrite("Monney/detectStainOnBanknoteHasDifferentBrightness/ouput/result.jpg", result);
	waitKey(0);
}

int countHorTenPixel(Mat src, int r, int c, bool v)
{
	int count = 0;
	for (int i = c - 5; i < c + 5; i++)
		if (i >= 0 && i <= src.cols) {
			if (src.at<uchar>(r, i) <= 10 && !v) count++;
			else if (src.at<uchar>(r, i) >= 100 && v) count++;
		}
	return count;
}

Point2i getTopLeanPoint(Mat src) {
	for (int i = 1; i < src.rows; i++)
		for (int j = 1; j < src.rows; j++) {
			if ((int)src.at<uchar>(i, j) <= 10 && countHorTenPixel(src, i, j, true) > 5)
				return(Point2i(i, j));
		}

	return Point2i(0, 0);
}


Point2i getBotLeanPoint(Mat src) {
	for (int i = src.rows - 1; i >= 0; i--)
		for (int j = 1; j < src.rows; j++) {
			if ((int)src.at<uchar>(i, j) <= 10 && countHorTenPixel(src, i, j, true) > 5)
				return(Point2i(i, j));
		}

	return Point2i(0, 0);
}

int detectOrientation(Mat templ, Mat sample) {

	Mat shape;
	threshold(sample, shape, 250, 255, THRESH_BINARY);
	Point2i top = getTopLeanPoint(shape);
	Point2i bot = getBotLeanPoint(shape);
	if (top.x == 0 && top.y == 0) return 0;
	return (top.y < bot.y ? 1 : -1) * nearbyint((atan((double)abs(top.x - bot.x) / abs(top.y - bot.y)) - atan((double)templ.rows / templ.cols)) / CV_PI * 180);

}

Mat rotate(Mat src, double angle)
{
	Mat dst, dst1;
	Point2f pt(src.cols / 2., src.rows / 2.);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(src, dst, r, Size(src.cols, src.rows));
	warpAffine(src, dst1, r, Size(src.cols, src.rows), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));
	threshold(dst1, dst1, 250, 255, THRESH_BINARY);
	return dst + dst1;
}

Mat getTemplateArea(Mat src, Mat _template)
{
	Point2i topLeft = Point2i((src.rows - _template.rows) / 2, (src.cols - _template.cols) / 2);
	Point2i botRight = Point2i(topLeft.x + _template.rows, topLeft.y + _template.cols);

	return src(Range(topLeft.x, botRight.x), Range(topLeft.y, botRight.y));
}

void insertionSort(int window[])
{
	int temp, i, j;
	for (i = 0; i < 9; i++) {
		temp = window[i];
		for (j = i - 1; j >= 0 && temp < window[j]; j--)
		{
			window[j + 1] = window[j];
		}
		window[j + 1] = temp;
	}
}

Mat medianFilter(Mat src)
{
	int window[9];
	Mat dst = src.clone();

	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			dst.at<uchar>(y, x) = 0.0;

	for (int y = 1; y < src.rows - 1; y++)
		for (int x = 1; x < src.cols - 1; x++)
		{
			window[0] = src.at<uchar>(y - 1, x - 1);
			window[1] = src.at<uchar>(y, x - 1);
			window[2] = src.at<uchar>(y + 1, x - 1);
			window[3] = src.at<uchar>(y - 1, x);
			window[4] = src.at<uchar>(y, x);
			window[5] = src.at<uchar>(y + 1, x);
			window[6] = src.at<uchar>(y - 1, x + 1);
			window[7] = src.at<uchar>(y, x + 1);
			window[8] = src.at<uchar>(y + 1, x + 1);
			insertionSort(window);
			dst.at<uchar>(y, x) = window[4];
		}
	return dst;
}

void detectStainOnBanknoteHasBeenRotated() {

	Mat templ = imread("Monney/detectStainOnBanknoteHasBeenRotated/img.jpg", IMREAD_GRAYSCALE);
	Mat sample = imread("Monney/detectStainOnBanknoteHasBeenRotated/img_fake.jpg", IMREAD_GRAYSCALE);
	imshow("sample", sample);

	resize(sample, sample, Size(templ.cols, templ.rows), INTER_LINEAR);


	int angle = detectOrientation(templ, sample);
	sample = rotate(sample, angle);
	sample = getTemplateArea(sample, templ);


	Mat result = abs(sample - templ);
	result = medianFilter(result);
	threshold(result, result, 40, 255, THRESH_BINARY);
	imshow("template", templ);
	imshow("diff", result);
	cv::imwrite("Monney/detectStainOnBanknoteHasBeenRotated/ouput/img.jpg", templ);
	cv::imwrite("Monney/detectStainOnBanknoteHasBeenRotated/ouput/img_fake.jpg", sample);
	cv::imwrite("Monney/detectStainOnBanknoteHasBeenRotated/ouput/result.jpg", result);
	waitKey(0);
}

void detectStainOnBanknoteHasDifferentSize() {

	Mat templ = imread("Monney/detectStainOnBanknoteHasDifferentSize/img.jpg", IMREAD_GRAYSCALE);
	Mat sample = imread("Monney/detectStainOnBanknoteHasDifferentSize/img_fake.jpg", IMREAD_GRAYSCALE);
	imshow("sample", sample);

	resize(sample, sample, Size(templ.cols, templ.rows), INTER_LINEAR);

	Mat result = abs(sample - templ);
	result = medianFilter(result);
	threshold(result, result, 40, 255, THRESH_BINARY);
	imshow("template", templ);
	imshow("diff", result);
	waitKey(0);
	cv::imwrite("Monney/detectStainOnBanknoteHasDifferentSize/ouput/img.jpg", templ);
	cv::imwrite("Monney/detectStainOnBanknoteHasDifferentSize/ouput/img_fake.jpg", sample);
	cv::imwrite("Monney/detectStainOnBanknoteHasDifferentSize/ouput/result.jpg", result);

}

int menu() {

	cout << "Select a task:\n";
	cout << "1. Detect stain on normal banknote\n";
	cout << "2. Detect stain on banknote has different brightness\n";
	cout << "3. Detect stain on banknote has been rotated\n";
	cout << "4. Detect stain on banknote has different size\n";
	cout << "Selected Task: ";
	int r;
	cin >> r;
	return r;
}

int main() {
	switch (menu())
	{
	case 1:
		detectStainOnNormalBanknote();
		break;
	case 2:
		detectStainOnBanknoteHasDifferentBrightness();
		break;
	case 3:
		detectStainOnBanknoteHasBeenRotated();
		break;
	case 4:
		detectStainOnBanknoteHasDifferentSize();
		break;
	default:
		cout << "??????????????????????\n";
		break;
	}
	waitKey(0);
	return 0;
}