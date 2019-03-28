/*
 * PairMatching.h
 *
 *  Created on: Jul 14, 2015
 *      Author: chu
 */

#ifndef PAIRMATCHING_H_
#define PAIRMATCHING_H_

#include <cstring>
#include "WindowMatching.h"

class Matching {
	cv::Mat im0;
	cv::Mat im1;
	cv::Mat wa0;
	cv::Mat wac0;
	cv::Mat wa1;
	cv::Mat wac1;
	cv::Mat bestPos;
	WindowMatching windmatch;

public:
	// check if pass by reference is needed in copy constructor
	Matching(cv::Mat im0, cv::Mat im1, cv::Mat bestPos = cv::Mat(IMAGE_HEIGHT - WINDOW_SIZE + 1,
			IMAGE_WIDTH - WINDOW_SIZE + 1, CV_32FC2, cv::Scalar(0.0, 0.0))) : im0(im0), im1(im1), bestPos(bestPos) {}
	Matching(std::string s1, std::string s2);


	static std::vector<cv::Vec2d> findEpipolar(const cv::Vec2d& p, const cv::Vec3d& trans);
	static cv::Vec3d biLinInt(int x1, int x2, int y1, int y2, double x, double y,
			const cv::Vec3d& ll, const cv::Vec3d& lr, const cv::Vec3d& ul, const cv::Vec3d& ur);
	static cv::Mat buildWindowMat(double i_centre, double j_centre, const cv::Mat& im);
	void preCalculate();
	void findBestPos();
	cv::Mat calcDepthMap();
	cv::Mat findBestPos(const cv::Mat& im0, const cv::Mat& im1);
	cv::Mat calcDepthMap(const cv::Mat& bestPos);
	cv::Mat getIm0();
	cv::Mat getIm1();
	cv::Mat getBestPos();
};


#endif /* PAIRMATCHING_H_ */
