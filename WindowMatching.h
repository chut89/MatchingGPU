/*
 * matching.h
 *
 *  Created on: 25.06.2015
 *      Author: chu
 */

#ifndef SRC_WINDOWMATCHING_H_
#define SRC_WINDOWMATCHING_H_

#include "opencv2/opencv.hpp"
#include <cmath>
#include "const.h"

class WindowMatching {
	cv::Mat wmat0;
	cv::Mat wmat1;
public:
	double w[WINDOW_SIZE][WINDOW_SIZE];
	WindowMatching(cv::Mat wmat0, cv::Mat wmat1) : wmat0(wmat0), wmat1(wmat1) {}
	WindowMatching();
	double weighted(int i, int j);
	cv::Vec3d weightedAverage(const cv::Mat& wmat);
	double wac(const cv::Mat& wmat, const cv::Vec3d& wa);
	double wcc(const cv::Mat& wmat0, const cv::Mat& wmat1, const cv::Vec3d& wa0,
			const cv::Vec3d& wa1, const double& wac0, const double& wac1);
	double* getw();
};
#endif /* SRC_MATCHING_H_ */
