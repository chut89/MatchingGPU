/*
 * const.h
 *
 *  Created on: Jul 14, 2015
 *      Author: chu
 */

#ifndef CONST_H_
#define CONST_H_

#define WINDOW_SIZE 7 // needs to be odd
#define TRANS_X 10
#define FOCAL_LENGTH 5
#define MAX_GREY_VAL 255
#define NUM_COLOUR_CHANNEL 3
#define IMAGE_WIDTH 450
#define IMAGE_HEIGHT 375
#define IM_SIZE 506250 // IM_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 3
#define WA_SIZE 491508 // WA_SIZE = (IMAGE_HEIGHT-WINDOW_SIZE+1)*(IMAGE_WIDTH-WINDOW_SIZE+1)*3
#define WAC_SIZE 163836 // WAC_SIZE = (IMAGE_HEIGHT-WINDOW_SIZE+1) * (IMAGE_WIDTH-WINDOW_SIZE+1)
#define POS_SIZE 327672 // POS_SIZE = (IMAGE_HEIGHT-WINDOW_SIZE+1) * (IMAGE_WIDTH-WINDOW_SIZE+1) * 2
#define EP_SIZE 145486368 // EP_SIZE = (IMAGE_HEIGHT-WINDOW_SIZE+1) * (IMAGE_WIDTH-WINDOW_SIZE+1) * (IMAGE_WIDTH-WINDOW_SIZE+1) * 2
#define WCC_SIZE 72743184 // WCC_SIZE = (IMAGE_HEIGHT-WINDOW_SIZE+1) * (IMAGE_WIDTH-WINDOW_SIZE+1) * (IMAGE_WIDTH-WINDOW_SIZE+1)
#define NUM_THREADS 32 // number of threads per dimension in each block
#endif /* CONST_H_ */
