// nvcc hello-world.cu -L /usr/local/cuda/lib -lcudart -o hello-world
#include <cstdlib>
#include <cstdio>
#include "Matching.h"
#include "const.h"
//#include "cam.h"
#include "WindowMatching.h"
#include "timer.h"
#include <cmath>

using namespace cv;

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__device__ float weighted[WINDOW_SIZE*WINDOW_SIZE];
__device__ float wa0[IM_SIZE]; 
__device__ float wa1[IM_SIZE]; 
__device__ float wac0[IMAGE_WIDTH*IMAGE_HEIGHT];
__device__ float wac1[IMAGE_WIDTH*IMAGE_HEIGHT];
//__device__ float ep0[EP_SIZE];
__device__ float* pos;

__global__ void weightedAverage(uchar const *im0, uchar const *im1) {
	// calculate weighted
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x < WINDOW_SIZE && threadIdx.y < WINDOW_SIZE) {
		int a = WINDOW_SIZE -1;
		float cos_x = cos(M_PI * (static_cast<int>(threadIdx.x) - WINDOW_SIZE/2) / a);
		float cos_y = cos(M_PI * (static_cast<int>(threadIdx.y) - WINDOW_SIZE/2) / a);
		weighted[threadIdx.y*WINDOW_SIZE+threadIdx.x] = cos_x*cos_x*cos_y*cos_y; 
		//printf("threadIdx.y %d, threadIdx.x %d: weight=%f \n", threadIdx.y, threadIdx.x, weighted[threadIdx.y*WINDOW_SIZE+threadIdx.x]);
	}
	__syncthreads();
	__shared__ uchar cache[(NUM_THREADS+WINDOW_SIZE-1)*(NUM_THREADS+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL];
	// load image values from global memory to shared memory
	uchar const *img = (blockIdx.z == 0) ? im0 : im1;
	for (int start_y = 0; start_y < blockDim.y+WINDOW_SIZE; start_y += blockDim.y) {
		for (int start_x = 0; start_x < blockDim.x+WINDOW_SIZE; start_x += (blockDim.x/NUM_COLOUR_CHANNEL)) {
			int maxThreadIdx_x = NUM_THREADS/NUM_COLOUR_CHANNEL * NUM_COLOUR_CHANNEL;
			if (threadIdx.x < maxThreadIdx_x && start_x*NUM_COLOUR_CHANNEL+threadIdx.x < (blockDim.x+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL
					&& start_y+threadIdx.y < blockDim.y+WINDOW_SIZE-1) {
				int idx_y = blockIdx.y * blockDim.y + (start_y + threadIdx.y) - WINDOW_SIZE/2;
				int idx_x = blockIdx.x * blockDim.x * NUM_COLOUR_CHANNEL + (start_x*NUM_COLOUR_CHANNEL+threadIdx.x) - WINDOW_SIZE/2*NUM_COLOUR_CHANNEL;
				cache[(start_y+threadIdx.y)*(blockDim.x+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL + (start_x*NUM_COLOUR_CHANNEL+threadIdx.x)] = (idx_y >= 0 && idx_y < IMAGE_HEIGHT
					&& idx_x >= 0 && idx_x < IMAGE_WIDTH*NUM_COLOUR_CHANNEL) ? img[idx_y*IMAGE_WIDTH*NUM_COLOUR_CHANNEL+idx_x] : 0;
			}
		}
	}
	__syncthreads();

	// calculate weighted average
	//*
	int idx_y = blockIdx.y*blockDim.y+threadIdx.y;
	int idx_x = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx_y < IMAGE_HEIGHT && idx_x < IMAGE_WIDTH) {
		float tmp_r = 0.0;
		float tmp_g = 0.0;
		float tmp_b = 0.0;
		for (int i = 0; i < WINDOW_SIZE; ++i)
			for (int j = 0; j < WINDOW_SIZE; ++j) {
				float weight = weighted[i*WINDOW_SIZE+j];
				int cache_idx_base = (threadIdx.y+i)*(blockDim.x+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL 
					+ (threadIdx.x+j)*NUM_COLOUR_CHANNEL;
				tmp_r += (weight * cache[cache_idx_base]);
				tmp_g += (weight * cache[cache_idx_base+1]);
				tmp_b += (weight * cache[cache_idx_base+2]);
			}
		tmp_r /= 9;
		tmp_g /= 9;
		tmp_b /= 9;
		//*
		if (blockIdx.x==0 && blockIdx.y==0&&blockIdx.z==0&&threadIdx.x==3&&threadIdx.y==3) {
			printf("[%f, %f, %f] ", tmp_r, tmp_g, tmp_b);
		}//*/
		int write_to_idx = idx_y*IMAGE_WIDTH*NUM_COLOUR_CHANNEL + idx_x*NUM_COLOUR_CHANNEL;
		float *wa = (blockIdx.z == 0) ? wa0 : wa1;
		wa[write_to_idx] = tmp_r;
		wa[write_to_idx+1] = tmp_g;
		wa[write_to_idx+2] = tmp_b;//*/
		//*
		if (blockIdx.x==0 && blockIdx.y==0&&blockIdx.z==0&&threadIdx.x==3&&threadIdx.y==3) {
			printf("[%f, %f, %f] ", wa[write_to_idx], wa[write_to_idx+1], wa[write_to_idx+2]);
		}//*/
	}
}
//*
__global__ void wac_gpu(uchar const *im0, uchar const *im1) {
	__shared__ uchar cache[(NUM_THREADS+WINDOW_SIZE-1)*(NUM_THREADS+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL];
	// load image values from global memory to shared memory
	uchar const *img = (blockIdx.z == 0) ? im0 : im1;
	for (int start_y = 0; start_y < blockDim.y+WINDOW_SIZE; start_y += blockDim.y) {
		for (int start_x = 0; start_x < blockDim.x+WINDOW_SIZE; start_x += (blockDim.x/NUM_COLOUR_CHANNEL)) {
			int maxThreadIdx_x = NUM_THREADS/NUM_COLOUR_CHANNEL * NUM_COLOUR_CHANNEL;
			if (threadIdx.x < maxThreadIdx_x && start_x*NUM_COLOUR_CHANNEL+threadIdx.x < (blockDim.x+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL
					&& start_y+threadIdx.y < blockDim.y+WINDOW_SIZE-1) {
				int idx_y = blockIdx.y * blockDim.y + (start_y + threadIdx.y) - WINDOW_SIZE/2;
				int idx_x = blockIdx.x * blockDim.x * NUM_COLOUR_CHANNEL + (start_x*NUM_COLOUR_CHANNEL+threadIdx.x) - WINDOW_SIZE/2*NUM_COLOUR_CHANNEL;
				cache[(start_y+threadIdx.y)*(blockDim.x+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL + (start_x*NUM_COLOUR_CHANNEL+threadIdx.x)] = (idx_y >= 0 && idx_y < IMAGE_HEIGHT
					&& idx_x >= 0 && idx_x < IMAGE_WIDTH*NUM_COLOUR_CHANNEL) ? img[idx_y*IMAGE_WIDTH*NUM_COLOUR_CHANNEL+idx_x] : 0;
			}
		}
	}
	__syncthreads();

	/*
	if (blockIdx.x==0 && blockIdx.y==0&&threadIdx.x==0&&threadIdx.y==0) {
		for (int i=0; i<38; ++i) {
			for (int j=0; j<38; ++j)
				printf("%f ", wa0[i*IMAGE_WIDTH*NUM_COLOUR_CHANNEL+j*NUM_COLOUR_CHANNEL]);
			printf("\n");
		}			
	}//*/
	// calculate weighted average correlation
	//*
	int idx_y = blockIdx.y*blockDim.y+threadIdx.y;
	int idx_x = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx_y < IMAGE_HEIGHT && idx_x < IMAGE_WIDTH) {
		float tmp = 0.0;
		int wa_idx_base = idx_y*IMAGE_WIDTH*NUM_COLOUR_CHANNEL + idx_x*NUM_COLOUR_CHANNEL;
		for (int i = 0; i < WINDOW_SIZE; ++i)
			for (int j = 0; j < WINDOW_SIZE; ++j) {
				int cache_idx_base = (threadIdx.y+i)*(blockDim.x+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL 
					+ (threadIdx.x+j)*NUM_COLOUR_CHANNEL;
				float *wa = (blockIdx.z == 0) ? wa0 : wa1;
				tmp += (weighted[i*WINDOW_SIZE+j] * ((cache[cache_idx_base]-wa[wa_idx_base]) * (cache[cache_idx_base]-wa[wa_idx_base])
					+ (cache[cache_idx_base+1]-wa[wa_idx_base+1]) * (cache[cache_idx_base+1]-wa[wa_idx_base+1])
					+ (cache[cache_idx_base+2]-wa[wa_idx_base+2]) * (cache[cache_idx_base+2]-wa[wa_idx_base+2])));
				
			}
		//*
		if (blockIdx.x==0 && blockIdx.y==0&&blockIdx.z==0&&threadIdx.x==3&&threadIdx.y==3) {
			printf("[wa:%f, tmp:%f]\n", wa0[wa_idx_base], tmp);
		}//*/
		int write_to_idx = idx_y*IMAGE_WIDTH + idx_x;
		float *wac = (blockIdx.z == 0) ? wac0 : wac1;
		wac[write_to_idx] = tmp;//*/

	}

}

__host__ __device__ float biLinInt(int x1, int x2, int y1, int y2, float x, float y,
		float q11, float q21, float q12, float q22) {
	if (y2 == y1) {
		if (x2 == x1)
			return q11;
		else
			return 1/(x2-x1) * (q11*(x2-x) + q21*(x-x1));
	} else if (x2 == x1) {
		return 1/(y2-y1) * (q11*(y2-y) + q12*(y-y1));
	} else
		return 1/(x2-x1)/(y2-y1) * (q11*(x2-x)*(y2-y) + q21*(x-x1)*(y2-y) +
				q12*(x2-x)*(y-y1) + q22*(x-x1)*(y-y1));
}

__global__ void wcc(uchar const *im0, uchar const  *im1, float *depth) {
	// load an image block of 38x38 belongs to rerefence image into shared memory
	__shared__ uchar cache0[(NUM_THREADS+WINDOW_SIZE-1)*(NUM_THREADS+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL];
	// TODO: copy loading code to weightedAverage() and wac_gpu()
	for (int start_y = 0; start_y < blockDim.y+WINDOW_SIZE; start_y += blockDim.y) {
		for (int start_x = 0; start_x < blockDim.x+WINDOW_SIZE; start_x += (blockDim.x/NUM_COLOUR_CHANNEL)) {
			int maxThreadIdx_x = NUM_THREADS/NUM_COLOUR_CHANNEL * NUM_COLOUR_CHANNEL;
			if (threadIdx.x < maxThreadIdx_x && start_x*NUM_COLOUR_CHANNEL+threadIdx.x < (blockDim.x+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL
					&& start_y+threadIdx.y < blockDim.y+WINDOW_SIZE-1) {
				int idx_y = blockIdx.y * blockDim.y + (start_y + threadIdx.y) - WINDOW_SIZE/2;
				int idx_x = blockIdx.x * blockDim.x * NUM_COLOUR_CHANNEL + (start_x*NUM_COLOUR_CHANNEL+threadIdx.x) - WINDOW_SIZE/2*NUM_COLOUR_CHANNEL;
				cache0[(start_y+threadIdx.y)*(blockDim.x+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL + (start_x*NUM_COLOUR_CHANNEL+threadIdx.x)] = (idx_y >= 0 && idx_y < IMAGE_HEIGHT
					&& idx_x >= 0 && idx_x < IMAGE_WIDTH*NUM_COLOUR_CHANNEL) ? im0[idx_y*IMAGE_WIDTH*NUM_COLOUR_CHANNEL+idx_x] : 0;
			}
		}
	}
	__syncthreads();
	/*
	if (blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0&&threadIdx.y==0) {
		for (int i=0; i<38; ++i) {
			for (int j=0; j<38; ++j)
				printf("%d ", cache0[i*(blockDim.x+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL+j*NUM_COLOUR_CHANNEL]);
			printf("\n");
		}			
	}
	//*/

	// define global best score and global pos_x, pos_y
	float best_score = 0.0f;
	float pos_x = 0.0f;
	float pos_y = 0.0f;
	// calculate indices for later use
	int idx_y = blockIdx.y*blockDim.y+threadIdx.y;
	int idx_x = blockIdx.x*blockDim.x+threadIdx.x;
	int wa0_idx_base = idx_y*IMAGE_WIDTH*NUM_COLOUR_CHANNEL + idx_x*NUM_COLOUR_CHANNEL;
	int wac0_idx = idx_y*IMAGE_WIDTH + idx_x;
	__shared__ uchar cache1[(NUM_THREADS+WINDOW_SIZE-1)*(NUM_THREADS+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL];	
	// alternatively load image block of 38x38 belongs to target image into shared memory
	for (int load_idx = 0; load_idx < ceil(static_cast<float>(IMAGE_WIDTH)/blockDim.x); ++load_idx) {
		for (int start_y = 0; start_y < blockDim.y+WINDOW_SIZE; start_y += blockDim.y) {
			for (int start_x = 0; start_x < blockDim.x+WINDOW_SIZE; start_x += (blockDim.x/NUM_COLOUR_CHANNEL)) {
				int maxThreadIdx_x = NUM_THREADS/NUM_COLOUR_CHANNEL * NUM_COLOUR_CHANNEL;
				if (threadIdx.x < maxThreadIdx_x && start_x*NUM_COLOUR_CHANNEL+threadIdx.x < (blockDim.x+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL
						&& start_y+threadIdx.y < blockDim.y+WINDOW_SIZE-1) {
					int idx_y = blockIdx.y * blockDim.y + (start_y + threadIdx.y) - WINDOW_SIZE/2;
					int idx_x = load_idx * blockDim.x * NUM_COLOUR_CHANNEL + (start_x*NUM_COLOUR_CHANNEL+threadIdx.x) - WINDOW_SIZE/2*NUM_COLOUR_CHANNEL;
					cache1[(start_y+threadIdx.y)*(blockDim.x+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL + (start_x*NUM_COLOUR_CHANNEL+threadIdx.x)] = (idx_y >= 0 && idx_y < IMAGE_HEIGHT
						&& idx_x >= 0 && idx_x < IMAGE_WIDTH*NUM_COLOUR_CHANNEL) ? im1[idx_y*IMAGE_WIDTH*NUM_COLOUR_CHANNEL+idx_x] : 0; //TODO: remember to recover im1
				}
			}
		}
		__syncthreads();

		/*
		if (blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0&&threadIdx.y==0&&load_idx==0) {
			for (int i=0; i<38; ++i) {
				for (int j=0; j<38; ++j) {
					printf("%d ", cache1[i*(blockDim.x+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL+j*NUM_COLOUR_CHANNEL]);
				}
				printf("\n");
			}			
		} 
		//*/
		if (idx_y < IMAGE_HEIGHT && idx_x < IMAGE_WIDTH) {
			// calculate the local best match along the epipolar line
			float local_best_score = 0.0f;
			float local_pos_x = 0.0f;
			float local_pos_y = 0.0f;
			int idx_x1;
			for (int target_offset = 0; target_offset < blockDim.x && (idx_x1=load_idx*blockDim.x + target_offset) < IMAGE_WIDTH; ++target_offset) {
				float tmp = 0.0f;
				int wa1_idx_base = idx_y*IMAGE_WIDTH*NUM_COLOUR_CHANNEL + idx_x1*NUM_COLOUR_CHANNEL;
				int wac1_idx = idx_y*IMAGE_WIDTH + idx_x1;
				for (int i = 0; i < WINDOW_SIZE; ++i)
					for (int j = 0; j < WINDOW_SIZE; ++j) {
						int cache0_idx_base = (threadIdx.y+i)*(blockDim.x+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL 
							+ (threadIdx.x+j)*NUM_COLOUR_CHANNEL;
						int cache1_idx_base = (threadIdx.y+i)*(blockDim.x+WINDOW_SIZE-1)*NUM_COLOUR_CHANNEL 
							+ (target_offset+j)*NUM_COLOUR_CHANNEL;					
						tmp += (weighted[i*WINDOW_SIZE+j] * ((cache0[cache0_idx_base]-wa0[wa0_idx_base]) * (cache1[cache1_idx_base]-wa1[wa1_idx_base])
							+ (cache0[cache0_idx_base+1]-wa0[wa0_idx_base+1]) * (cache1[cache1_idx_base+1]-wa1[wa1_idx_base+1])
							+ (cache0[cache0_idx_base+2]-wa0[wa0_idx_base+2]) * (cache1[cache1_idx_base+2]-wa1[wa1_idx_base+2])));
						
					}

				tmp /= sqrt(wac0[wac0_idx] * wac1[wac1_idx] * 9);

				// update local best match
				if (tmp > local_best_score) {
					local_best_score = tmp;
					local_pos_x = idx_x1;
					local_pos_y = idx_y;
				}
				/*
				if (blockIdx.x==0 && blockIdx.y==0&&threadIdx.x==3&&threadIdx.y==3) {
					printf("[local_best_score:%f, local_pos_x:%f, idx_x1:%d]\n", local_best_score, local_pos_x, idx_x1);
				}//*/
			}
			// update global best match
			if (local_best_score > best_score) {
				best_score = local_best_score;
				pos_x = local_pos_x;
				pos_y = local_pos_y;
			}
			/*
			if (blockIdx.x==0 && blockIdx.y==0&&threadIdx.x==3&&threadIdx.y==3) {
				printf("[best_score:%f, pos_x:%f, local_pos_x:%f]\n", best_score, pos_x, local_pos_x);
			}//*/
		}
	}

	if (idx_y < IMAGE_HEIGHT && idx_x < IMAGE_WIDTH)
		depth[wac0_idx] = (IMAGE_WIDTH - pos_x + idx_x) / (2*IMAGE_WIDTH);
	//*
	if (blockIdx.x==0 && blockIdx.y==0) {
		if (threadIdx.x == 30 && threadIdx.y == 30) {
			printf("[idx_y:%d, idx_x:%d, pos_x:%f]\n", idx_y, idx_x, pos_x);
		}		
	}//*/
}

int main(int argc, char** argv)
{	
	//*
	Mat imHost0 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat imHost1 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
	/*
	for (int i=349; i<375; ++i) {
		for (int j=445; j<450; ++j)
			std::cout<<imHost0.at<Vec3b>(i, j)<<" ";
		printf("\n");
	}
	//for (int i=0; i<36; ++i) {
	//	printf("%d ", imHost0.data[i]);
	//}//*/
	/*
	for (int i = 0; i < IMAGE_HEIGHT; ++i) {
		for (int j = 0; j < 255; ++j) {
			imHost0.at<Vec3b>(i, j) = Vec3b(1*(j+0), 1*(j+0), 1*(j+0));
			imHost1.at<Vec3b>(i, j) = Vec3b(1*(j+0), 1*(j+0), 1*(j+0));
		}

	}//*/

	uchar* im0; 
	uchar* im1; 
	HANDLE_ERROR(cudaMalloc((void**) &im0, IM_SIZE*sizeof(uchar)));
	HANDLE_ERROR(cudaMalloc((void**) &im1, IM_SIZE*sizeof(uchar)));
	/*
	float* score;
	HANDLE_ERROR(cudaMalloc((void**) &score, WAC_SIZE*sizeof(float)));
	float* pos;
	HANDLE_ERROR(cudaMalloc((void**) &pos, POS_SIZE*sizeof(float)));//*/
	float* depth;
	HANDLE_ERROR(cudaMalloc((void**) &depth, IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(float)));
	GpuTimer timer;
	timer.Start();
	//uchar imHostptr0[IM_SIZE];
	//uchar imHostptr1[IM_SIZE];
	//memcpy(imHostptr0, imHost0.data, IM_SIZE*sizeof(uchar));
	//memcpy(imHostptr1, imHost1.data, IM_SIZE*sizeof(uchar));
	// Note that following 2 lines do not always work, need to use intermediate pointers
	HANDLE_ERROR(cudaMemcpy(im0, imHost0.data, IM_SIZE * sizeof(uchar), cudaMemcpyHostToDevice)); 
	HANDLE_ERROR(cudaMemcpy(im1, imHost1.data, IM_SIZE * sizeof(uchar), cudaMemcpyHostToDevice));
	dim3 dimGrid(ceil(static_cast<float>(IMAGE_WIDTH)/NUM_THREADS), ceil(static_cast<float>(IMAGE_HEIGHT)/NUM_THREADS), 2);
	dim3 dimBlock(NUM_THREADS, NUM_THREADS);
	weightedAverage<<<dimGrid, dimBlock>>>(im0, im1);
	wac_gpu<<<dimGrid, dimBlock>>>(im0, im1);
	dimGrid = dim3(ceil(static_cast<float>(IMAGE_WIDTH)/NUM_THREADS), ceil(static_cast<float>(IMAGE_HEIGHT)/NUM_THREADS));
	wcc<<<dimGrid, dimBlock>>>(im0, im1, depth);
	timer.Stop();
	Mat depthIm(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32F, Scalar(0.0));
	//float *tdepth = new float[WAC_SIZE];
	HANDLE_ERROR(cudaMemcpy(depthIm.data, depth, sizeof(float)*IMAGE_WIDTH*IMAGE_HEIGHT, cudaMemcpyDeviceToHost));
	printf("\n");
	depthIm.convertTo(depthIm, CV_8U, 255.0);
	printf("Your code ran in: %f msecs.\n", timer.Elapsed());

	imshow("depth image", depthIm);
	waitKey(0);
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	imwrite("output_gpu.png", depthIm, compression_params);

	HANDLE_ERROR(cudaFree(im0));
	HANDLE_ERROR(cudaFree(im1));
	im0 = NULL;
	im1 = NULL;
	//HANDLE_ERROR(cudaFree(score));
	HANDLE_ERROR(cudaFree(depth));//*/

	return 0;
}
