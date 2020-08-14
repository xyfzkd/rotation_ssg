#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

#define COLS 3
#define ROWS 3

extern "C"    void iteration_mat1()
{
    cufftComplex *result_temp_din = (cufftComplex*)malloc(COLS*ROWS * sizeof(cufftComplex));
    cufftHandle p;
    / / Enter the assignment data
    for (size_t j = 0; j < ROWS; j++)
    {
        for (size_t i = 0; i < COLS; i++)
        {
            result_temp_din[i + j*COLS].x = (i + 1)*(j + 1);
            cout << result_temp_din[i + j*COLS].x << " ";
            result_temp_din[i + j*COLS].y = 0;
        }
    }
    cout << endl;

    size_t pitch;

    cufftComplex *t_result_temp_din;
    cudaMallocPitch((void**)&t_result_temp_din, &pitch, COLS * sizeof(cufftComplex), ROWS);

    cufftComplex *t_result_temp_out;
    cudaMallocPitch((void**)&t_result_temp_out, &pitch, COLS * sizeof(cufftComplex), ROWS);

    / / Add the value to the Device
    //cudaMemcpy2D(t_result_temp_din, pitch, result_temp_din, COLS * sizeof(cufftComplex), COLS * sizeof(cufftComplex), ROWS, cudaMemcpyHostToDevice);
    cudaMemcpy(t_result_temp_din,result_temp_din,  ROWS * sizeof(cufftComplex)* COLS, cudaMemcpyHostToDevice);

    //forward fft to make transformation rules
    cufftPlan2d(&p, ROWS, COLS, CUFFT_C2C);

    / / Perform the transformation
    cufftExecC2C(p, (cufftComplex*)t_result_temp_din, (cufftComplex*)t_result_temp_out, CUFFT_FORWARD);

    / / The value is attached to the host
    cudaMemcpy(result_temp_din,  t_result_temp_out, ROWS * sizeof(cufftComplex)* COLS, cudaMemcpyDeviceToHost);
    //cudaMemcpy2D(result_temp_din, pitch, t_result_temp_out, COLS * sizeof(cufftComplex), sizeof(cufftComplex)* ROWS, COLS, cudaMemcpyDeviceToHost);


    / / Extract the real and imaginary parts
    for (size_t j = 0; j < ROWS; j++)
    {
        for (size_t i = 0; i < COLS; i++)
        {
            Cout << result_temp_din[i + j*COLS].x << " ";//Real
            Cout << result_temp_din[i + j*COLS].y << endl;// imaginary part
        }
    }

}

