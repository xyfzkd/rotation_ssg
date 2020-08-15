#include <cufftw.h>
#include <cufft.h>
#include "cuda_runtime.h"
//#include "src/acc/cuda/cuda_alignPatch.h"
#include "src/acc/acc_alignPatch.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "device_launch_parameters.h"

#define GPU

#define pi 3.1415926535
#define LENGTH 100000 //signal sampling points


void CuFFT::inverseFourierTransform(
        MultidimArray<fComplex>& src,
        MultidimArray<float>& dest)
{
#ifdef TEST
    float Data[LENGTH] = { 1,2,3,4 };
    float fs = 1000000.000;//sampling frequency
    float f0 = 200000.00;// signal frequency
    for (int i = 0; i < LENGTH; i++)
    {
        Data[i] = 1.35*cos(2 * pi*f0*i / fs);//signal gen,
    }

    cufftComplex *CompData = (cufftComplex*)malloc(LENGTH * sizeof(cufftComplex));//allocate memory for the data in host
    int i;
    for (i = 0; i < LENGTH; i++)
    {
        CompData[i].x = Data[i];
        CompData[i].y = 0;
    }

    cufftComplex *d_fftData;
    cudaMalloc((void**)&d_fftData, LENGTH * sizeof(cufftComplex));// allocate memory for the data in device
    cudaMemcpy(d_fftData, CompData, LENGTH * sizeof(cufftComplex), cudaMemcpyHostToDevice);// copy data from host to device

    cufftHandle plan;// cuda library function handle
    cufftPlan1d(&plan, LENGTH, CUFFT_C2C, 1);//declaration
    cufftExecC2C(plan, (cufftComplex*)d_fftData, (cufftComplex*)d_fftData, CUFFT_FORWARD);//execute
    cudaDeviceSynchronize();//wait to be done
    cudaMemcpy(CompData, d_fftData, LENGTH * sizeof(cufftComplex), cudaMemcpyDeviceToHost);// copy the result from device to host

    for (i = 0; i < LENGTH / 2; i++)
    {
        printf("i=%d\tf= %6.1fHz\tRealAmp=%3.1f\t", i, fs*i / LENGTH, CompData[i].x*2.0 / LENGTH);
        printf("ImagAmp=+%3.1fi", CompData[i].y*2.0 / LENGTH);
        printf("\n");
    }
    cufftDestroy(plan);
    free(CompData);
    cudaFree(d_fftData);
#endif
#ifdef GPU
    if (!areSizesCompatible(dest, src))
    {
        resizeRealToMatch(dest, src);
    }

    MultidimArray<fComplex> src2 = src;

    std::vector<int> N(0);
    if (dest.zdim > 1) N.push_back(dest.zdim);
    if (dest.ydim > 1) N.push_back(dest.ydim);
    N.push_back(dest.xdim);
    /* https://docs.nvidia.com/cuda/cufft/index.html#cufftdoublecomplex 4.2.1 */
    cufftHandle planIn;
    cufftComplex *comp_data;
    cufftReal *real_data;

//    if (cudaGetLastError() != cudaSuccess){
//        fprintf(stderr, "Cuda error: Failed to allocate\n");
//        return;
//    }

    cudaMalloc((void**)&real_data, sizeof(cufftComplex)*N[0]*N[1]);
    cudaMalloc((void**)&comp_data, sizeof(cufftComplex)*N[0]*(N[1]/2+1));


    cudaMemcpy(comp_data, (cufftComplex*) MULTIDIM_ARRAY(src2), sizeof(cufftComplex)*N[0]*(N[1]/2+1), cudaMemcpyHostToDevice);
    printf("nihoa\n");
    cudaDeviceSynchronize();//wait to be done
    cudaMemcpy(real_data, (cufftReal*) MULTIDIM_ARRAY(dest), sizeof(cufftComplex)*N[0]*N[1], cudaMemcpyHostToDevice);

    /* Create a 2D FFT plan. */
    cufftPlan2d(&planIn,  N[0], N[1], CUFFT_C2R);

    /* https://stackoverflow.com/questions/16511526/cufft-and-fftw-data-structures-are-cufftcomplex-and-fftwf-complex-interchangabl
     * Are cufftComplex and fftwf_complex interchangable? yes!
     */

    /* https://docs.nvidia.com/cuda/cufft/index.html 3.9.3 */

    cufftExecC2R(planIn, comp_data, real_data);

    cudaMemcpy(MULTIDIM_ARRAY(dest),real_data, sizeof(cufftComplex)*N[0]*N[1], cudaMemcpyDeviceToHost);

    cudaFree(comp_data);
    cudaFree(real_data);
#endif
}
