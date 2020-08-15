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

#define DATASIZE 8
#define BATCH 2

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %dn", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


void CuFFT::inverseFourierTransform(
        MultidimArray<fComplex>& src,
        MultidimArray<float>& dest)
{
#ifdef TEST
    // --- Host side input data allocation and initialization
    cufftReal *hostInputData = (cufftReal*)malloc(DATASIZE*BATCH*sizeof(cufftReal));
    for (int i=0; i<BATCH; i++)
        for (int j=0; j<DATASIZE; j++) hostInputData[i*DATASIZE + j] = (cufftReal)(i + 1);

    // --- Device side input data allocation and initialization
    cufftReal *deviceInputData; gpuErrchk(cudaMalloc((void**)&deviceInputData, DATASIZE * BATCH * sizeof(cufftReal)));
    cudaMemcpy(deviceInputData, hostInputData, DATASIZE * BATCH * sizeof(cufftReal), cudaMemcpyHostToDevice);

    // --- Host side output data allocation
    cufftComplex *hostOutputData = (cufftComplex*)malloc((DATASIZE / 2 + 1) * BATCH * sizeof(cufftComplex));

    // --- Device side output data allocation
    cufftComplex *deviceOutputData; gpuErrchk(cudaMalloc((void**)&deviceOutputData, (DATASIZE / 2 + 1) * BATCH * sizeof(cufftComplex)));

    // --- Batched 1D FFTs
    cufftHandle handle;
    int rank = 1;                           // --- 1D FFTs
    int n[] = { DATASIZE };                 // --- Size of the Fourier transform
    int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
    int idist = DATASIZE, odist = (DATASIZE / 2 + 1); // --- Distance between batches
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
    int batch = BATCH;                      // --- Number of batched executions
    cufftPlanMany(&handle, rank, n,
                  inembed, istride, idist,
                  onembed, ostride, odist, CUFFT_R2C, batch);

    //cufftPlan1d(&handle, DATASIZE, CUFFT_R2C, BATCH);
    cufftExecR2C(handle,  deviceInputData, deviceOutputData);

    // --- Device->Host copy of the results
    gpuErrchk(cudaMemcpy(hostOutputData, deviceOutputData, (DATASIZE / 2 + 1) * BATCH * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    for (int i=0; i<BATCH; i++)
        for (int j=0; j<(DATASIZE / 2 + 1); j++)
            printf("%i %i %f %fn", i, j, hostOutputData[i*(DATASIZE / 2 + 1) + j].x, hostOutputData[i*(DATASIZE / 2 + 1) + j].y);

    cufftDestroy(handle);
    gpuErrchk(cudaFree(deviceOutputData));
    gpuErrchk(cudaFree(deviceInputData));
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

    cufftComplex *host_comp_data, *device_comp_data;
    cufftReal    *host_real_data, *device_real_data;

//    if (cudaGetLastError() != cudaSuccess){
//        fprintf(stderr, "Cuda error: Failed to allocate\n");
//        return;
//    }
    host_comp_data = (cufftComplex*) MULTIDIM_ARRAY(src2);
    host_real_data = MULTIDIM_ARRAY(dest);

    gpuErrchk(cudaMalloc((void**)&device_real_data, sizeof(cufftReal)*N[0]*N[1]));
    gpuErrchk(cudaMalloc((void**)&device_comp_data, sizeof(cufftComplex)*N[0]*(N[1]/2+1)));


    cudaMemcpy(device_comp_data, host_comp_data, sizeof(cufftComplex)*N[0]*(N[1]/2+1), cudaMemcpyHostToDevice);
    printf("nihoa\n");

    cufftHandle planIn;


    /* Create a 2D FFT plan. */
    cufftPlan2d(&planIn,  N[0], N[1], CUFFT_C2R);

    /* https://stackoverflow.com/questions/16511526/cufft-and-fftw-data-structures-are-cufftcomplex-and-fftwf-complex-interchangabl
     * Are cufftComplex and fftwf_complex interchangable? yes!
     */

    /* https://docs.nvidia.com/cuda/cufft/index.html 3.9.3 */

    cufftExecC2R(planIn, device_comp_data, device_real_data);

    cudaMemcpy(host_real_data, device_real_data, sizeof(cufftReal)*N[0]*N[1], cudaMemcpyDeviceToHost);

    cufftDestroy(planIn);
    gpuErrchk(cudaFree(device_comp_data));
    gpuErrchk(cudaFree(device_real_data));
#endif
}
