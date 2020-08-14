#include <cufftw.h>
#include <cufft.h>
#include "cuda_runtime.h"
//#include "src/acc/cuda/cuda_alignPatch.h"
#include "src/acc/acc_alignPatch.h"


void CuFFT::inverseFourierTransform(
        MultidimArray<fComplex>& src,
        MultidimArray<float>& dest)
{
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
    cudaMemcpy(real_data, MULTIDIM_ARRAY(dest), sizeof(cufftComplex)*N[0]*N[1], cudaMemcpyHostToDevice);

    /* Create a 2D FFT plan. */
    cufftPlan2d(&planIn,  N[0], N[1], CUFFT_C2R);

    /* https://stackoverflow.com/questions/16511526/cufft-and-fftw-data-structures-are-cufftcomplex-and-fftwf-complex-interchangabl
     * Are cufftComplex and fftwf_complex interchangable? yes!
     */

    /* https://docs.nvidia.com/cuda/cufft/index.html 3.9.3 */

    if (cufftExecC2R(planIn, comp_data, real_data) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        return;
    }
    cudaFree(comp_data);
    cudaFree(real_data);
}