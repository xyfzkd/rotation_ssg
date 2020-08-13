#include "acc_alignPatch.h"


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
    if (d > 1) N.push_back(d);
    if (h > 1) N.push_back(h);
    N.push_back(w);

    /* https://docs.nvidia.com/cuda/cufft/index.html#cufftdoublecomplex 4.2.1 */
    cufftHandle planIn;
    cufftComplex *data;

    cudaMalloc((void**)&data, sizeof(cufftComplex)*N[0]*(N[1]/2+1));
    if (cudaGetLastError() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to allocate\n");
        return;
    }

    /* Create a 2D FFT plan. */
    cufftPlan2D(&planIn,  N[0], N[1], CUFFT_C2R);

    /* https://stackoverflow.com/questions/16511526/cufft-and-fftw-data-structures-are-cufftcomplex-and-fftwf-complex-interchangabl
     * Are cufftComplex and fftwf_complex interchangable? yes!
     */

    /* https://docs.nvidia.com/cuda/cufft/index.html 3.9.3 */

    if (cufftExecC2R(p.getBackward(),(cufftComplex*) MULTIDIM_ARRAY(src2), MULTIDIM_ARRAY(dest)) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        return;
    }
    cudaFree(data);
}