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
#include "time.h"
#include "src/multidim_array.h"

#define PRINTCOMP

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

/**********************************************************************/
/* function for simulate data for iFFT
 * input of type MultidimArray<fComplex> *, simulate random
 * data, and this function should be integrated into class MultidimArray
 * there is macro RELION_ALIGNED_MALLOC, with data simulator initRandom or others.
 * However, it seems as if it's designed for common type.
 * I decide to fix the omission.
 *
 *         T* ptr=NULL;
 *         long int n;
 *         if (mode == "uniform")
 *             FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY_ptr(*this,n,ptr)
 *             *ptr = static_cast< T >(rnd_unif(op1, op2));
 *
 *  what if T is fComplex?
 */
/***********************************************************************/

void rand_comp(MultidimArray<fComplex>& s){
    T* ptr=NULL;
    long int n;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY_ptr(*s,n,ptr)
    (*ptr).real = static_cast< T >(rnd_unif(op1, op2));
    (*ptr).imag = static_cast< T >(rnd_unif(op1, op2));

#ifdef PRINTCOMP
    for (int i=0; i < 16; i++){
        printf("%3.1f %3.1f \n", *s.data[i].real,  *s.data[i].imag)
    }
#endif
}

/*******************************************************/
/* CuFFT for 2D image c2r realization
 * param@ src: MultidimArray<fComplex>, using array part
 * param@ dest: MultidimArray<float>, using array part
 */
/*******************************************************/

void CuFFT::inverseFourierTransform(
        MultidimArray<fComplex>& src,
        MultidimArray<float>& dest)
{
    /* http://www.orangeowlsolutions.com/archives/1173 arct
     * https://docs.nvidia.com/cuda/cufft/index.html#cufftdoublecomplex 4.2.1
     * https://docs.nvidia.com/cuda/cufft/index.html 3.9.3
     * https://www.beechwood.eu/using-cufft/ time
     * */
    if (!areSizesCompatible(dest, src))
    {
        resizeRealToMatch(dest, src);
    }

    MultidimArray<fComplex> src2 = src;

    std::vector<int> N(0);
    if (dest.zdim > 1) N.push_back(dest.zdim);
    if (dest.ydim > 1) N.push_back(dest.ydim);
    N.push_back(dest.xdim);

    float elapsedTime = 0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    cufftComplex *host_comp_data, *device_comp_data;
    cufftReal    *host_real_data, *device_real_data;

    /* https://stackoverflow.com/questions/16511526/cufft-and-fftw-data-structures-are-cufftcomplex-and-fftwf-complex-interchangabl
     * Are cufftComplex and fftwf_complex interchangable? yes!
     */
    host_comp_data = (cufftComplex*) MULTIDIM_ARRAY(src2);
    host_real_data = MULTIDIM_ARRAY(dest);

    gpuErrchk(cudaMalloc((void**)&device_real_data, sizeof(cufftReal)*N[0]*N[1]));
    gpuErrchk(cudaMalloc((void**)&device_comp_data, sizeof(cufftComplex)*N[0]*(N[1]/2+1)));


    cudaMemcpy(device_comp_data, host_comp_data, sizeof(cufftComplex)*N[0]*(N[1]/2+1), cudaMemcpyHostToDevice);

    cufftHandle planIn;


    /* Create a 2D FFT plan. */
    cufftPlan2d(&planIn,  N[0], N[1], CUFFT_C2R);


    cufftExecC2R(planIn, device_comp_data, device_real_data);

    cudaMemcpy(host_real_data, device_real_data, sizeof(cufftReal)*N[0]*N[1], cudaMemcpyDeviceToHost);

    cufftDestroy(planIn);
    gpuErrchk(cudaFree(device_comp_data));
    gpuErrchk(cudaFree(device_real_data));

    //GET CALCULATION TIME
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);
    printf("CUFFT Calculation COMPLETED IN : % 5.3f ms \n",elapsedTime);
}
